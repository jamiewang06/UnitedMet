import numpy as np
import pandas as pd
import torch
import pickle
from statsmodels.stats.multitest import multipletests

def count_obs(data, n_batch, J, batch_index_vector):
    def count_obs_batch(data):
        return np.sum(~np.isnan(data), axis=0)

    n_obs = np.zeros(shape=[n_batch, J]).astype(int)
    for b in range(n_batch):
        n_obs[b, :] = count_obs_batch(data[batch_index_vector == b])
    return n_obs

# Order and Rank the normalized data (directly order, decreasing)
def order_and_rank(data, n_obs, N, J, n_batch, batch_index_vector):
    # offset = 0
    orders = np.zeros([N, J]).astype(int)
    ranks = np.zeros([N, J]).astype(int)
    for b in range(n_batch):
        batch = data[batch_index_vector == b]
        batch = np.where(np.isnan(batch), 0, batch)  # important, replace nan with 0
        indices = batch.argsort(axis=0,
                                kind='stable')  # Returns the indices that would sort an array in ascending order.
        orders[batch_index_vector == b] = indices[::-1]  # order: indices of rank 1 item, rank 2 item, etc.
        # (order of tied value is overwhelmingly behaving weird)
        ranks_temp = batch.argsort(axis=0, kind='stable')[::-1].argsort(
            axis=0, kind='stable')  # rank (largest item has rank 0, second largest has rank 1, etc.)
        # if rank > n_obs, set rank = n_obs (get censored rank)
        ranks[batch_index_vector == b] = np.where(ranks_temp > n_obs[b, :], n_obs[b, :], ranks_temp)
        # offset = offset + data[batch_index_vector == b].shape[0]
    return orders, ranks

def test_train_split_met_rna(met_data, start_row, stop_row, targets, n_obs, ranks, f_test_prop, targets_test_feature_indices_pre):
    """
    testing and training sets are only metabolomics data.
    """
    
    training = np.copy(met_data)
    testing = np.full(training.shape, np.nan)
    targets_test_sample_indices = []
    targets_test_feature_indices = []

    len_available = 0
    # Mask all metabolite data of the target batch in the training set
    # training and testing set only have metabolomics data
    
    for i, target in enumerate(targets):
        test_sample_indices = np.arange(start_row[target-1], stop_row[target-1])
        test_feature_indices = np.array(targets_test_feature_indices_pre[i])
        ixgrid = np.ix_(test_sample_indices, test_feature_indices)

        training[ixgrid] = np.nan
        print("test samples", test_sample_indices)
        print("test features", test_feature_indices)
        print("training: ", training)
        solo = np.all(np.isnan(training[:, test_feature_indices]), axis=0) # test features that are not measured in other batches
        missing = np.all(np.isnan(met_data[ixgrid]), axis=0) # test features missing in target batch
        #available_features = np.arange(met_data.shape[1])[(~missing) & (~solo)]
        available_test_features = test_feature_indices[(~missing) & (~solo)]
        print("available test features: ", available_test_features)
        len_available += len(available_test_features)
        test_feature_indices = np.random.choice(available_test_features,
                                                size=int(round(f_test_prop * len(available_test_features))), replace=False)
        # sort the indices in ascending order
        test_sample_indices = np.sort(test_sample_indices)
        test_feature_indices = np.sort(test_feature_indices)
        
        targets_test_sample_indices.append(test_sample_indices)
        targets_test_feature_indices.append(test_feature_indices)
        # Reset the n_obs, all test metabolites in the target batch are set to 0
        n_obs[target - 1, test_feature_indices] = 0
        for sidx in test_sample_indices:
            for fidx in test_feature_indices:
                testing[sidx, fidx] = ranks[sidx, fidx]
        
    print("na in training", np.sum(np.isnan(training)), training.size)
    print("not na in testing", np.sum(~np.isnan(testing)), testing.size)
    print("total available:", len_available)

    return testing, training, n_obs, targets_test_sample_indices, targets_test_feature_indices, len_available

def split_folds_across_isotope(data, n_batch, batch_index_vector, n_folds):  # modified for isotope
    """
    The difference from the cross_validation() in the PL_single_array_cv.py code is that:
    the folds are not the same length, thus in format of list of arrays but not ndarray
    """
    folds = [[] for _ in range(n_folds)]  # _ is "throwaway" variable name in python

    for bidx in range(n_batch):
        batch = data[batch_index_vector == bidx]
        missing = np.all(np.isnan(batch), axis=0)
        solo = np.all(np.isnan(data[batch_index_vector != bidx]),
                      axis=0)  # solo are the data that don't overlap with other datasets
        if n_batch == 2:  # modified for isotope
            solo = ~solo
        available = np.arange(batch.shape[1])[(~missing) & (~solo)]  # features for cross validation in each batch
        np.random.shuffle(available)  # shuffles the features
        splits = np.array_split(available, n_folds)  # Split an array into multiple sub-arrays
        for fold_idx in range(n_folds):
            folds[fold_idx].extend([splits[fold_idx]])
    if n_batch == 2:  # add the new crated batch to the folds
        for fold_idx in range(n_folds):
            if fold_idx != n_folds - 1:
                folds[fold_idx].extend([splits[fold_idx + 1]])
            else:
                folds[n_folds - 1].extend([splits[0]])
            # folds[i] is a list of arrays storing all the (#column) in the #batch array in that i+1 fold across datasets
    # folds = [np.array(fold) for fold in folds]
    # Couldn't use this line because the arrays in each folds are not the same length, thus cannot be converted to ndarray
    return folds

def gumbel_sampling_3D(x):
    """
    x: logits, 3D array
    vectorized gumbel re-parameterization ~ categorical sampling without replacement
    """
    G = np.random.gumbel(0, 1, size=(x.shape[0], x.shape[1], x.shape[2]))
    Z = -(x + G)  # normal prior
    return np.argsort(Z, axis=1, kind='stable')  # return the indices of from largest Z to smallest Z

def smart_perm_2D(x, permutation):
    """
        x: 2D tensor of logits
        permutation: 2D tensor of orders
        permutae columns of x according to the permutation (adapted from the github version where they permute rows)
    """
    assert x.size() == permutation.size()
    if x.ndimension() == 2:
        d1, d2 = x.size()
        x_permuted = x[
            permutation.flatten(),
            torch.arange(d2).unsqueeze(0).repeat((1, d1)).flatten()
        ].view(d1, d2)
    else:
        ValueError("Only 2 dimension expected")
    return x_permuted

def load_joint_targets_test_features(file_path, cancer_type):
    target_feature_dir = f"{file_path}/data/{cancer_type}_features.pkl"
    with open(target_feature_dir, "rb") as file:
        target_1_test_features, target_2_test_features = pickle.load(file)
    
    return [target_1_test_features, target_2_test_features]

def saving_data_for_pyro_posterior(embedding_dir, N, K, J, met_names, start_row, stop_row, batch_sizes):
    with open(f'{embedding_dir}/data_for_posterior_pyro.npy', 'wb') as f:  # 'wb': write as binary
        np.save(f, N)  # save arrays in numpy binary .npy files
        np.save(f, K)
        np.save(f, J)
        np.save(f, start_row)
        np.save(f, stop_row)
        np.save(f, batch_sizes)
        f.close()
    pd.DataFrame({'met_names': met_names}).to_csv(f'{embedding_dir}/met_names_pyro.csv')

def saving_data_for_testing(embedding_dir, testing, targets_test_sample_indices, targets_test_feature_indices, n_obs_pre, censor_indicator, targets):
    with open(f'{embedding_dir}/data_for_testing_pyro.npy', 'wb') as f:  # 'wb': write as binary
        np.save(f, testing)
        for i in range(len(targets)): #store each array separately since of potentially different lengths
            np.save(f, targets_test_sample_indices[i])
            np.save(f, targets_test_feature_indices[i])
        np.save(f, n_obs_pre)
        np.save(f, censor_indicator)
        f.close()

def correct_for_mult(data):
    data['p_adj'] = multipletests(pvals=data['pval'], method="fdr_bh", alpha=0.1)[1]
    return data

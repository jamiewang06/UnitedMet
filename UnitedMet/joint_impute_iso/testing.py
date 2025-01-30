import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr
from Performance_Benchmarking.scripts_isotope.utils_isotope import correct_for_mult
from sklearn.metrics import roc_auc_score, roc_curve
from Performance_Benchmarking.scripts_isotope.visualization import *

def testing_function_met_rna(testing, rank_hat_mean, rank_hat_std,
                     test_sample_indices, test_feature_indices,
                     n_obs_pre, met_names, censor_indicator, iter, target, f_test_prop):
    """
    The difference between this function and the version in the PL_across_array.cv.py file is that I added the
    feature_index_vec in the output dataframe. This is to make it easier to match the predictions to the actual values
    """
    actual_vec = testing[test_sample_indices, :][:, test_feature_indices].flatten('F')
    predicted_vec_temp = rank_hat_mean[test_sample_indices, :][:, test_feature_indices]
    predicted_vec_no_censor = predicted_vec_temp.flatten('F')
    predicted_vec_temp = np.where(predicted_vec_temp > n_obs_pre[target-1, test_feature_indices],
                                  n_obs_pre[target-1, test_feature_indices], predicted_vec_temp)
    predicted_vec = predicted_vec_temp.flatten('F')
    predicted_revec_temp = rank_hat_mean[test_sample_indices, :][:, test_feature_indices].argsort(
        axis=0, kind='stable').argsort(axis=0, kind='stable')  # re-rank the predicted values
    # if rank > n_obs, set rank = n_obs (get censored rank)
    predicted_revec_no_censor = predicted_revec_temp.flatten('F')
    predicted_revec_temp = np.where(
        predicted_revec_temp > n_obs_pre[target-1, test_feature_indices],
        n_obs_pre[target-1, test_feature_indices], predicted_revec_temp)
    predicted_revec = predicted_revec_temp.flatten('F')
    std_vec = rank_hat_std[test_sample_indices, :][:, test_feature_indices].flatten('F')
    test_feature_names = [met_names[i] for i in test_feature_indices]
    feature_vec = np.repeat(test_feature_names, len(test_sample_indices))
    feature_index_vec = np.repeat(test_feature_indices, len(test_sample_indices))
    sample_vec = np.tile(test_sample_indices, len(test_feature_indices))
    censor_vec = censor_indicator[test_sample_indices, :][:, test_feature_indices].flatten('F')
    actual_pred_df = pd.DataFrame(
        {'actual_rank': actual_vec, 'predicted_rank': predicted_vec, 'predicted_rerank': predicted_revec,
         'predicted_rank_no_censor': predicted_vec_no_censor, 'predicted_rerank_no_censor': predicted_revec_no_censor,
         'std': std_vec, 'feature': feature_vec,'feature_index': feature_index_vec, 'sample_index': sample_vec, 'censor': censor_vec})
    actual_pred_df['iteration'] = iter
    actual_pred_df['f_test_prop'] = f_test_prop
    return actual_pred_df

def results_analysis(actual_pred_res_df, batch_sizes, target, num_samples):
    cor = actual_pred_res_df.groupby(["f_test_prop", "iteration", "feature"])[['actual_rank', 'predicted_rerank']].corr(
        method='spearman').iloc[0::2, -1].to_frame()
    pval = actual_pred_res_df.groupby(["f_test_prop", "iteration", "feature"])[['actual_rank', 'predicted_rerank']].corr(
        method=lambda x, y: spearmanr(x, y)[1]).iloc[0::2, -1].to_frame()
    cor = cor.dropna(axis=0)
    pval = pval.dropna(axis=0)
    ave_std = actual_pred_res_df.groupby(["f_test_prop", "iteration", "feature"])[['std']].mean()
    censor_prop = actual_pred_res_df.groupby(["f_test_prop", "iteration", "feature"])[['censor']].mean()


    ########################################################## By iteration
    # obtain rho for each feature in each iteration
    by_iter_rho = (pd.merge(cor.rename(columns={"predicted_rerank": "rho"}),
                            pval.rename(columns={"predicted_rerank": "pval"}),
                            how='left',
                            on=['f_test_prop', 'iteration', 'feature'])
                   .dropna(axis=0)
                   .groupby(["f_test_prop", "iteration"], group_keys=False)
                   .apply(correct_for_mult)
                   # bin into significant/ns
                   .assign(sig=lambda dataframe: dataframe['p_adj'].map(lambda p_adj: True if p_adj < 0.1 else False))
                   ).merge(ave_std.rename(columns={"std": "ave_std"}), how='left',
                           on=['f_test_prop', 'iteration', 'feature']
                   ).merge(censor_prop.rename(columns={"censor": "censor_prop"}),
                                              how='left', on=['f_test_prop', 'iteration', 'feature'])


    # calculate median rho by iteration
    # first, z-transform and group appropriately
    g_median_by_iter = (by_iter_rho
                        .assign(z_score=lambda dataframe: dataframe['rho'].map(lambda rho: np.arctanh(rho)))
                        .groupby(["f_test_prop", "iteration"], group_keys=False)
                        )

    # compute median z-score and inverse z-transform to obtain median rho
    median_rho_iter = (pd.DataFrame({"median_z": g_median_by_iter.median()['z_score'],
                                     "sig_in": g_median_by_iter.sum()['sig'] / g_median_by_iter.size()})
                       .assign(median_rho=lambda dataframe: dataframe['median_z'].map(lambda z: np.tanh(z)))
                       )

    ########################################################## By feature
    # calculate median rho for each feature
    # once again, z-transform and group appropriately
    g_median_by_feature = (by_iter_rho
                           .assign(z_score=lambda dataframe: dataframe['rho'].map(lambda rho: np.arctanh(rho)))
                           .groupby(["f_test_prop", "feature"], group_keys=False))

    # compute median z-score and inverse z-transform to obtain median rho
    median_rho_feature = (pd.DataFrame({"median_z": g_median_by_feature.median()['z_score'],
                                        "sig_in": g_median_by_feature.sum()['sig'] / g_median_by_feature.size()})
                          .assign(median_rho=lambda dataframe: dataframe['median_z'].map(lambda z: np.tanh(z)))
                          .assign(ave_ave_std=lambda dataframe: g_median_by_feature.mean()['ave_std'])
                          .assign(censor_prop=lambda dataframe: g_median_by_feature.mean()['censor_prop'])
                          # flag well-predicted metabolites
                          .assign(sig_in_most=(lambda dataframe: dataframe['sig_in']
                                               # note that if a flag for "well-predicted metabolite" is desired, a condition for positive rho must be added here
                                               .map(lambda sig_in: True if sig_in > 0.90 else False)))
                          .reset_index()
                          )
    median_rho_feature['ave_ave_std'] = median_rho_feature['ave_ave_std'] / batch_sizes[target - 1]
    median_rho_feature['ave_ave_se'] = median_rho_feature['ave_ave_std'] / np.sqrt(num_samples)
    # I kept the names still to be standard error but it is actually standard deviation (calculating using rank percentage)

    n_pos_sig = median_rho_feature.loc[(median_rho_feature['median_rho'] > 0) & (
                median_rho_feature['sig_in_most'] == True), 'sig_in_most'].sum()
    ratio_pos_sig = n_pos_sig / median_rho_feature.shape[0]
    print(f'Testing metabolites that have positive rhos and significant p values: {n_pos_sig}')
    print(f'Total available testing metabolites : {median_rho_feature.shape[0]}')
    print(f'The percentage of testing metabolites with positive rhos and significant p values: {ratio_pos_sig * 100}%')
    print("Testing metabolites with spearman's rhos >= 0.3: ", median_rho_feature[median_rho_feature['median_rho'] >= 0.3].shape[0])

    return by_iter_rho, median_rho_iter, median_rho_feature

def results_visualization(median_rho_feature, median_rho_iter, actual_pred_res_df, f_test_prop, target, sub_dir, plots_dir):

    scatterplot_std(median_rho_feature, target, 'ave_ave_std', 'median_rho', 'std',sub_dir, plots_dir)
    scatterplot_std(median_rho_feature, target, 'ave_ave_se', 'median_rho', 'se', sub_dir, plots_dir)
    median_rho_iter_plot(median_rho_iter, plots_dir)
    print("Median spearman's rho among all metabolites", median_rho_iter['median_rho'].mean())
    median_rho_feature_plot(median_rho_feature, f_test_prop, plots_dir)
    sig_in_iter_plot(median_rho_feature, plots_dir)
    # scatter plot for the most well-predicted metabolite
    scatterplot_all_iter(actual_pred_res_df,
                     median_rho_feature.loc[median_rho_feature['median_rho'] == median_rho_feature['median_rho'].
                     max(), 'feature'].values[0], plots_dir, mode='iteration')
    scatterplot_all_iter(actual_pred_res_df,
                     median_rho_feature.loc[median_rho_feature['median_rho'] == median_rho_feature['median_rho'].
                     max(), 'feature'].values[0], plots_dir, mode='censor')
    predicted_rank_histogram(actual_pred_res_df, plots_dir)
    scatterplot_all(actual_pred_res_df, plots_dir, actual_pred_res_df)

def results_probabilistic_analysis(actual_pred_res_df, testing, rank_hat_draws, test_feature_indices,
                                   test_sample_indices, n_obs_pre, target, median_rho_feature, plots_dir, sub_dir):
    """
    Probabilistic analysis of posterior draws, compared with ground-truth values
    Added the reranking of posterior draws compared with previous version of code in MetabolicModel
    """
    # prepare the posterior rank_draws (re-ranked) and ground truth data for only testing features
    test_features = actual_pred_res_df['feature'].unique()
    testing = testing[test_sample_indices, :][:, test_feature_indices]
    rank_hat_draws_test = rank_hat_draws[:, test_sample_indices[0]: (test_sample_indices[-1] + 1), test_feature_indices]
    rank_hat_draws_test = rank_hat_draws_test.argsort(
        axis=1, kind='stable').argsort(axis=1, kind='stable')  # re-rank the predicted values
    # if rank > n_obs, set rank = n_obs (get censored rank)
    rank_hat_draws_test = np.where(
        rank_hat_draws_test > n_obs_pre[target-1, test_feature_indices],
        n_obs_pre[target-1, test_feature_indices], rank_hat_draws_test)
    # Calculate the probability of significant predictions and significant high abundance predictions
    testing = testing / test_sample_indices.shape[0]  # transformed all ranks to [0,1]
    rank_hat_draws_test = rank_hat_draws_test / test_sample_indices.shape[0]  # transformed all ranks to [0,1]
    rank_hat_upper = np.quantile(rank_hat_draws_test, 0.95, axis=0)
    rank_hat_lower = np.quantile(rank_hat_draws_test, 0.05, axis=0)
    rank_hat_range = rank_hat_upper - rank_hat_lower
    significant_predictions = rank_hat_range <= 0.5
    significant_high = np.quantile(rank_hat_draws_test, 0.9, axis=0) <= 0.3
    print("The average percentage of significant predictions of all metabolites: ",
          np.mean(np.sum(significant_predictions, axis=0) / significant_predictions.shape[0]))
    print("The median percentage of significant predictions of all metabolites: ",
          np.median(np.sum(significant_predictions, axis=0) / significant_predictions.shape[0]))

    # change the order of features to match the order of significant_predictions
    median_rho_feature.index = median_rho_feature['feature']
    median_rho_feature = median_rho_feature.loc[
    test_features]

    # Spearman's rho v.s. Percentage of significant predictions
    x = np.sum(significant_predictions, axis=0) / significant_predictions.shape[0]
    y = median_rho_feature['median_rho']
    plt.scatter(x, y)
    plt.xlabel('percentage of significant predictions')
    plt.ylabel('spearman rho')
    plt.savefig(f'{plots_dir}/scatterplot_median_rho_significant_predictions.pdf')
    # plt.savefig(f'{plots_dir}/scatterplot_standard_error_significant_predictions.pdf')
    plt.close()
    rho, pval = spearmanr(x, y)
    print(f"Spearman rho vs percentage of significant predictions: spearman's rho = {rho}, pval = {pval}")

    # Average standard error v.s. Percentage of significant predictions
    x = np.sum(significant_predictions, axis=0) / significant_predictions.shape[0]
    y = median_rho_feature['ave_ave_se']
    plt.scatter(x, y)
    plt.xlabel('percentage of significant predictions')
    plt.ylabel('standard error')
    plt.savefig(f'{plots_dir}/scatterplot_standard_error_significant_predictions.pdf')
    plt.close()
    rho, pval = spearmanr(x, y)
    print(f"Standard error vs percentage of significant predictions: spearman's rho = {rho}, pval = {pval}")

    # ROC curve
    # ---------------------------------------------------------------------------------------------------------
    cutoff = 0.1
    # true positive
    testing_high = (testing <= cutoff).astype(int)
    # probability of predicted positive
    rank_hat_test_high = np.sum((rank_hat_draws_test <= cutoff).astype(int), axis=0) / rank_hat_draws_test.shape[0]
    testing_high = testing_high.flatten(order='F')
    rank_hat_test_high = rank_hat_test_high.flatten(order='F')
    fpr, tpr, thresholds = roc_curve(testing_high, rank_hat_test_high, pos_label=1)
    auc_score = roc_auc_score(testing_high, rank_hat_test_high)
    print("AUC Score:", auc_score)
    plt.rcParams['figure.figsize'] = [6, 6]
    ax = plt.subplot(111)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({sub_dir[target - 1]})')
    plt.text(0.85, 0.02, "AUC = %.3f" % (auc_score), horizontalalignment='center',
             verticalalignment='center',
             transform=ax.transAxes)
    plt.savefig(f'{plots_dir}/roc_curve.pdf')
    plt.close()




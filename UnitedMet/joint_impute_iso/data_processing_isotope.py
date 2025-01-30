import numpy as np
import os
import pandas as pd
from itertools import repeat

def file_system_init_isotope(file_path, cancer_type, imputation, imputation_dir, dir, results_dir):  # modified for isotope
    # Input Metabolomics and Transcriptomics data directory
    rna_matched_data_dir = f"{file_path}/data/RNA_matched_{cancer_type}"
    met_matched_data_dir = f"{file_path}/data/MET_matched_{cancer_type}"
    rna_imputation_data_dir = f"{file_path}/data/RNA_matched_imputation"
    if not os.path.exists(rna_matched_data_dir) or not os.path.exists(met_matched_data_dir):
        raise OSError(f"Input Metabolomics and Transcriptomics data directory don't exist.")
    if imputation and not os.path.exists(rna_imputation_data_dir):
        raise OSError(f"Input RNA data directory for imputation doesn't exist.")

    # Define results directory and other sub-directories inside it
    if cancer_type == 'isotope':  # imputation of kidney cancer isotope data
        sub_dir = ['MITO']  # notice that mito is the first one
        if imputation:
            sub_dir = [imputation_dir] + sub_dir  # notice that imputation dir is placed the first one
    elif cancer_type == 'NSCLC_human':  # imputation of lung cancer isotope data
        sub_dir = ['NSCLC_human']
        if imputation:
            sub_dir = [imputation_dir] + sub_dir  # notice that imputation dir is placed the first one
    elif cancer_type == 'MITO':  # mito benchmarking
        sub_dir = ['MITO1', 'MITO2']
    elif cancer_type == 'NSCLC':  # NSCLC benchmarking (not good to train all tracers together)
        sub_dir = ['NSCLC1', 'NSCLC2']
    elif cancer_type == 'NSCLC_G6':  # NSCLC glucose tracer 6h
        sub_dir = ['NSCLC_G6_1', 'NSCLC_G6_2']
    elif cancer_type == 'NSCLC_Q6':  # NSCLC glutamine tracer 6h
        sub_dir = ['NSCLC_Q6_1', 'NSCLC_Q6_2']
    elif cancer_type == 'NSCLC_G24':  # NSCLC glucose tracer 24h
        sub_dir = ['NSCLC_G24_1', 'NSCLC_G24_2']
    elif cancer_type == 'NSCLC_Q24':  # NSCLC glutamine tracer 24h
        sub_dir = ['NSCLC_Q24_1', 'NSCLC_Q24_2']
    else:
        raise ValueError(f"Invalid cancer type: {cancer_type}")
    proportions = list(repeat(0, len(sub_dir)))
    target = sub_dir.index(dir) + 1
    proportions[target - 1] = 1

    plots_dir = f'{results_dir}/plots'
    embedding_dir = f'{results_dir}/embeddings'

    for dir in [plots_dir,  embedding_dir]:
        try:
            os.makedirs(dir)
        except FileExistsError:
            print(f"Directory '{dir}' already exists.")
    return rna_matched_data_dir, met_matched_data_dir, rna_imputation_data_dir, sub_dir, proportions, \
            plots_dir,  embedding_dir, target

def tic_normalization_across(data, batch_index_vector):
    normalized_data = np.copy(data)  # Note that, when we modify data, normalized_data changes
    n_batches = batch_index_vector.max() + 1
    for bidx in range(n_batches):  # range(i) include numbers from 0 to i-1
        batch_rows = np.arange(data.shape[0])[
            batch_index_vector == bidx]  # np.arange(n) is same as np.array(range(3))
        batch = data[batch_rows]  # get the separate dataframe for distinct batches
        nan_mask = np.isnan(batch)  # Test element-wise for NaN and return result as a boolean array
        missing = np.all(nan_mask, axis=0)  # boolean array about whether for a whole column is all nan in each row
        min_batch = np.nanmin(batch)  # Return minimum of an array, ignoring any NaNs
        for row in range(batch.shape[0]):
            n_censored = np.sum(nan_mask[row]) - np.sum(missing)  # remember to substrate missing numbers
            row_tic = np.nansum(batch[row]) + 0.5 * n_censored * min_batch
            batch[row, :] = batch[row, :] / row_tic
        batch = batch / np.nansum(batch, axis=1, keepdims=True)  # make sure the sum of each row is 1
        normalized_data[batch_rows] = batch
    return normalized_data  # missing values are still NAN but not filled with np.nanmin(batch) now

def count_iso_metabolites(isotopologues, iso_index):
    # setup initial values
    count = 1
    metabolite_count = []
    metabolite_count_map = {}
    for i in range(1, len(isotopologues)):
        if (isotopologues[i])[:-iso_index] == (isotopologues[i - 1])[:-iso_index]:
            count += 1
            if i == len(isotopologues) - 1:
                metabolite_count.append(count)
                metabolite_count_map[isotopologues[i - 1][:-iso_index]] = count
        else:
            metabolite_count.append(count)
            metabolite_count_map[(isotopologues[i - 1])[:-iso_index]] = count
            count = 1
    return metabolite_count, metabolite_count_map

def isotope_normalization(data, metabolite_map, reference_type):
    # Normalize the data by dividing the ratio of glucose m+6/all glucose isotopologues
    if reference_type == 'tissue_glucose':
        # need to drop some samples to make sure that there is no nan values in Glucose m+6 isotopologues
        # Normalize by Glucose m+6
        normalized_data = np.copy(data)
        for i in range(data.shape[1]):
            if i != metabolite_map['Glucose m+6']:
                normalized_data[:, i] = data[:, i] / (data[:, metabolite_map['Glucose m+6']])
    elif reference_type == 'tissue_pyruvate':
        # need to drop some samples to make sure that there is no nan values in Pyruvate m+3 isotopologues
        # Normalize by Pyruvate m+3
        normalized_data = np.copy(data)
        for i in range(data.shape[1]):
            if i != metabolite_map['Pyruvate m+3']:
                normalized_data[:, i] = data[:, i] / (data[:, metabolite_map['Pyruvate m+3']])
    return normalized_data  # missing values are still NAN but not filled with np.nanmin(batch) now

def FindVariableFeatures(data, names,n_batch, start_row, stop_row, nfeatures=2000):
    """
    input data: samples by features matrix
    input names: feature names
    """
    VMR_matrix = np.zeros((n_batch, data.shape[1]))
    for b in range(n_batch):
        batch = data[start_row[b]:stop_row[b], :]
        # calculate the Variance-Mean Ratio (index of dispersion) for each gene
        VMR = np.nanvar(batch, axis=0) / np.nanmean(batch, axis=0)
        # rank the genes by VMR (rank 0 is the most variable (largest) gene)
        VMR_matrix[b, :] = np.argsort(VMR, kind='stable')[::-1].argsort(kind='stable')
    VMR_median = np.median(VMR_matrix, axis=0)
    # find the top nfeatures most variable genes based on the median VMR rank across batches
    # smaller VMR rank means more variable
    variable_gene_indices = np.argsort(VMR_median, kind='stable')[:nfeatures]
    variable_gene_names = (np.array(names)[variable_gene_indices]).tolist()
    variable_data = data[:, variable_gene_indices]
    return variable_data, variable_gene_names

def data_normalization_cleaning_isotope(met_data, rna_data, metabolite_map,
                                gene_map, sample_map_rna, imputation, batch_index_vector, n_batch,
                                start_row, stop_row, remove_solo, reference_type,
                                cancer_type, variable_gene=False, nfeatures=2000):
    """
    Although the sum of each row are not all 1,000,000, the tpm matrices of RNA are already normalized
    Don't need to do: rna_data = rna_data / np.nansum(rna_data, axis=1, keepdims=True)
    """
    # TIC Normalize the MET data
    if cancer_type == 'isotope' or cancer_type == 'MITO' or cancer_type == 'NSCLC_human':
        met_data = isotope_normalization(met_data, metabolite_map, reference_type)
    if imputation:
        missing_data = np.full((len(batch_index_vector[batch_index_vector == 0]), len(metabolite_map)), np.nan)
        met_data = np.concatenate([missing_data, met_data], axis=0)
    if remove_solo == True:
        if not imputation:
            # when imputation, there is no need to remove solo metabolites.
            # set all solo metabolites to nan which will be removed later in this function
            met_data = filter_solo_met_data(met_data, n_batch, start_row, stop_row)

    ########################################################## Concatenate MET and RNA data
    data = np.concatenate([met_data, rna_data], axis=1)
    met_names = list(metabolite_map.keys())
    rna_names = list(gene_map.keys())
    if np.all(np.isnan(data), axis=0).sum() == 0:
        print("The aggregate data is clean.")
    else:
        print("Some features are NaN in all samples and need to be removed from the aggregate data.\
              If remove_solo == True, solo metabolites will also be removed in this step.")
        delete_index = np.where(np.all(np.isnan(data), axis=0))[0]
        met_delete_index = delete_index[delete_index < len(metabolite_map)]
        rna_delete_index = delete_index[delete_index >= len(metabolite_map)] - len(metabolite_map)
        # Update the data
        data = np.delete(data, delete_index, axis=1)
        # Update the met_data and rna_data
        met_data = np.delete(met_data, met_delete_index, axis=1)
        rna_data = np.delete(rna_data, rna_delete_index, axis=1)
        # Update the feature names and maps
        met_names = np.delete(np.array(met_names), met_delete_index).tolist()
        rna_names = np.delete(np.array(rna_names), rna_delete_index).tolist()
        print(" Features which are all NAs were removed.")
    if variable_gene:
        rna_data, rna_names = FindVariableFeatures(rna_data, rna_names, n_batch,
                                                                  start_row, stop_row,nfeatures=2000)
        data = np.concatenate([met_data, rna_data], axis=1)
    metabolite_map = {v: k for k, v in enumerate(met_names)}
    gene_map = {v: k for k, v in enumerate(rna_names)}


    feature_names = met_names + rna_names
    sample_names = list(sample_map_rna.keys())
    return data, met_data, rna_data, metabolite_map, gene_map, met_names, rna_names, feature_names, sample_names

def select_tumor_samples(fpath, df, MET_RNA_map, type):
    dir = fpath.split('_')[-1].split('.')[0]  # get the sub_dir name from file names
    if type == 'met':
        df = df.loc[MET_RNA_map.loc[(MET_RNA_map['sub_dir'] == dir) & (MET_RNA_map['TN'] == 'Tumor')].index]
    elif type == 'rna':
        df = df.loc[MET_RNA_map.loc[(MET_RNA_map['sub_dir'] == dir) & (MET_RNA_map['TN'] == 'Tumor'), 'RNAID']]
    return df

def filter_solo_met_data(met_data, n_batch, start_row, stop_row):
    """
    This function is used to filter out metabolites that are only present in one batch.
    It will set all solo metabolites to nan in the corresponding batch.
    These metabolites will be removed later in the data_normalization_cleaning function.
    Loop through all batches.
    When imputation, it will also loop through the imputation met dataset (which is all nan), but it won't do anything.
    But there is no need to remove solo metabolites when imputation.
    """
    for b in range(n_batch):
        # Mask all metabolite data of the each batch once
        temp = np.copy(met_data)
        temp[start_row[b]:stop_row[b], :] = np.nan
        solo = np.all(np.isnan(temp), axis=0)  # metabolites that are only present in one batch
        met_data[start_row[b]:stop_row[b], solo] = np.nan
    return met_data


def load_met_data_isotope(matched_data_dir, tumor_only):  # modified for isotope
    """
    When imputation, the loaded rna data will have one more batch than the loaded met data.
    """
    # First create an empty dataframe(sample*feature) with the shape of merged datasets, establish maps for features/samples/batches
    features = set()  # because we need the characteristic of non-repeating elements in set() to make 'features' non-redundant
    nrows = 0
    samples = []
    dirlist = os.listdir(matched_data_dir)
    dirlist.sort()
    for batch_idx, fpath in enumerate(
            dirlist):  # enumerate will return the iterable index and value in a list of tuples
        if 'csv' not in fpath:
            continue
        df = pd.read_csv(f'{matched_data_dir}/{fpath}', header=0, index_col=0)  # don't need to transpose the df now
        features.update(df.columns)
        samples.extend(df.index)  # with extend, each element of the iterable gets appended onto the list
        nrows += df.shape[0]
    feature_map = {s: i for i, s in enumerate(features)}  # create a dictionary {key0:value0,key1:value1} key-value pairs
    sample_map = {s: i for i, s in enumerate(samples)}
    data = np.full((nrows, len(features)),
                   np.nan)  # numpy.full(shape, fill_value) Return a new array of given shape & fill_value
    batch_index_vector = np.zeros(nrows, dtype=int)  # np.nan: nan, np.zeros return a array filled with zeros

    # Secondly, fill in the data from each dataset to the empty dataframe
    sidx = 0
    batch_idx = 0
    for fpath in dirlist:
        if 'csv' not in fpath:
            continue
        df = pd.read_csv(f'{matched_data_dir}/{fpath}', header=0, index_col=0)
        for feature in df.columns:  # fill in the data by feature columns
            fidx = feature_map[feature]
            data[sidx:sidx + df.shape[0], fidx] = df[
                feature].values  # df.values: Return a Numpy representation without axes labels
        batch_index_vector[sidx:sidx + df.shape[
            0]] = batch_idx  # [start, stop) For both dataframes and lists, df[i:j] or list[i:j] include ith but not jth item
        sidx += df.shape[0]  # so the index need to be start:start+number
        batch_idx += 1
    return data, feature_map, sample_map, batch_index_vector

def load_rna_data_isotope(matched_data_dir, tumor_only, proportions, imputation, imputation_dir=None, rna_imputation_data_dir=None):
    """
    The difference between this function and the met loading function is that we only take the intersection of genes.
    When imputation is True, we will load the rna data of the imputation set first and then
    load the other matched rna  data.
    Pay attention that the imputation dataset will always be put in the first batch.
    Added the filter to remove genes that are all zeros or all nans.
    modified for isotope
    """
    # First create an empty dataframe(sample*feature) with the shape of merged datasets, establish maps for features/samples/batches
    features = set()  # because we need the characteristic of non-repeating elements in set() to make 'features' non-redundant
    nrows = 0
    samples = []
    batch_names = []
    batch_sizes = []
    dirlist = os.listdir(matched_data_dir)
    dirlist.sort()
    if imputation:
        dirlist = [f'matched_tpm_{imputation_dir}.csv'] + dirlist  # put the imputation dataset in the first batch
    for batch_idx, fpath in enumerate(
            dirlist):  # enumerate will return the iterable index and value in a list of tuples
        if 'csv' not in fpath:
            continue
        elif imputation & (batch_idx == 0):
            df = pd.read_csv(f'{rna_imputation_data_dir}/{fpath}', header=0, index_col=0)
        else:
            df = pd.read_csv(f'{matched_data_dir}/{fpath}', header=0, index_col=0)  # don't need to transpose the df now
        # remove the genes that are all zeros or all nans
        df = df.loc[:, (df != 0).any(axis=0)]
        df = df.loc[:, (df.notnull()).any(axis=0)]
        if batch_idx == 0:
            features.update(df.columns)
        else:
            features = features.intersection(df.columns)  # only keep the intersection of features
        samples.extend(df.index)  # with extend, each element of the iterable gets appended onto the list
        batch_names.append(
            fpath)  # with append, whatever the object is, it gets added onto the end of my_list as a single entry
        batch_sizes.append(df.shape[0])
        nrows += df.shape[0]
    feature_map = {s: i for i, s in enumerate(features)}  # create a dictionary {key0:value0,key1:value1} key-value pairs
    sample_map = {s: i for i, s in enumerate(samples)}
    batch_map = {s: i for i, s in enumerate(batch_names)}
    data = np.full((nrows, len(features)),
                   np.nan)  # numpy.full(shape, fill_value) Return a new array of given shape & fill_value
    batch_index_vector = np.zeros(nrows, dtype=int)  # np.nan: nan, np.zeros return a array filled with zeros

    # Secondly, fill in the data from each dataset to the empty dataframe
    sidx = 0
    batch_idx = 0
    start_row = []
    stop_row = []
    for fpath in dirlist:
        if 'csv' not in fpath:
            continue
        elif imputation & (batch_idx == 0):
            df = pd.read_csv(f'{rna_imputation_data_dir}/{fpath}', header=0, index_col=0)
        else:
            df = pd.read_csv(f'{matched_data_dir}/{fpath}', header=0, index_col=0)
        for feature in features:  # fill in the data by feature columns
            fidx = feature_map[feature]
            data[sidx:sidx + df.shape[0], fidx] = df[
                feature].values  # df.values: Return a Numpy representation without axes labels
        batch_index_vector[sidx:sidx + df.shape[
            0]] = batch_idx  # [start, stop) For both dataframes and lists, df[i:j] or list[i:j] include ith but not jth item
        start_row.append(sidx)
        stop_row.append(sidx + df.shape[0])
        sidx += df.shape[0]  # so the index need to be start:start+number
        batch_idx += 1
    start_row = np.array(start_row)
    stop_row = np.array(stop_row)
    s_test_prop_settings = pd.DataFrame(list(zip(batch_names, proportions)), columns=["batches",
                                                                                      "s_test_prop"])  # zip() create paired tuples, need to use list/tuple() to display zip object
    n_batch = len(batch_map)
    print("The target dataset is assigned as follows ('1' indicates the target):")
    print(s_test_prop_settings)

    return data, batch_index_vector, feature_map, sample_map, batch_map, batch_sizes, start_row, stop_row, s_test_prop_settings, n_batch

def check_match(sample_map_met, sample_map_rna, MET_RNA_map, sub_dir, tumor_only, imputation, batch_index_vector):
    sample_met = list(sample_map_met.keys())
    if imputation:
        sample_rna = list(sample_map_rna.keys())[len(batch_index_vector[batch_index_vector == 0]):]
    else:
        sample_rna = list(sample_map_rna.keys())
    met_order = []
    rna_order = []
    # have to do a for loop because the order of subdir in MET_RNA_map is not the same as the order of subdir in sub_dir
    for i in range(len(sub_dir)):
        if imputation & (i == 0):
            continue
        if tumor_only:
            met_order.extend(MET_RNA_map.loc[(MET_RNA_map['sub_dir'] == sub_dir[i]) & (MET_RNA_map['TN'] == 'Tumor')].index.tolist())
            rna_order.extend(MET_RNA_map.loc[(MET_RNA_map['sub_dir'] == sub_dir[i]) & (MET_RNA_map['TN'] == 'Tumor'), 'RNAID'].tolist())
        else:
            met_order.extend(MET_RNA_map.loc[MET_RNA_map['sub_dir']==sub_dir[i]].index.tolist())
            rna_order.extend(MET_RNA_map.loc[MET_RNA_map['sub_dir']==sub_dir[i], 'RNAID'].tolist())
    if sample_met == met_order:
        print('MetaID matched')
    else:
        print('MetaID is not matched')
    if sample_rna == rna_order:
        print('RNAID matched')
    else:
        print('RNAID is not matched')


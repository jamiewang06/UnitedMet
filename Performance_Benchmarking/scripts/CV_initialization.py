import argparse
import csv
from Performance_Benchmarking.scripts.data_processing import *
from Performance_Benchmarking.scripts.pyro_model import generate_pyro_data
from Performance_Benchmarking.scripts.utils import test_train_split_met_rna, split_folds_across
from Performance_Benchmarking.scripts.cross_validation import saving_data_for_cv
import pickle

if __name__ == "__main__":
    ############################################################# Parse command-line options & arguments
    # Create a parser
    parser = argparse.ArgumentParser(description="Impute metabolomics data from transcriptomics data",
                                     formatter_class=argparse.RawTextHelpFormatter)
    # Add options/arguments
    # Options for cross-validation
    parser.add_argument('-cd', '--crossval_dims', help='Range of dimensions to evaluate in cross-validation',
                        required=False, nargs=2, type=int, default=[1, 202])
    parser.add_argument('-cs', '--crossval_steps', help='Steps of dimensions in cross-validation', required=False,
                        type=int, default=5)
    parser.add_argument('-cf', '--crossval_folds', help='Number of folds for cross-validation', required=False,
                        type=int, default=10)

    # Options for input data
    parser.add_argument('-im', '--imputation', help='Whether to do imputation without reference or not (benchmarking)',
                        required=False, action='store_true', default=False)
    parser.add_argument('-id', '--imputation_dir', help="dir for imputation ", required=False, type=str, default="TCGA")
    # Options for testing
    parser.add_argument('-ct', '--cancer_type', help="which cancer type is datasets from", required=True, type=str)
    parser.add_argument('-t', '--target', help="target dataset", required=False, type=int, default=1)
    parser.add_argument('-fp', '--file_path', help="the path of the parent directory to read/save all files",
                        required=False, type=str, default="/data1/reznike/xiea1/MetabolicModel")
    parser.add_argument('-rd', '--results_dir', help="results_dir", required=False, type=str)
    # Options for benchmarking runs
    parser.add_argument('-f', '--feature_test_prop',
                        help='proportion of features to be tested in each iteration',
                        required=False, type=float, default=1)

    # Parsing arguments from command lines
    args = parser.parse_args()
    cv_dims = args.crossval_dims
    cv_step = args.crossval_steps
    cv_folds = args.crossval_folds
    imputation = args.imputation
    imputation_dir = args.imputation_dir
    cancer_type = args.cancer_type
    target = args.target
    file_path = args.file_path
    results_dir = args.results_dir
    f_test_prop = args.feature_test_prop


    # ------------------------------------- Initialize Variables/Directories/Files -------------------------------------
    seed = 42
    (rna_matched_data_dir, met_matched_data_dir, rna_imputation_data_dir, sub_dir, proportions,
     plots_dir, embedding_dir) = file_system_init(file_path, cancer_type, imputation, imputation_dir, target, results_dir)
    # Load a master file for mapping between RNA and MET data
    if cancer_type == 'PanCancer':
        MET_RNA_map = pd.read_csv(f'{file_path}/data/MasterMapping_PanCancer.csv', header=0, index_col='MetabID')
    else:
        MET_RNA_map = pd.read_csv(f'{file_path}/data/MasterMapping_updated.csv', header=0, index_col='MetabID')

    # -------------------------------------- MAIN ----------------------------------
    # Load MET and RNA data
    met_data, metabolite_map, sample_map_met, met_batch_index_vector = load_met_data(met_matched_data_dir)
    if imputation:
        rna_data, batch_index_vector, gene_map, sample_map_rna, batch_map, batch_sizes, start_row, stop_row, \
        s_test_prop_settings, n_batch = load_rna_data(rna_matched_data_dir,  proportions,imputation,
                                                      imputation_dir, rna_imputation_data_dir)
    else:
        rna_data, batch_index_vector, gene_map, sample_map_rna, batch_map, batch_sizes, start_row, stop_row, \
        s_test_prop_settings, n_batch = load_rna_data(rna_matched_data_dir,  proportions, imputation)
    # Check the order of samples in loaded RNA/MET datasets are matched and same as the order in the master map
    check_match(sample_map_met, sample_map_rna, MET_RNA_map, sub_dir, imputation, batch_index_vector)
    # Normalization, Concatenate MET and RNA data, and remove features with all NaNs
    data, met_data, rna_data, metabolite_map, gene_map, met_names, rna_names, feature_names, sample_names = \
        data_normalization_cleaning(met_data, rna_data, met_batch_index_vector, metabolite_map,
                                    gene_map, sample_map_rna, imputation, batch_index_vector, n_batch,
                                    start_row, stop_row)
    n_mets, n_genes = met_data.shape[1], rna_data.shape[1]

    # -------------------------------------- Prepare data for CV ----------------------------------
    n_dims = None
    N, J, K, n_obs, orders, ranks, censor_indicator = generate_pyro_data(data, n_dims, n_batch, batch_index_vector)
    if imputation:
        training = np.copy(met_data)
    else:
        np.random.seed(seed)
        testing, training, n_obs, test_sample_indices, test_feature_indices, len_available = test_train_split_met_rna(
        met_data, start_row, stop_row, target, n_obs, ranks, f_test_prop)  # updated n_obs specify the training set
    ranks = ranks[:, :len(met_names)]  # subset to only include the metabolite features to save memory
    saving_data_for_cv(results_dir, orders, n_obs, batch_sizes, start_row, stop_row, ranks, n_mets, n_genes)

    # -------------------------------------- Initialize CV ----------------------------------
    print('Beginning cross-validation.')
    out_header = ["n_dims", "fold", "mae"]
    with open(f'{results_dir}/cv_folds_scores.csv', 'w+') as file:
        writer = csv.writer(file)
        writer.writerow(out_header)
    open(f"{results_dir}/scores.txt",
         "w").close()  # 'w': Opens a file for writing. Creates a new file if it does not exist or truncates(empty) the file if it exists.

    # -------------------------------------- Split and save folds ----------------------------------
    np.random.seed(seed)
    folds = split_folds_across(training, n_batch, batch_index_vector, cv_folds)
    if not os.path.exists(f"{results_dir}/folds"):
        os.makedirs(f"{results_dir}/folds")
    for fidx, fold in enumerate(folds):
        fold_temp = f"{results_dir}/folds/fold_{fidx}.pickle"
        pickle.dump(fold, open(fold_temp, 'wb+'))





















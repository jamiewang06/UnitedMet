import torch
import argparse
from Performance_Benchmarking.scripts_isotope.data_processing_isotope import *
from Performance_Benchmarking.scripts_isotope.pyro_model import generate_pyro_data, run_pyro_svi, svi_loss
from Performance_Benchmarking.scripts_isotope.utils_isotope import test_train_split_met_rna, saving_data_for_pyro_posterior, saving_data_for_testing

if __name__ == "__main__":
    ############################################################# Parse command-line options & arguments
    # Create a parser
    parser = argparse.ArgumentParser(description="Impute metabolomics data from transcriptomics data",
                                     formatter_class=argparse.RawTextHelpFormatter)
    # Add options/arguments
    parser.add_argument('-f', '--feature_test_prop', help='feature test/mask proportion?', required=False,
                        default=1, type=float)
    parser.add_argument('-n', '--ndims', help='Number of embedding dimensions for NMF', required=False, default=30,
                        type=int)
    parser.add_argument('-ve', '--verbose_embeddings', help="Write W, and H matrices to .csv files",
                        action='store_true', default=False)
    # Options for pyro SVI
    parser.add_argument('-ns', '--n_steps', help='Iteration steps of pyro SVI', required=False, type=int, default=2000)
    parser.add_argument('-lr', '--learning_rate', help='Learning rate of pyro SVI', required=False, type=float,
                        default=0.001)
    # Options for input data
    parser.add_argument('-s', '--remove_solo', help='When not imputing, whether to remove solo metabolites that are only present in one batch or not',
                        required=False, action='store_true', default=False)
    parser.add_argument('-to', '--tumor_only', help='Whether to only include tumor samples or not',
                        required=False, action='store_true', default=False)
    parser.add_argument('-im', '--imputation', help='Whether to do imputation without reference or not (benchmarking)',
                        required=False, action='store_true', default=False)
    parser.add_argument('-id', '--imputation_dir', help="dir for imputation ", required=False, default="TCGA", type=str)
    parser.add_argument('-ct', '--cancer_type', help="which cancer type is datasets from", required=False, default="ccRCC", type=str)
    parser.add_argument('-d', '--directory', help="which dataset to impute", required=False, type=str, default="RC18")
    parser.add_argument('-rd', '--results_directory', help="the path of the results directory", required=False,
                        type=str, default="/data1/reznike/xiea1/MetabolicModel/results_RNA_ccRCC/RC18")
    parser.add_argument('-fp', '--file_path', help="the path of the parent directory to read/save all files",
                        required=False, type=str, default="/data1/reznike/xiea1/MetabolicModel")

    # Parsing arguments from command lines
    args = parser.parse_args()
    f_test_prop = args.feature_test_prop
    n_dims = args.ndims
    ve = args.verbose_embeddings
    n_steps = args.n_steps
    lr = args.learning_rate
    remove_solo = args.remove_solo
    tumor_only = args.tumor_only
    imputation = args.imputation
    imputation_dir = args.imputation_dir
    cancer_type = args.cancer_type
    dir = args.directory
    results_dir = args.results_directory
    file_path = args.file_path

    # --------------------------------------Initialize Variables/Directories/Files--------------------------------------
    # Don't launch job arrays anymore (only 1 trial needed).
    seed = 42
    # Input Metabolomics and Transcriptomics data directory
    rna_matched_data_dir, met_matched_data_dir, rna_imputation_data_dir, sub_dir, proportions, \
    plots_dir, embedding_dir, target = file_system_init_isotope(file_path, cancer_type, imputation, imputation_dir, dir, results_dir)


    # -------------------------------------- Load and process MET and RNA data --------------------------------------
    met_data, metabolite_map, sample_map_met, met_batch_index_vector = load_met_data_isotope(met_matched_data_dir, tumor_only)
    if imputation:
        rna_data, batch_index_vector, gene_map, sample_map_rna, batch_map, batch_sizes, start_row, stop_row, \
        s_test_prop_settings, n_batch = load_rna_data_isotope(rna_matched_data_dir, tumor_only, proportions,
                                                      imputation, imputation_dir, rna_imputation_data_dir)
    else:
        rna_data, batch_index_vector, gene_map, sample_map_rna, batch_map, batch_sizes, start_row, stop_row, \
        s_test_prop_settings, n_batch = load_rna_data_isotope(rna_matched_data_dir, tumor_only, proportions,
                                                      imputation)
    # Normalization, Concatenate MET and RNA data, and remove features with all NaNs
    data, met_data, rna_data, metabolite_map, gene_map, met_names, rna_names, feature_names, sample_names = \
        data_normalization_cleaning_isotope(met_data, rna_data, metabolite_map,
                                            gene_map, sample_map_rna, imputation, batch_index_vector, n_batch,
                                            start_row, stop_row, remove_solo, 'tissue_pyruvate',
                                            cancer_type)
    pd.DataFrame(data, index=sample_names, columns=feature_names).to_csv(
        f'{embedding_dir}/normalized_met_rna_data_pyro.csv')

    # -------------------------------------- MAIN --------------------------------------
    N, J, K, n_obs, orders, ranks, censor_indicator = generate_pyro_data(data, n_dims, n_batch,
                                                                                batch_index_vector)
    n_obs_pre = np.copy(n_obs)  # save the n_obs of the original data
    if not imputation:
        np.random.seed(seed)
        testing, training, n_obs, test_sample_indices, test_feature_indices, len_available = test_train_split_met_rna(
            met_data, start_row, stop_row, target, n_obs, ranks, f_test_prop)  # updated n_obs specify the training set

    torch.manual_seed(seed)
    W_loc, W_scale, H_loc, H_scale, loss_list = run_pyro_svi(N, J, K, n_batch, start_row, stop_row, n_obs, orders, n_steps, lr)
    #W_loc, W_scale, H_loc, H_scale, loss_list = run_pyro_svi_alter(N, J, K, n_batch, start_row, stop_row, n_obs, orders, n_steps, lr, n_iter=5)
    svi_loss(loss_list, plots_dir)  # plot the loss function of the model

    with open(f'{embedding_dir}/W_H_loc_scale.npy', 'wb') as f:  # 'wb': write as binary
        np.save(f, W_loc)
        np.save(f, W_scale)
        np.save(f, H_loc)
        np.save(f, H_scale)
        f.close()

    saving_data_for_pyro_posterior(embedding_dir, N, K, J, met_names, start_row, stop_row, batch_sizes)
    if not imputation:
        saving_data_for_testing(embedding_dir, testing, test_sample_indices, test_feature_indices, n_obs_pre, censor_indicator)


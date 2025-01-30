import numpy as np
import pandas as pd
import argparse
from Performance_Benchmarking.scripts_isotope.pyro_model import pyro_posterior
from Performance_Benchmarking.scripts_isotope.testing import testing_function_met_rna, results_analysis, results_visualization, results_probabilistic_analysis

if __name__ == "__main__":
    ############################################################# Parse command-line options & arguments
    # Create a parser
    parser = argparse.ArgumentParser(description="Impute metabolomics data from transcriptomics data",
                                     formatter_class=argparse.RawTextHelpFormatter)
    # Add options/arguments
    parser.add_argument('-f', '--feature_test_prop', help='feature test/mask proportion?', required=False,
                        default=1, type=float)
    parser.add_argument('-ve', '--verbose_embeddings', help="Write W, and H matrices to .csv files",
                        action='store_true', default=False)
    parser.add_argument('-im', '--imputation', help='Whether to do imputation without reference or not (benchmarking)',
                        required=False, action='store_true', default=False)
    parser.add_argument('-id', '--imputation_dir', help="dir for imputation ", required=False, type=str, default='TCGA')
    parser.add_argument('-ct', '--cancer_type', help="which cancer type is datasets from", required=False, type=str, default='ccRCC')
    parser.add_argument('-d', '--directory', help="which dataset to impute", required=False, type=str, default='RC18')
    parser.add_argument('-rd', '--results_directory', help="the path of the results directory", required=False, type=str,
                        default="/juno/work/reznik/xiea1/MetabolicModel/results_RNA_ccRCC/RC18")
    args = parser.parse_args()
    f_test_prop = args.feature_test_prop
    ve = args.verbose_embeddings
    imputation = args.imputation
    imputation_dir = args.imputation_dir
    cancer_type = args.cancer_type
    dir = args.directory
    results_dir = args.results_directory

    # ----------------------- Initialize Variables/Directories/Files -----------------------
    job_index = 1  # Used to be the job index in a job array. I don't launch job arrays anymore (only 1 trial needed).
    print('The job index is: ', job_index)
    seed = 42
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
    elif cancer_type == 'NSCLC':  # NSCLC benchmarking
        sub_dir = ['NSCLC1', 'NSCLC2']
    elif cancer_type == 'NSCLC_G6':  # NSCLC glucose tracer
        sub_dir = ['NSCLC_G6_1', 'NSCLC_G6_2']
    elif cancer_type == 'NSCLC_Q6':  # NSCLC glutamine tracer
        sub_dir = ['NSCLC_Q6_1', 'NSCLC_Q6_2']
    elif cancer_type == 'NSCLC_G24':  # NSCLC glucose tracer 24h
        sub_dir = ['NSCLC_G24_1', 'NSCLC_G24_2']
    elif cancer_type == 'NSCLC_Q24':  # NSCLC glutamine tracer 24h
        sub_dir = ['NSCLC_Q24_1', 'NSCLC_Q24_2']
    else:
        raise ValueError(f"Invalid cancer type: {cancer_type}")
    target = sub_dir.index(dir) + 1
    plots_dir = f'{results_dir}/plots'
    embedding_dir = f'{results_dir}/embeddings'

    # -------------------------------Load data from files--------------------------------
    with open(f'{embedding_dir}/data_for_posterior_pyro.npy', 'rb') as f:
        N = np.load(f)
        K = np.load(f)
        J = np.load(f)
        start_row = np.load(f)
        stop_row = np.load(f)
        batch_sizes = np.load(f)
        f.close()
    n_batch = len(start_row)
    met_names = pd.read_csv(f'{embedding_dir}/met_names_pyro.csv')['met_names'].values

    if not imputation:
        with open(f'{embedding_dir}/data_for_testing_pyro.npy', 'rb') as f:
            testing = np.load(f)
            test_sample_indices = np.load(f)
            test_feature_indices = np.load(f)
            n_obs_pre = np.load(f)
            censor_indicator = np.load(f)
            f.close()

    with open(f'{embedding_dir}/W_H_loc_scale.npy', 'rb') as f:
        W_loc = np.load(f)
        W_scale = np.load(f)
        H_loc = np.load(f)
        H_scale = np.load(f)
        f.close()
    print(f'Reading posterior latent variables from {embedding_dir}/W_H_loc_scale.npy')

    #------------------------Matrix Multiplication and Posterior Prediction------------------------
    num_samples = 1000  # number of samples to draw from posterior
    rank_hat_draws, rank_hat_mean, rank_hat_std = pyro_posterior(W_loc, W_scale, H_loc, H_scale, n_batch, start_row, stop_row, ve, met_names, embedding_dir, seed, num_samples=1000)

    # ------------------------Testing and Results Analysis------------------------
    if not imputation:
        # note that here we directly store results in the actual_pred_res_df, because n_iter = 1
        actual_pred_res_df = testing_function_met_rna(testing, rank_hat_mean, rank_hat_std,
                                      test_sample_indices, test_feature_indices,
                                      n_obs_pre, met_names, censor_indicator, job_index, target, f_test_prop)
        actual_pred_res_df.to_csv(f'{results_dir}/actual_vs_predicted_ranks.csv')
        by_iter_rho, median_rho_iter, median_rho_feature = results_analysis(actual_pred_res_df, batch_sizes, target, num_samples=1000)
        median_rho_feature.to_csv(f'{results_dir}/median_rho_feature.csv')
        results_visualization(median_rho_feature, median_rho_iter, actual_pred_res_df, f_test_prop, target, sub_dir,
                              plots_dir)
        results_probabilistic_analysis(actual_pred_res_df, testing, rank_hat_draws, test_feature_indices,
                                       test_sample_indices, n_obs_pre, target, median_rho_feature, plots_dir, sub_dir)





















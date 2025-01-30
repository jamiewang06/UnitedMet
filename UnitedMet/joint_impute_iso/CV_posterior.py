from datetime import datetime
import numpy as np
import pickle
import os
import argparse
import csv
from Performance_Benchmarking.scripts_isotope.pyro_model import posterior_prediction_3D
from Performance_Benchmarking.scripts_isotope.cross_validation import cv_score_mae_met_rna

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Impute metabolomics data_Cross Validation_posterior",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-rd', '--results_dir', help="results_dir", required=False, type=str)
    parser.add_argument('-cd', '--crossval_dims', help='Range of dimensions to evaluate in cross-validation',
                        required=False, nargs=2, type=int, default=[1, 202])
    parser.add_argument('-cs', '--crossval_steps', help='Steps of dimensions in cross-validation', required=False,
                        type=int, default=10)
    parser.add_argument('-cf', '--crossval_folds', help='Number of folds for cross-validation', required=False,
                        type=int, default=10)
    args = parser.parse_args()
    results_dir = args.results_dir
    cv_dims = args.crossval_dims
    cv_step = args.crossval_steps
    cv_folds = args.crossval_folds

    # job_index = os.environ.get("LSB_JOBINDEX")  # LSF: Use $LSB_JOBINDEX in the shell script, don't include the $ in py file
    job_index = os.environ.get("SLURM_ARRAY_TASK_ID")  # SLURM: get the SLURM_ARRAY_TASK_ID from the environment
    print('The job index is: ', job_index)
    job_index = int(job_index)

    # -------------------------- Read posterior latent variables and other data ---------------------------------
    cv_embedding_dir = f'{results_dir}/cv_embeddings'
    with open(f'{cv_embedding_dir}/W_H_loc_scale_{job_index}.npy', 'rb') as f:
        W_loc = np.load(f)
        W_scale = np.load(f)
        H_loc = np.load(f)
        H_scale = np.load(f)
        f.close()
    print(f'Reading posterior latent variables from {cv_embedding_dir}/W_H_loc_scale_{job_index}.npy')

    with open(f'{results_dir}/matrices.npy', 'rb') as f:
        orders = np.load(f)
        n_obs_pre = np.load(f)  # since the saved n_obs is the original one, n_ons_pre = n_obs
        batch_sizes = np.load(f)
        start_row = np.load(f)
        stop_row = np.load(f)
        ranks = np.load(f)
        f.close()

    # Calculate the corresponding dimension and fold number from the job index in the array
    n_dims_options = np.arange(start=cv_dims[0], stop=cv_dims[1], step=cv_step)
    didx = (job_index-1)//cv_folds
    n_dims = n_dims_options[didx]
    fidx = (job_index-1) % cv_folds
    print('The dimension is: ', n_dims)
    print('The fold is: ', fidx)

    n_batch = n_obs_pre.shape[0]
    fold_path = f"{results_dir}/folds/fold_{fidx}.pickle"
    fold = pickle.load(open(fold_path, 'rb'))

    # ---------------------------Initialize---------------------------
    seed = 42
    num_samples = 1000
    #mae = False  # test the likelihood way to do CV

    np.random.seed(seed)
    W_draws = np.random.normal(W_loc, W_scale, size=(num_samples, W_loc.shape[0], W_loc.shape[1]))
    np.random.seed(seed)
    H_draws = np.random.normal(H_loc, H_scale, size=(num_samples, H_loc.shape[0], H_loc.shape[1]))
    X_draws = np.matmul(W_draws, H_draws[:, :, :ranks.shape[1]])  # subset to only metabolites to save memory
    rank_hat_draws = posterior_prediction_3D(X_draws, n_batch, start_row, stop_row)
    rank_hat_mean = np.mean(rank_hat_draws, axis=0)
    score = cv_score_mae_met_rna(rank_hat_mean, ranks, n_batch, start_row, stop_row, fold, n_obs_pre)

    # Write cv scores
    with open(f"{results_dir}/scores.txt", 'a') as f:  # 'a': Append Only
        f.write(str(didx) + " " + str(fidx) + " " + str(score) + "\n")  # remove 30 later
        f.close()
    with open(f'{results_dir}/cv_folds_scores.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow([n_dims, fidx, score])

    # Record the finish time for parallel jobs
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("The time of writing scores =", current_time)
    print('The total cv_score is: ', score)
    print(f'End of job {job_index}.')

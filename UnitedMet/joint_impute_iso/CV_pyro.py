import numpy as np
import pickle
import os
import torch
import argparse
from Performance_Benchmarking.scripts_isotope.pyro_model import run_pyro_svi

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Impute metabolomics data_Cross Validation_pyro",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-rd', '--results_dir', help="results_dir", required=False, type=str)
    parser.add_argument('-cd', '--crossval_dims', help='Range of dimensions to evaluate in cross-validation',
                        required=False, nargs=2, type=int, default=[1, 202])
    parser.add_argument('-cs', '--crossval_steps', help='Steps of dimensions in cross-validation', required=False,
                        type=int, default=10)
    parser.add_argument('-cf', '--crossval_folds', help='Number of folds for cross-validation', required=False,
                        type=int, default=10)
    # Options for pyro SVI
    parser.add_argument('-n', '--n_steps', help='Iteration steps of pyro SVI', required=False, type=int, default=2000)
    parser.add_argument('-lr', '--learning_rate', help='Learning rate of pyro SVI', required=False, type=float,
                        default=0.001)

    args = parser.parse_args()
    results_dir = args.results_dir
    cv_dims = args.crossval_dims
    cv_step = args.crossval_steps
    cv_folds = args.crossval_folds
    n_steps = args.n_steps
    lr = args.learning_rate

    # job_index = os.environ.get("LSB_JOBINDEX")  # LSF: Use $LSB_JOBINDEX in the shell script, don't include the $ in py file
    job_index = os.environ.get("SLURM_ARRAY_TASK_ID")  # SLURM: get the SLURM_ARRAY_TASK_ID from the environment
    print('The job index is: ', job_index)
    job_index = int(job_index)

    # Random seed
    seed = 42

    # MAIN CODE
    with open(f'{results_dir}/matrices.npy', 'rb') as f:
        orders = np.load(f)
        n_obs = np.load(f)
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

    n_batch = n_obs.shape[0]
    fold_path = f"{results_dir}/folds/fold_{fidx}.pickle"
    fold = pickle.load(open(fold_path, 'rb'))
    n_obs_pre = np.copy(n_obs)  # save the original n_obs of the training data
    for b in range(n_batch):
        n_obs[b, fold[b]] = 0  # Reset the n_obs according to the fold!!
    N = orders.shape[0]  # number of samples
    J = orders.shape[1]  # number of features
    K = n_dims

    torch.manual_seed(seed)
    W_loc, W_scale, H_loc, H_scale, loss_list = run_pyro_svi(N, J, K, n_batch, start_row, stop_row, n_obs, orders, n_steps, lr)
    cv_embedding_dir = f'{results_dir}/cv_embeddings'
    if not os.path.exists(cv_embedding_dir):
        os.makedirs(cv_embedding_dir)
    with open(f'{cv_embedding_dir}/W_H_loc_scale_{job_index}.npy', 'wb') as f:  # 'wb': write as binary
        np.save(f, W_loc)
        np.save(f, W_scale)
        np.save(f, H_loc)
        np.save(f, H_scale)
        f.close()


import numpy as np
import matplotlib.pyplot as plt
from Performance_Benchmarking.scripts_isotope.pyro_model import PlackettLuce_2D
import torch
from kneed import KneeLocator

def saving_data_for_cv(results_dir, orders, n_obs, batch_sizes, start_row, stop_row, ranks):
    with open(f'{results_dir}/matrices.npy', 'wb') as f:  # 'wb': write as binary
        np.save(f, orders)  # save arrays in numpy binary .npy files
        np.save(f, n_obs)
        np.save(f, batch_sizes)
        np.save(f, start_row)
        np.save(f, stop_row)
        np.save(f, ranks)
        f.close()

def create_lsf_jobs_cv(results_dir, sub_dir, target, cv_dims, cv_step, n_folds, n_steps, lr, n_indices, logs_dir):
    job_file_path = f"{results_dir}/CV_job.sh"
    with open(job_file_path, 'w+') as f:
        job = f"""
#!/bin/bash
module load gcc/11.2.0
source /home/xiea1/.bashrc
conda activate pyro
cd /juno/work/reznik/xiea1/pyro

job_opts="-n 23 -W 2:59"
script="scripts.CV_pyro"
opts="-rd {results_dir} -cd {cv_dims[0]} {cv_dims[1]} -cs {cv_step} -cf {n_folds} -n {n_steps} -lr {lr}"
    
bsub -J "{sub_dir[target - 1]}_CV"[1-{n_indices}] $job_opts -R "span[hosts=1] rusage[mem=3]" -o {logs_dir}/CV_array_%J.out python3 -m $script $opts
bsub -J 'final_CV' -w "done("{sub_dir[target - 1]}_CV")" python3 scripts/empty.py -r {results_dir}
        """
        f.write(job)

# FUNCTIONS FOR CV
def cv_score_likelihood_across(X_draws_mean, orders, n_batch, start_row, stop_row, fold, n_obs_pre):
    """
    Some details in the function might need to be changed to apply it to the RNA-->MET data.
    For example, the array of the target dataset in fold will be an empty array, which might be problematic.
    modified: added fold[b].shape[0] == 0 check
    """
    log_likelihood = 0
    for b in range(n_batch):
        if fold[b].shape[0] == 0:
            continue
        X_test = X_draws_mean[start_row[b]:stop_row[b], fold[b]]
        orders_test = orders[start_row[b]:stop_row[b], fold[b]]
        # ordered from largest to smallest
        X_test_permuted = np.take_along_axis(X_test, orders_test, axis=0)
        # Normal prior
        X_test_permuted_cumsum = np.logaddexp.accumulate(X_test_permuted[::-1, :], axis=0)[::-1, :]
        # since each metabolite j have different n_obs, we couldn't vectorize the calculation on the whole 3d matrix
        for j in range(X_test.shape[1]):
            k = n_obs_pre[b, fold[b][j]]  # pay attention to the different indexing here
            # get the average log likelihood for each metabolite over 1000 draws and sum it up through the loop
            # Normal prior
            log_likelihood += np.sum(X_test_permuted[:k, j] - X_test_permuted_cumsum[:k, j])

    return log_likelihood

def cv_score_likelihood_pyro(X_draws_mean, orders, ranks, n_batch, start_row, stop_row, fold, n_obs_pre):
    """
        Using the Plackett-Luce likelihood function in the PlackettLuce_2D class I wrote to calculate the log likelihood.
        I have already double checked the output of this function is the same as the cv_score_likelihood_across function.
    """
    log_likelihood = 0
    orders = orders[:, :ranks.shape[1]]  # since the X_draws_mean is only metabolites, shape need to be matched
    X_draws_mean = torch.tensor(X_draws_mean)
    orders = torch.tensor(orders)
    for b in range(n_batch):
        if fold[b].shape[0] == 0:
            continue
        n_obs_fold = np.zeros(X_draws_mean.shape[1]).astype(int)
        n_obs_fold[fold[b]] = n_obs_pre[b, fold[b]]
        n_obs_fold = torch.tensor(n_obs_fold)
        log_likelihood += PlackettLuce_2D(X_draws_mean[start_row[b]:stop_row[b], :],
                                          n_obs_fold).log_prob(orders[start_row[b]:stop_row[b], :])
    return log_likelihood

def cv_score_mae_met_rna(rank_hat_mean, ranks, n_batch, start_row, stop_row, fold, n_obs_pre):
    """
    The difference between this function and the one in PL_across_cv.py is that I added a condition check to
    skip the empty fold of the target dataset, which will result in empty slices of rank and NaN of mae.
    """
    mae = 0
    for b in range(n_batch):
        if fold[b].shape[0] == 0:
            continue
        # get predicted ranks of the fold
        rank_hat_mean_test = rank_hat_mean[start_row[b]:stop_row[b], fold[b]]
        rerank_hat_mean_test = rank_hat_mean_test.argsort(
            axis=0, kind='stable').argsort(axis=0, kind='stable')  # re-rank the predicted values
        # if rank > n_obs, set rank = n_obs (get censored rank)
        rerank_hat_mean_test = np.where(
            rerank_hat_mean_test > n_obs_pre[b, fold[b]], n_obs_pre[b, fold[b]],
            rerank_hat_mean_test)
        ranks_test = ranks[start_row[b]:stop_row[b], fold[b]]
        mae += np.abs(rerank_hat_mean_test - ranks_test).mean()
    return mae

def summarize_cv_results(cv_dims, cv_step, n_folds, results_dir, plots_dir):
    n_dims_options = np.arange(start=cv_dims[0], stop=cv_dims[1], step=cv_step)
    scores = np.zeros((len(n_dims_options), n_folds))
    with open(f"{results_dir}/scores.txt", "r") as f:
        for line in f:
            sp = line.split()
            scores[int(sp[0]), int(sp[1])] = float(sp[2])  # mae
            #scores[int(sp[0]), int(sp[1])] = float(sp[2].replace('tensor(', '').rstrip(','))
            # likelihood (since scores are in tensor string formats)
        f.close()

    # get best params and one_se_rule params
    scores_means = np.mean(scores, axis=1)
    scores_se = np.std(scores, axis=1) / np.sqrt(n_folds)
    n_dims_min = n_dims_options[np.argmin(scores_means)]  # get the minimum of cv_score
    cutoff = scores_means[np.argmin(scores_means)] + scores_se[np.argmin(scores_means)]
    n_dims_one_se_rule = n_dims_options[
        np.argmax(scores_means < cutoff)]  # get the first index of cv_score that is less than cutoff
    print(f'Best dims is {n_dims_min}, while one se rule dims is {n_dims_one_se_rule}')

    # find the knee point
    kneedle = KneeLocator(n_dims_options, scores_means, curve='convex', direction='decreasing')
    n_dims_knee = kneedle.knee
    knee_y = kneedle.knee_y
    plot_cv_scores(scores_means, scores_se, n_dims_options, cutoff, n_dims_min, n_dims_one_se_rule,
                   n_dims_knee, knee_y, results_dir, plots_dir)

    return n_dims_min, n_dims_one_se_rule, n_dims_knee

# plot the cv_score
def plot_cv_scores(scores_means, scores_se, n_dims_options, cutoff, n_dims_min, n_dims_one_se_rule,
                   n_dims_knee, knee_y, results_dir, plots_dir):
    plt.rcParams['figure.figsize'] = [6, 6]
    plt.plot(n_dims_options, scores_means, color='black')
    plt.fill_between(n_dims_options, scores_means + 1 * scores_se, scores_means - 1 * scores_se, alpha=.1)
    #plt.axhline(cutoff, linestyle='dotted')
    plt.axvline(x=n_dims_knee, linestyle='dotted', linewidth=2.0)
    plt.scatter([n_dims_one_se_rule], [scores_means[np.argmax(scores_means < cutoff)]], marker='o', color='orange',
                label='One SE Rule')
    plt.scatter([n_dims_min], [scores_means[np.argmin(scores_means)]], marker='o', color='blue', label='Minimum rule')
    plt.scatter([n_dims_knee], [knee_y], marker='o', color='red', label='Elbow point')
    plt.legend()
    plt.xlabel("# embedding dimensions")  # add X-axis label
    plt.ylabel("fold-average MAE")  # add Y-axis label
    plt.title(results_dir.rsplit("/", 1)[-1])  # add title
    plt.grid()
    plt.savefig(f'{plots_dir}/cv_score.pdf')
    plt.close()



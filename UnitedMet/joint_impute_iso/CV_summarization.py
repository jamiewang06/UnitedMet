import csv
import argparse
from Performance_Benchmarking.scripts_isotope.cross_validation import summarize_cv_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Impute metabolomics data_Cross Validation Summarization",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-rd', '--results_dir', help="results_dir", required=False, type=str,
                        default="/juno/work/reznik/xiea1/MetabolicModel/results_RNA_ccRCC/RC18")
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

    plots_dir = f'{results_dir}/plots'

    n_dims_min, n_dims_one_se_rule, n_dims_knee = summarize_cv_results(cv_dims, cv_step, cv_folds, results_dir, plots_dir)
    with open(f'{results_dir}/cv_best_dims.csv', 'w+') as file:
        writer = csv.writer(file)
        writer.writerow(['best_n_dims_min', 'best_n_dims_one_se_rule', 'best_n_dims_elbow'])
        writer.writerow([n_dims_min, n_dims_one_se_rule, n_dims_knee])


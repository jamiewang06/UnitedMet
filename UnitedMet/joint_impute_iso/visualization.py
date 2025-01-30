import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from collections import Counter

# scatter plot for standard deviation/error of each feature
def scatterplot_std(data, target, x, y, mode, sub_dir, plots_dir):
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.scatter(x=data[x], y=data[y], c=data['censor_prop'])
    # plt.xlim(left=1)
    if mode == 'std':
        plt.xlabel('Average Standard Deviation')
    elif mode == 'se':
        plt.xlabel('Average Standard Error')
    # plt.axline((0, 1), slope=-1)
    plt.ylabel('Spearman Rho')
    plt.colorbar(label='Proportion of Censored Samples')
    plt.title(f'{sub_dir[target - 1]}')
    plt.savefig(f'{plots_dir}/scatterplot_{x}_{y}.pdf')
    plt.close()
    rho, pval = spearmanr(data[x], data[y])
    print(f"Spearman rho vs average standard error/deviation:, spearman's rho = {rho}, pval = {pval}")


# Box plot for median rho by iteration
def median_rho_iter_plot(data, plots_dir):
    plt.rcParams['figure.figsize'] = [8, 4]
    boxplot = (data
               .reset_index()
               .pivot(columns="f_test_prop", values="median_rho")
               .boxplot())
    # plt.ylim([-1,1])
    plt.ylabel('median rho')
    plt.xlabel('proportion of features simulated missing')
    plt.savefig(f'{plots_dir}/Box plot of median rho by iteration.pdf')
    plt.close()


# Bar plot for median rho by metabolite
def median_rho_feature_plot(data, f_test_prop, plots_dir):
    bar_df = (data.sort_values(by=["median_rho"], ascending=False)
              .assign(
        color=lambda df: df.apply(lambda row: 'red' if row['sig_in_most'] and row['median_rho'] > 0 else 'grey',
                                  axis=1)))

    plt.rcParams['figure.figsize'] = [80, 30]
    plt.bar(bar_df["feature"], bar_df["median_rho"], color=bar_df["color"])
    plt.xlabel("metabolite")
    plt.ylabel("median rho")
    plt.title(f'median rho for metabolites at {f_test_prop} test proportion')
    plt.xticks(rotation=-90)
    plt.margins(x=0.01)
    plt.savefig(f'{plots_dir}/Bar plot of median rho by metabolites.pdf')
    plt.close()


# Plot the bar graph of metabolites that are significantly & positively predicted in each iteration
def sig_in_iter_plot(df, plots_dir):
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.hist(df.loc[df['median_rho'] > 0, 'sig_in'], density=True)
    plt.xlabel("Proportion of iterations where the metabolites are significantly predicted and positively correlated")
    plt.ylabel("Density")
    plt.title("How significantly predicted & positively correlated metabolites overlap across iterations")
    plt.savefig(f'{plots_dir}/Metabolite distribution of significant prediction.pdf')
    plt.close()

# Plot the scatterplot of actual vs predicted ranks for specific metabolites
def scatterplot_all_iter(data, met, plots_dir, mode):
    test = data.loc[data['feature'] == met]
    count = Counter(zip(test['actual_rank'], test['predicted_rerank']))
    test['size'] = [10 * count[(xx, yy)] for xx, yy in zip(test['actual_rank'], test['predicted_rerank'])]
    plt.rcParams['figure.figsize'] = [15, 10]
    if mode == 'iteration':
        test.plot.scatter(x='actual_rank', y='predicted_rerank', c='iteration', s='size')
    if mode == 'censor':
        test.plot.scatter(x='actual_rank', y='predicted_rerank', c='censor', s='size')
    plt.axline((0, 0), slope=1)
    plt.title(met)
    plt.savefig(f"{plots_dir}/scatterplot_{met}_all_iter_{mode}.pdf")
    plt.close()
    rho, pval = spearmanr(test['actual_rank'], test['predicted_rerank'])
    print(f"Actual vs predicted ranks for {met}: spearman's rho = {rho}, pval = {pval}")



def predicted_rank_histogram(data, plots_dir):
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.hist(data['predicted_rank'])
    plt.xlabel('predicted rank')
    plt.ylabel('count')
    plt.title('predicted rank histogram')
    plt.savefig(f'{plots_dir}/predicted_rank_histogram.pdf')
    plt.close()


def scatterplot_all(data, plots_dir, actual_pred_res_df):
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.scatter(data['actual_rank'], data['predicted_rerank'])
    plt.axline((0, 0), slope=1)
    plt.xlabel('actual rank')
    plt.ylabel('predicted rank')
    plt.savefig(f'{plots_dir}/scatterplot_all.pdf')
    plt.close()
    rho, pval = spearmanr(actual_pred_res_df['actual_rank'], actual_pred_res_df['predicted_rerank'])
    print(f"Actual vs predicted ranks for all metabolites: spearman's rho = {rho}, pval = {pval}")

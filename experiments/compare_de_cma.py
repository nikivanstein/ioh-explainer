"""This script compares the results from modular CMA-ES and modular DE.
It first loads the configuration spaces of the two frameworks and the ioh-xplainer modules.
It then uses a "compare" function to process the performance data between the frameworks in one latex table.

Writes a compare-new.tex file with the comparison in latex table format.
"""
import ioh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from iohxplainer import explainer

from config import cmaes_explainer, de_explainer, de_features, cma_features


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def compare(alg1, alg2, normalize=False):
    # assuming both alg1 and alg2 are explainer objects
    if not isinstance(alg1, explainer):
        raise "instance alg1 should be an explainer object"
    df1 = alg1.df
    df2 = alg2.df

    comparison_stats = {}

    for dim in intersection(alg1.dims, alg2.dims):
        dim_df1 = df1[df1["dim"] == dim]
        dim_df2 = df2[df2["dim"] == dim]
        stat_index = f"d={dim}"

        comparison_stats[stat_index] = pd.DataFrame(
            columns=[
                "Function",
                f"single-best {alg1.algname}",
                f"single-best {alg2.algname}",
                f"avg-best {alg1.algname}",
                f"avg-best {alg2.algname}",
                f"{alg1.algname}",
                f"{alg2.algname}",
            ]
        )

        # split df per function
        # get avg best config
        _, df_best_mean1 = alg1._get_average_best(dim_df1)
        _, df_best_mean2 = alg2._get_average_best(dim_df2)

        for fid in intersection(alg1.fids, alg2.fids):
            func = ioh.get_problem(fid, dimension=dim, instance=1)
            fid_df1 = dim_df1[dim_df1["fid"] == fid]
            fid_df2 = dim_df2[dim_df2["fid"] == fid]

            _, df_single_best1 = alg1.get_single_best(fid, dim)
            _, df_single_best2 = alg2.get_single_best(fid, dim)

            # Define the new row to be added
            avg_best_avg1 = df_best_mean1[df_best_mean1["fid"] == fid]["auc"].mean()
            avg_best_var1 = df_best_mean1[df_best_mean1["fid"] == fid]["auc"].std()

            avg_best_avg2 = df_best_mean2[df_best_mean2["fid"] == fid]["auc"].mean()
            avg_best_var2 = df_best_mean2[df_best_mean2["fid"] == fid]["auc"].std()

            avg_avg1 = fid_df1["auc"].mean()
            avg_var1 = fid_df1["auc"].std()

            avg_avg2 = fid_df2["auc"].mean()
            avg_var2 = fid_df2["auc"].std()

            avg_single1 = df_single_best1["auc"].mean()
            var_single1 = df_single_best1["auc"].std()

            avg_single2 = df_single_best2["auc"].mean()
            var_single2 = df_single_best2["auc"].std()
            if normalize:
                avg_single1 = avg_single1 / alg1.budget
                var_single1 = var_single1 / alg1.budget
                avg_best_avg1 = avg_best_avg1 / alg1.budget
                avg_best_var1 = avg_best_var1 / alg1.budget
                avg_avg1 = avg_avg1 / alg1.budget
                avg_var1 = avg_var1 / alg1.budget

                avg_single2 = avg_single2 / alg2.budget
                var_single2 = var_single2 / alg2.budget
                avg_best_avg2 = avg_best_avg2 / alg2.budget
                avg_best_var2 = avg_best_var2 / alg2.budget
                avg_avg2 = avg_avg2 / alg2.budget
                avg_var2 = avg_var2 / alg2.budget

            single_best1 = f"{avg_single1:.2f} ({var_single1:.2f})"
            single_best2 = f"{avg_single2:.2f} ({var_single2:.2f})"

            avg_best1 = f"{avg_best_avg1:.2f} ({avg_best_var1:.2f})"
            avg_best2 = f"{avg_best_avg2:.2f} ({avg_best_var2:.2f})"

            avg1 = f"{avg_avg1:.2f} ({avg_var1:.2f})"
            avg2 = f"{avg_avg2:.2f} ({avg_var2:.2f})"

            # single best significance
            single_sig = False
            avg_sig = False
            res = stats.ttest_ind(
                df_single_best1["auc"].values, df_single_best2["auc"].values
            )

            if res.pvalue < 0.05:
                single_sig = True
                if avg_single1 > avg_single2:
                    single_best1 = "\\textbf{" + single_best1 + f"}} ({res.pvalue:.3f})"
                else:
                    single_best2 = "\\textbf{" + single_best2 + f"}} ({res.pvalue:.3f})"

            # avg best significance
            res = stats.ttest_ind(fid_df1["auc"].values, fid_df2["auc"].values)
            if res.pvalue < 0.05:
                avg_sig = True
                if avg_best1 > avg_best2:
                    avg_best1 = "\\textbf{" + avg_best1 + f"}} ({res.pvalue:.3f})"
                else:
                    avg_best2 = "\\textbf{" + avg_best2 + f"}} ({res.pvalue:.3f})"

            # avg significance
            res = stats.ttest_ind(dim_df1["auc"].values, dim_df2["auc"].values)
            if res.pvalue < 0.05:
                avg_sig = True
                if avg1 > avg2:
                    avg1 = "\\textbf{" + avg1 + f"}} ({res.pvalue:.3f})"
                else:
                    avg2 = "\\textbf{" + avg2 + f"}} ({res.pvalue:.3f})"

            new_row = {
                "Function": f"f{fid} {func.meta_data.name}",
                f"single-best {alg1.algname}": single_best1,
                f"single-best {alg2.algname}": single_best2,
                f"avg-best {alg1.algname}": avg_best1,
                f"avg-best {alg2.algname}": avg_best2,
                f"{alg1.algname}": avg1,
                f"{alg2.algname}": avg2,
            }

            # Use the loc method to add the new row to the DataFrame
            comparison_stats[stat_index].loc[
                len(comparison_stats[stat_index])
            ] = new_row
            # check if the single best is significantly better than the avg best

    return comparison_stats


print("Loading CMA")
data_file = "cma_final_processed.pkl"
cmaes_explainer.load_results(data_file)
# use aucLarge for D30
cmaes_explainer.df.loc[cmaes_explainer.df["dim"] == 30, "auc"] = cmaes_explainer.df.loc[
    cmaes_explainer.df["dim"] == 30, "aucLarge"
]

print("Loading DE")
data_file = "de_final_processed.pkl"
de_explainer.load_results(data_file)
# auc Large
de_explainer.df.loc[de_explainer.df["dim"] == 30, "auc"] = de_explainer.df.loc[
    de_explainer.df["dim"] == 30, "aucLarge"
]

dffinal = compare(de_explainer, cmaes_explainer)

with open(f"../output/compare-p.tex", "w") as fh:
    for dim in ["d=5", "d=30"]:
        dffinal[dim].to_latex(
            buf=fh,
            index=False,
            multicolumn_format="c",
            float_format="%.2f",
            caption=f"Performance comparison of Modular CMA and Modular DE with {dim}. Boldface indicates a significant improvement either between single-best configurations, average best configurations or all configurations.",
        )

##CLUSTER PLOTS FOR FUNCTION COMPARISON (BENCHMARK ANALYSIS)
all_cors = []

df = de_explainer.df
cma_es_df = cmaes_explainer.df
for dim in de_explainer.dims:
    fid_auc_matrix = []
    fid_auc_matrix_cma = []
    for fid in de_explainer.fids:
        df_dim_fid = df[(df["dim"] == dim) & (df["fid"] == fid)]

        cma_es_df_fid = cma_es_df[(cma_es_df["dim"] == dim) & (cma_es_df["fid"] == fid)]

        fid_auc_matrix.append(df_dim_fid.groupby(de_features)["auc"].mean())
        fid_auc_matrix_cma.append(cma_es_df_fid.groupby(cma_features)["auc"].mean())

    fid_auc_matrix = np.array(fid_auc_matrix).T
    fid_auc_matrix_cma = np.array(fid_auc_matrix_cma).T

    df_matrix = pd.DataFrame(
        fid_auc_matrix, columns=[f"$f_{{{i}}}$" for i in np.arange(1, 25)]
    )
    corr = df_matrix.corr()
    all_cors.append(corr)

    df_matrix2 = pd.DataFrame(
        fid_auc_matrix_cma, columns=[f"$f_{{{i}}}$" for i in np.arange(1, 25)]
    )
    corr2 = df_matrix2.corr()
    all_cors.append(corr2)

    g = sns.clustermap(
        corr, method="complete", cmap="viridis", annot=True, annot_kws={"size": 8}
    )
    plt.tight_layout()
    plt.savefig(f"../output/de-clustermap-{dim}d.pdf")
    plt.clf()

    g = sns.heatmap(corr, annot=False, cmap="viridis")
    plt.tight_layout()
    plt.savefig(f"../output/de-heatmap-{dim}d.pdf")
    plt.clf()

    g = sns.clustermap(
        corr2, method="complete", cmap="viridis", annot=True, annot_kws={"size": 8}
    )
    plt.tight_layout()
    plt.savefig(f"../output/cma-clustermap-{dim}d.pdf")
    plt.clf()

    g = sns.heatmap(corr2, annot=False, cmap="viridis")
    plt.tight_layout()
    plt.savefig(f"../output/cma-heatmap-{dim}d.pdf")
    plt.clf()

all_cors = np.array(all_cors)

print(all_cors.shape)
min_all_cors = np.min(all_cors, axis=0)
print(min_all_cors.shape)

alldf = pd.DataFrame(min_all_cors, columns=[f"$f_{{{i}}}$" for i in np.arange(1, 25)])

g = sns.clustermap(
    alldf, method="complete", cmap="viridis", annot=True, annot_kws={"size": 8}
)
plt.tight_layout()
plt.savefig(f"../output/all-clustermap.pdf")
plt.clf()
plt.clf()
g = sns.heatmap(alldf, annot=False, cmap="viridis")
plt.tight_layout()
plt.savefig(f"../output/min-cor-all.pdf")
plt.clf()

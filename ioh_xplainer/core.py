import sys
from functools import partial
from itertools import product
from multiprocessing import Pool, cpu_count

import math
import ioh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tqdm
import xgboost
import catboost as cb
from ConfigSpace import ConfigurationSpace
from ConfigSpace.util import generate_grid
import scipy.stats as stats

from .utils import auc_logger, ioh_f0, runParallelFunction


class explainer(object):
    """Explain an iterative optimization heuristic by evaluating a large set of hyper-parameter configurations and exploring
    the hyper-parameter influences on AUC of the ECDF curves. Uses AI models and Shap to generate explainations that practitioners
    can use to learn the strengths and weaknesses of an optimization algorithm in a variety of environments.

    Attributes:
        optimizer (function): The ioh to be explained.
        config_space (ConfigurationSpace): Configuration space listing all hyper-parameters to vary.
        dims (list, optional): List of dimensions to evaluate. Defaults to [5, 10, 20].
        fids (list, optional): List of function ids to evaluate from the BBOB suite. Defaults to [1,5,7,13,18,20,23].
        iids (int, optional): Number of instances to evaluate. Defaults to 5.
        reps (int, optional): Number of random seeds to evaluate. Defaults to 5.
        sampling_method (str, optional): Either "grid" or "random". Defaults to "grid".
        seed (int, optional): The seed to start with. Defaults to 1.
        verbose (bool, optional): Output additional logging information. Defaults to False.
    """

    def __init__(
        self,
        optimizer,
        config_space,
        algname = "optimizer",
        dims=[5, 10, 20],
        fids=[1, 5, 7, 13, 18, 20, 23],
        iids=5,
        reps=5,
        sampling_method="grid",  # or random
        grid_steps_dict=None,  # used for grid sampling
        sample_size=1000,  # only used with random method
        budget=10000,
        seed=1,
        verbose=False,
    ):
        """Initialize the optimizer .

        Args:
            optimizer (function): The algorithm to be evaluated and explained, should handle the ioh problem as objective function.
            config_space (ConfigurationSpace): Configuration space listing all hyper-parameters to vary.
            algname (string, optional): Name of the algorithm. Defaults to "optimizer".
            dims (list, optional): List of dimensions to evaluate. Defaults to [5, 10, 20].
            fids (list, optional): List of function ids to evaluate from the BBOB suite. Defaults to [1,5,7,13,18,20,23].
            iids (int, optional): Number of instances to evaluate. Defaults to 5.
            reps (int, optional): Number of random seeds to evaluate. Defaults to 5.
            sampling_method (str, optional): Either "grid" or "random". Defaults to "grid".
            grid_steps_dict (dict, optional): A dictionary including number of steps per hyper-parameter. Used for "grid" sampling method. Defaults to None.
            sample_size (int, optional): The number samples for a random sample scheme. Defaults to 1000.
            budget (int, optional): The budget for evaluation (one optimization run). Defaults to 10000.
            seed (int, optional): The seed to start with. Defaults to 1.
            verbose (bool, optional): Output additional logging information. Defaults to False.
        """

        self.optimizer = optimizer
        self.algname = algname
        self.config_space = config_space
        self.dims = dims
        self.fids = fids
        self.iids = iids
        self.reps = reps
        self.sampling_method = sampling_method
        self.grid_steps_dict = grid_steps_dict
        self.sample_size = sample_size
        self.verbose = verbose
        self.budget = budget
        self.models = {}
        self.df = pd.DataFrame(
            columns=["fid", "iid", "dim", "seed", *config_space.keys(), "auc"]
        )
        np.random.seed(seed)

    def _create_grid(self):
        """Generate the configurations to evaluate."""
        if self.sampling_method == "grid":
            self.configuration_grid = generate_grid(
                self.config_space, self.grid_steps_dict
            )
        else:
            self.configuration_grid = self.config_space.sample_configuration(
                self.sample_size
            )
        if self.verbose:
            print(f"Evaluating {len(self.configuration_grid)} configurations.")

    def _run_verification(self, args):
        """Run validation on the given configurations for multiple random seeds.

        Args:
            args (list): List of [dim, fid, iid, config_id], including all information to run one configuration.

        Returns:
            list: A list of dictionaries containing the auc scores of each random repetition.
        """
        dim, fid, iid, config_i = args
        config = self.configuration_grid[config_i]
        # func = auc_func(fid, dimension=dim, instance=iid, budget=self.budget)
        func = ioh.get_problem(fid, dimension=dim, instance=iid)
        myLogger = auc_logger(self.budget, triggers=[ioh.logger.trigger.ALWAYS])
        func.attach_logger(myLogger)
        return_list = []
        for seed in range(self.reps):
            np.random.seed(seed)
            self.optimizer(func, config, budget=self.budget, dim=dim, seed=seed)
            auc = myLogger.auc
            func.reset()
            myLogger.reset(func)
            return_list.append(
                {"fid": fid, "iid": iid, "dim": dim, "seed": seed, **config, "auc": auc}
            )
        return return_list

    def run(self, paralell=True):
        """Run the evaluation of all configurations.

        Args:
            paralell (bool, optional): Use multiple threads or not. Defaults to True.
        """
        # create the configuration grid
        self._create_grid()
        # run all the optimizations
        for i in tqdm.tqdm(range(len(self.configuration_grid))):
            if paralell:
                partial_run = partial(self._run_verification)
                args = product(self.dims, self.fids, np.arange(self.iids), [i])
                res = runParallelFunction(partial_run, args)
                for tab in res:
                    for row in tab:
                        self.df.loc[len(self.df)] = row
            else:
                for dim in self.dims:
                    for fid in self.fids:
                        for iid in range(self.iids):
                            tab = self._run_verification([dim, fid, iid, i])
                            for row in tab:
                                self.df.loc[len(self.df)] = row
        if self.verbose:
            print(self.df)

    def save_results(self, filename="results.pkl"):
        """Save results to a pickle file .

        Args:
            filename (str, optional): The file to save the results to. Defaults to "results.pkl".
        """
        self.df.to_pickle(filename)

    def load_results(self, filename="results.pkl"):
        """Load results from a pickle file .

        Args:
            filename (str, optional): The filename where the results are stored. Defaults to "results.pkl".
        """
        self.df = pd.read_pickle(filename)

    def check_bias(self, config, dim, num_runs=100, file_prefix=None):
        """Runs the bias result on the given configuration .

        Args:
            config (dict): Configuration of an optimzer.
            dim (int): Dimensionality
            num_runs (int): number of runs on f0, should be either 30,50,100,200,500 or 600 (600 gives highest precision)
            file_prefix (string): prefix to store the image, if None it will show instead of save. Defaults to None.
        """
        from BIAS import BIAS
        samples = []
        f0 = ioh_f0()
        if self.verbose:
            print("Running 100 evaluations on f0 for bias detection..")
        for i in np.arange(100):
            self.optimizer(f0, config, budget=self.budget, dim=dim, seed=i)
            scaled_x = (f0.state.current_best.x + 5) / 10.0
            samples.append(scaled_x)
            f0.reset()

        samples = np.array(samples)
        test = BIAS()
        y, preds = test.predict_deep(samples)
        filename = None
        if file_prefix != None:
            filename = f"{file_prefix}_bias_{config}-{dim}.png"
        if y != "unif":
            if self.verbose:
                print(
                    f"Warning! Single best configuration shows structural bias of type {y}."
                )
            test.explain(samples, preds, filename=filename)
        return y
    
    def behaviour_stats(self, fids = None, per_fid = False):
        behaviour = {}
        #Random Robustness single best = var / (a – b)**2/12
        #instance robustness single best
        #Global robustness avg best
        if fids == None:
            fids = self.fids
        if per_fid:
            fid_behaviour = {}
            for fid in fids:
                fid_behaviour[f'f{fid}'] = self.behaviour_stats(fids = [fid])
            
            return pd.concat(fid_behaviour, axis=0)
        for dim in self.dims:
            uniform_std = math.sqrt((self.budget)**2 / 12)

            dim_df = self.df[(self.df['dim'] == dim)]
            stat_index = f"d={dim}"
            df = pd.DataFrame(columns=["Measure", "value"])
            #calculate statistics for all parameters
            var_all = dim_df['auc'].std()
            mean_performance = dim_df['auc'].mean()
            df.loc[len(df)] = {"Measure": "Algorithm stability", "value": 1 - (var_all / uniform_std)}

            #calculate statistics for avg best
            _, df_best_mean = self._get_average_best(dim_df)
            df_best_mean = df_best_mean[df_best_mean['fid'].isin(fids)] #filter on fids for auc
            var_avg_best = df_best_mean['auc'].std()
            avg_best_performance = df_best_mean['auc'].mean()
            df.loc[len(df)] = {"Measure": "Invar. avg. best", "value": 1 - (var_avg_best / uniform_std)}

            if (len(self.df['iid'].unique()) > 1):
                iid_vars_avg_best = []
                for iid in list(self.df['iid'].unique()):
                    #calculate variance per iid
                    iid_vars_avg_best.append(df_best_mean[df_best_mean['iid'] == iid]['auc'].std())
                iid_var_avg_best = np.mean(iid_vars_avg_best)
                df.loc[len(df)] = {"Measure": "S-inv. avg. best", "value": 1 - (iid_var_avg_best / uniform_std)}

            if (len(self.df['seed'].unique()) > 1):
                seed_vars_avg_best = []
                for seed in list(self.df['seed'].unique()):
                    #calculate variance per iid
                    seed_vars_avg_best.append(df_best_mean[df_best_mean['seed'] == seed]['auc'].std())
                seed_var_avg_best = np.mean(seed_vars_avg_best)
                df.loc[len(df)] = {"Measure": "I-inv. avg. best", "value": 1 - (seed_var_avg_best / uniform_std)}

            #calculate statistics for single best per function
            vars_single_best = []
            single_best_performances = []
            all_single_best_auc = []
            for fid in fids:
                fid_df = dim_df[dim_df['fid']==fid]
                _, df_single_best = self._get_single_best(fid_df)
                vars_single_best.append(df_single_best['auc'].std())
                single_best_performances.append(df_single_best['auc'].mean())
                all_single_best_auc.extend(df_single_best['auc'].values)
                seed_vars_single_best = []
                for seed in list(self.df['seed'].unique()):
                    #calculate variance per iid
                    seed_vars_single_best.append(df_single_best[df_single_best['seed'] == seed]['auc'].std())
                iid_vars_single_best = []
                for iid in list(self.df['iid'].unique()):
                    #calculate variance per iid
                    iid_vars_single_best.append(df_single_best[df_single_best['iid'] == iid]['auc'].std())
            all_single_best_auc = np.array(all_single_best_auc)
            single_best_performance = np.mean(single_best_performances)

            mean_var_single_best = np.mean(vars_single_best)
            df.loc[len(df)] = {"Measure": "Invar. single best", "value": 1 - (mean_var_single_best / uniform_std)}
            
            if (len(self.df['iid'].unique()) > 1):
                iid_var_single_best = np.mean(iid_vars_avg_best)
                df.loc[len(df)] = {"Measure": "S-inv. single best", "value": 1 - (iid_var_single_best / uniform_std)}

            if (len(self.df['seed'].unique()) > 1):
                seed_var_single_best = np.mean(seed_vars_avg_best)
                df.loc[len(df)] = {"Measure": "I-inv. single best", "value": 1 - (seed_var_single_best / uniform_std)}

            #gains for avg best and single best
            df.loc[len(df)] = {"Measure": "Average norm. perf.", "value": mean_performance / self.budget}
            df.loc[len(df)] = {"Measure": "Gain avg. best", "value": (avg_best_performance - mean_performance) / self.budget}
            df.loc[len(df)] = {"Measure": "Gain single best", "value": (single_best_performance - mean_performance) / self.budget}
            sig_avg = 0
            res = stats.ttest_ind(dim_df['auc'].values, df_best_mean['auc'].values)
            if res.pvalue < 0.05:
                sig_avg = 1

            df.loc[len(df)] = {"Measure": "sig. impr. avg best", "value": sig_avg}
            sig_single = 0
            res = stats.ttest_ind(df_best_mean['auc'].values.flatten(), all_single_best_auc.flatten())
            if res.pvalue < 0.05:
                sig_single = 1

            df.loc[len(df)] = {"Measure": "sig. impr. s. best vs avg best", "value": sig_single}
            behaviour[stat_index] = df

            #df.loc[len(df)] = {"Measure": "Exp. Max Gain of ELA", "value": single_best_performance - avg_best_performance}
            #self.behaviour[stat_index] = df

        return pd.concat(behaviour, axis=1)
    
    def _get_single_best(self, fid_df):
        name_list = [*self.config_space.keys()]
        single_best = fid_df.groupby(name_list)['auc'].mean().idxmax()
        df_single_best = fid_df
        for i in range(len(name_list)):
            df_single_best = df_single_best[df_single_best[name_list[i]] == single_best[i]]
        return single_best, df_single_best
    
    def get_single_best(self, fid, dim):
        subdf = self.df
        dim_df = subdf[subdf['dim'] == dim]
        fid_df = dim_df[dim_df['fid'] == fid]
        return self._get_single_best(fid_df)
        
    
    def _get_average_best(self, dim_df):
        name_list = [*self.config_space.keys()]
        best_mean = dim_df.groupby(name_list)['auc'].mean().idxmax()
        df_best_mean = dim_df
        for i in range(len(name_list)):
            df_best_mean = df_best_mean[df_best_mean[name_list[i]] == best_mean[i]]
        return best_mean, df_best_mean

    def get_average_best(self, dim):
        """Returns average best configuration for given dimensionality and the data belonging to it."""
        dim_df = self.df[self.df['dim'] == dim]
        return self._get_average_best(dim_df)
        

    def performance_stats(self, normalize = True, latex = False):
        """Show the performance of the algorithm, average best per dimension and single-best per fid.

        Args:
            normalize (bool, optional): Normalize the auc by using the budget. Defaults to True.
            latex (bool, optional): Formats the table for latex output. Defaults to False.

        Returns:
            [type]: [description]
        """
        self.stats = {}
        
        for dim in self.dims:
            dim_df = self.df[self.df['dim'] == dim]
            stat_index = f"d={dim}"
            if latex:
                self.stats[stat_index] = pd.DataFrame(columns=["Function", "single-best", "avg-best", "all"])
            else:
                self.stats[stat_index] = pd.DataFrame(columns=["Function", "single-best mean", "single-best std", "avg-best mean", "avg-best std", "all mean", "all std"])
            #split df per function
            #get avg best config
            _, df_best_mean = self._get_average_best(dim_df)
            for fid in self.fids:
                func = ioh.get_problem(fid, dimension=dim, instance=1)
                fid_df = dim_df[dim_df['fid'] == fid]
                _, df_single_best = self.get_single_best(fid, dim)
                
                # Define the new row to be added
                avg_best_avg = df_best_mean[df_best_mean['fid'] == fid]['auc'].mean()
                avg_best_var = df_best_mean[df_best_mean['fid'] == fid]['auc'].std()
                avg_avg = fid_df['auc'].mean()
                avg_var = fid_df['auc'].std()
                avg_single = df_single_best['auc'].mean()
                var_single = df_single_best['auc'].std()
                if (normalize):
                    avg_single = avg_single / self.budget
                    var_single = var_single / self.budget
                    avg_best_avg = avg_best_avg / self.budget
                    avg_best_var = avg_best_var / self.budget
                    avg_avg = avg_avg / self.budget
                    avg_var = avg_var / self.budget

                #single best significance
                single_sig = False
                avg_sig = False
                res = stats.ttest_ind(df_single_best['auc'].values, df_best_mean[df_best_mean['fid'] == fid]['auc'].values)
                if res.pvalue < 0.05:
                    single_sig = True
                #avg best significance
                res = stats.ttest_ind(df_best_mean[df_best_mean['fid'] == fid]['auc'].values, fid_df['auc'].values)
                if res.pvalue < 0.05:
                    avg_sig = True


                if latex:
                    new_row = {'Function': f"f{fid} {func.meta_data.name}", 
                                "single-best": f"{avg_single:.2f} ({var_single:.2f})", 
                                "avg-best": f"{avg_best_avg:.2f} ({avg_best_var:.2f})", 
                                "all": f"{avg_avg:.2f} ({avg_var:.2f})"}
                    if single_sig:
                        new_row["single-best"] = "\\textbf{"+f"{avg_single:.2f} ({var_single:.2f})"+"}"
                    if avg_sig:
                        new_row["avg-best"] = "\\textbf{"+f"{avg_best_avg:.2f} ({avg_best_var:.2f})"+"}"
                else:
                    new_row = {'Function': f"f{fid} {func.meta_data.name}", 
                               "single-best mean": avg_single, "single-best std": var_single, 
                               "avg-best mean": avg_best_avg, "avg-best std": avg_best_var, 
                               "all mean": avg_avg, "all std": avg_var}
                
                # Use the loc method to add the new row to the DataFrame
                self.stats[stat_index].loc[len(self.stats[stat_index])] = new_row
                #check if the single best is significantly better than the avg best
                
        return pd.concat(self.stats, axis=1)

    def to_latex_report(self, include_behaviour=True, filename=None, img_dir=None):
        """Generate a latex report including tables and figures

        Args:
            include_behaviour (bool, optional): Include alg stability stats or not. Defaults to True.
            filename (string, optional): To store to file, when None returns string. Defaults to None.
            img_dir (string, optional): Where to store the images, if None it will store in the base directory. Defaults to None.

        Returns:
            string: Latex string or none when writing to a file.
        """
        self.performance_stats(latex=True)
        file_content = f"% Performance stats per dimension and function for {self.algname}. Boldface for the single-best configuration indicates a significant improvement over the average best configuration (for that dimension), Boldface for the average best configuration indicates a significant improvement over the average AUC of all configurations.\n"

        concat_df = pd.concat(self.stats, axis=1)
        file_content = file_content + concat_df.to_latex(index=False, multicolumn_format = "c", caption = "Performance of single-best, average best and average algorithm performance over all configurations per function and dimension.")

        if include_behaviour:
            file_content += f"% Behaviour stats per dimension and function for {self.algname}\n"
            behaviour_df = self.behaviour_stats(per_fid=False)
            file_content = file_content + behaviour_df.to_latex(index=False, 
                                                                multicolumn_format = "c", 
                                                                float_format="%.2f",
                                                                caption = f"Algorithm stability of {self.algname}")
        #generate files and latex code for the shap summary plots
        figures_text = ""
        if img_dir == None:
            img_dir = ""

        self.plot(partial_dependence=False,
            best_config=False,
            file_prefix=f"{img_dir}/img_",
            check_bias=False,
            keep_order=True)

        num_cols = 4
        if (len(self.fids) % 4 == 0):
            num_cols = 4
        else:
            num_cols = 1
        for dim in self.dims:
            figures_text += "\\begin{figure}[t]\n\\centering\n"
            for fid_i in range(0,len(self.fids),num_cols):
                if num_cols == 4:
                    figures_text += "\t\\includegraphics[height=0.15\\textheight,trim=0mm 0mm 30mm 0mm,clip]{" \
                        + f"{img_dir}img_summary_f{self.fids[fid_i]}_d{dim}.png"+ "}\n" \
                        + "\t\\includegraphics[height=0.15\\textheight,trim=60mm 0mm 30mm 0mm,clip]{" \
                        + f"{img_dir}img_summary_f{self.fids[fid_i+1]}_d{dim}.png"+ "}\n" \
                        + "\t\\includegraphics[height=0.15\\textheight,trim=60mm 0mm 30mm 0mm,clip]{" \
                        + f"{img_dir}img_summary_f{self.fids[fid_i+2]}_d{dim}.png"+ "}\n" \
                        + "\t\\includegraphics[height=0.15\\textheight,trim=60mm 0mm 0mm 0mm,clip]{" \
                        + f"{img_dir}img_summary_f{self.fids[fid_i+3]}_d{dim}.png"+ "}\n"
                else:
                    figures_text += "\t\\includegraphics[width=0.3\\textwidth,trim=0mm 0mm 0mm 0mm,clip]{" \
                        + f"{img_dir}img_summary_f{self.fids[fid_i]}_d{dim}.png"+ "}\n"
            #caption
            figures_text += "\\caption{Hyper-parameter contributions per benchmark function for d="+str(dim)+". \\label{fig:shapxplaind"+str(dim)+"}}\n\n"
            figures_text += "\\end{figure}\n\n"

        if filename != None:
            with open(f'{filename}.tex', "w") as fh:
                fh.write(figures_text)
                fh.write(file_content)
        else:
            return figures_text + file_content
       

    def plot(
        self,
        partial_dependence=True,
        best_config=True,
        file_prefix=None,
        check_bias=True,
        keep_order=False
    ):
        """Plots the explainations for the evaluated algorithm and set of hyper-parameters.

        Args:
            partial_dependence (bool, optional): Show partial dependence plots. Defaults to True.
            best_config (bool, optional): Show force plot of the best single optimizer. Defaults to True.
            file_prefix (str, optional): Prefix for the file-name when saving figures. Defaults to None, meaning figures are not saved.
            check_bias (bool, optional): Check the best configuration for structural bias. Defaults to False.
            keep_order (bool, optional): Uses a fixed order for the features, handy if you want to plot multiple next to each other.
        """
        df = self.df
        df = df.rename(
            columns={"iid": "Instance variance", "seed": "Stochastic variance"}
        )
        categorical_columns = df.dtypes[df.dtypes == 'object'].index.to_list()
        df[categorical_columns] = df[categorical_columns].apply(lambda col:pd.Categorical(col).codes)
        #for c in categorical_columns:
        #df[c] = df[c].astype('str')
        #df[c] = df[c].astype("category")

        categorical_columns = df.dtypes[df.dtypes == 'category'].index.to_list()

        for fid in self.fids:
            for dim in self.dims:
                subdf = df[(df["fid"] == fid) & (df["dim"] == dim)]
                X = subdf[
                    [
                        *self.config_space.keys(),
                        "Instance variance",
                        "Stochastic variance",
                    ]
                ]
                y = subdf["auc"].values
                bst = cb.CatBoostRegressor(iterations=100)
                
                bst.fit(X, y,
                    cat_features=categorical_columns, verbose=False)
                # explain the model's prediction using SHAP values on the first 1000 training data samples
                explainer = shap.TreeExplainer(bst)
                shap_values = explainer.shap_values(X)

                if keep_order:
                    order = list(X.columns.values)
                    col2num = {col: i for i, col in enumerate(X.columns)}
                    order = list(map(col2num.get, order))

                    shap.plots.beeswarm(
                        explainer(X),
                        show=False,
                        order=order,
                        max_display=20,
                        color=plt.get_cmap("viridis"),
                    )
                else:
                    shap.plots.beeswarm(
                        explainer(X),
                        show=False,
                        color=plt.get_cmap("viridis"),
                    )
                axes = plt.gcf().axes
                plt.tight_layout()
                plt.xlabel(f"Hyper-parameter contributions on $f_{{{fid}}}$ in $d={dim}$")
                if file_prefix != None:
                    plt.savefig(f"{file_prefix}summary_f{fid}_d{dim}.png")
                else:
                    plt.show()
                
                plt.clf()

                if partial_dependence:
                    # show dependency plots for all features
                    for hyper_parameter in range(len(self.config_space.keys())):
                        shap.dependence_plot(
                            hyper_parameter,
                            shap_values,
                            X,
                            show=False,
                            cmap=plt.get_cmap("viridis"),
                        )
                        plt.tight_layout()
                        if file_prefix != None:
                            plt.savefig(
                                f"{file_prefix}pdp_{hyper_parameter}_f{fid}_d{dim}.png"
                            )
                        else:
                            plt.show()
                        plt.clf()

                if best_config:
                    # show force plot of best configuration
                    # get best configuration from subdf
                    best_config = np.argmax(y)
                    if self.verbose:
                        print(
                            "best config ",
                            X.iloc[best_config][self.config_space.keys()],
                            "with auc ",
                            y[best_config],
                        )
                    if check_bias:
                        self.check_bias(
                            X.iloc[best_config][self.config_space.keys()],
                            dim=dim,
                            file_prefix=file_prefix,
                        )
                    shap.force_plot(
                        y,
                        shap_values[best_config, :],
                        X.iloc[best_config],
                        matplotlib=True,
                        show=False,
                        plot_cmap="PkYg",
                    )

                    if file_prefix != None:
                        plt.savefig(f"{file_prefix}bestconfig_f{fid}_d{dim}.png")
                    else:
                        plt.show()
                    plt.clf()

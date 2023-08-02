import sys
from functools import partial
from itertools import product
from multiprocessing import Pool, cpu_count

import ioh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tqdm
import xgboost
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
    
    def performance_stats(self):
        self.stats = {}
        
        for dim in self.dims:
            dim_df = self.df[self.df['dim'] == dim]
            stat_index = f"d={dim}"
            self.stats[stat_index] = pd.DataFrame(columns=["Function", "single-best", "avg-best", "avg"])
            #split df per function
            #get avg best config
            name_list = [*self.config_space.keys()]
            best_mean = dim_df.groupby(name_list)['auc'].mean().idxmin()
            df_best_mean = dim_df
            for i in range(len(name_list)):
                df_best_mean = df_best_mean[df_best_mean[name_list[i]] == best_mean[i]]
            
            for fid in self.fids:
                func = ioh.get_problem(fid, dimension=dim, instance=1)
                fid_df = dim_df[dim_df['fid'] == fid]
                single_best = fid_df.groupby(name_list)['auc'].mean().idxmin()
                df_single_best = fid_df
                for i in range(len(name_list)):
                    df_single_best = df_single_best[df_single_best[name_list[i]] == single_best[i]]
                
                # Define the new row to be added
                avg_best_avg = df_best_mean[df_best_mean['fid'] == fid]['auc'].mean()
                avg_best_var = df_best_mean[df_best_mean['fid'] == fid]['auc'].var()
                avg_avg = fid_df['auc'].mean()
                avg_var = fid_df['auc'].var()
                new_row = {'Function': func.meta_data.name, 
                            "single-best":f"{df_single_best['auc'].mean():.2f} ({df_single_best['auc'].var():.2f})", 
                            "avg-best": f"{avg_best_avg:.2f} ({avg_best_var:.2f})", 
                            "avg": f"{avg_avg:.2f} ({avg_var:.2f})"}
                
                # Use the loc method to add the new row to the DataFrame
                self.stats[stat_index].loc[len(self.stats[stat_index])] = new_row

                #check if the single best is significantly better than the avg best
                res = stats.ttest_rel(df_single_best['auc'], df_best_mean[df_best_mean['fid'] == fid]['auc'].values)
                if res.pvalue < 0.05:
                    self.stats[stat_index].style.format(lambda x: "\\textbf{" + f'{x}' + "}", subset=(len(self.stats[stat_index])-1, "single-best"))

        return pd.concat(self.stats, axis=1)

    def to_latex_report(self, concat_dims = True, filename=None):
        if len(self.stats) == 0:
            self.performance_stats()
        if concat_dims:
            concat_df = pd.concat(self.stats, axis=1)
            if filename != None:
                with open(f'{filename}.tex', "w") as fh:
                    concat_df.to_latex(buf=fh,
                        index=False)
            else:
                print(concat_df.to_latex(
                        index=False))
        else:
            for dim in self.dims:
                df = self.stats[dim]
                if filename != None:
                    with open(f'{filename}-{dim}.tex', "w") as fh:
                        df.to_latex(buf=fh, index=False)
                else:
                    print(df.to_latex(
                            index=False))

    def plot(
        self,
        partial_dependence=True,
        best_config=True,
        file_prefix=None,
        check_bias=True,
    ):
        """Plots the explainations for the evaluated algorithm and set of hyper-parameters.

        Args:
            partial_dependence (bool, optional): Show partial dependence plots. Defaults to True.
            best_config (bool, optional): Show force plot of the best single optimizer. Defaults to True.
            file_prefix (str, optional): Prefix for the file-name when saving figures. Defaults to None, meaning figures are not saved.
            check_bias (bool, optional): Check the best configuration for structural bias. Defaults to False.
        """
        df = self.df
        df = df.rename(
            columns={"iid": "Instance variance", "seed": "Stochastic variance"}
        )
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

                # train xgboost model on experiments data (TODO show accuracy with 5-fold or something similar)
                bst = xgboost.train(
                    {"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100
                )
                # explain the model's prediction using SHAP values on the first 1000 training data samples
                explainer = shap.TreeExplainer(bst)
                shap_values = explainer.shap_values(X)

                shap.summary_plot(
                    shap_values,
                    X,
                    show=False,
                    plot_type="dot",
                    cmap=plt.get_cmap("viridis"),
                )  # layered_violin
                axes = plt.gcf().axes
                axes[0].invert_xaxis()
                plt.xlabel(f"Hyper-parameter contributions on $f_{fid}$ in $d={dim}$")
                if file_prefix != None:
                    plt.savefig(f"{file_prefix}summary_f{fid}_d{dim}.png")
                else:
                    plt.show()

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
                        if file_prefix != None:
                            plt.savefig(
                                f"{file_prefix}pdp_{hyper_parameter}_f{fid}_d{dim}.png"
                            )
                        else:
                            plt.show()

                if best_config:
                    # show force plot of best configuration
                    # get best configuration from subdf
                    best_config = np.argmin(y)
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
                        explainer.expected_value,
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

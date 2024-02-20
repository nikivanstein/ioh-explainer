import math
import os
import sys
from functools import partial
from itertools import product
from multiprocessing import Pool, cpu_count

import catboost as cb
import ioh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import shap
import tqdm
from ConfigSpace import ConfigurationSpace
from ConfigSpace.util import generate_grid
from sklearn.neighbors import KNeighborsRegressor

from .utils import (
    get_f0,
    get_query_string_from_dict,
    get_query_string_from_dict_for_others,
    intersection,
    run_verification,
    runParallelFunction,
    wrap_f0,
)


class explainer(object):
    """Explain an iterative optimization heuristic by evaluating a large set of hyper-parameter configurations and exploring
    the hyper-parameter influences on AUC of the ECDF curves. Uses AI models and Shap to generate explainations that practitioners
    can use to learn the strengths and weaknesses of an optimization algorithm in a variety of environments.

    Attributes:
        optimizer (function): The ioh to be explained.
        config_space (ConfigurationSpace): Configuration space listing all hyper-parameters to vary.
        dims (list, optional): List of dimensions to evaluate. Defaults to [5, 10, 20].
        fids (list, optional): List of function ids to evaluate from the BBOB suite. Defaults to [1,5,7,13,18,20,23].
        iids (list, optional): List of instance ids to evaluate. Defaults to [1,5].
        reps (int, optional): Number of random seeds to evaluate. Defaults to 5.
        sampling_method (str, optional): Either "grid" or "random". Defaults to "grid".
        seed (int, optional): The seed to start with. Defaults to 1.
        verbose (bool, optional): Output additional logging information. Defaults to False.
    """

    def __init__(
        self,
        optimizer,
        config_space,
        algname="optimizer",
        dims=[5, 10, 20],
        fids=[1, 5, 7, 13, 18, 20, 23],
        iids=[1, 5],
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
            iids (list, optional): Number of instances to evaluate. Defaults to [1,5].
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
        self.biastest = None
        self.df = pd.DataFrame(
            columns=[
                "fid",
                "iid",
                "dim",
                "seed",
                *config_space.keys(),
                "auc",
                "aucLarge",
            ]
        )
        np.random.seed(seed)

    def get_grid_effect(self, conf, aucs, dim, fid=None, iid=None):
        """Get the relative improvement of each feature for a specific configuration using the other grid samples.
        Only works if the sampling method is a grid.

        Args:
            conf (dict): Configuration to check.
            aucs (array): Array of the configuration to check.
            dim (int): Dimension.
            fid (int, optional): Function id to use for the comparison. Defaults to None.
            iid (int, optional): Instance id to use for the comparison. Defaults to None.

        Returns:
            dict: Dictionary with added feature results.
        """
        mean_auc = aucs["auc"].mean()
        conf_to_send = conf.copy()
        for f in self.config_space.keys():
            res_others = self.get_results_for_other_configs(
                conf_to_send, f, dim, fid, iid
            )
            mean_other_aucs = np.array(res_others).mean()
            # print(f, conf[f], (mean_auc - mean_other_aucs))
            # conf[f] = f"{conf[f]} ({(mean_auc - mean_other_aucs):.2f})"
            conf[f"{f} effect"] = mean_auc - mean_other_aucs
        return conf

    def analyse_best(
        self,
        filename=None,
        check_bias=False,
        bias_folder="bias_plots/",
        full_run=False,
        full_run_folder="/data/",
        reps=10,
        grid_effect=False,
    ):
        """Analyse the best configurations (single best and average best) per dim and fid in more detail.
        Optionally also checks the configurations for structural bias, and performs a full re-run with more random seeds with ioh analyser.

        Args:
            filename (string, optional): The file to store the dataframe in tex format, if None no file is written. Defaults to None.
            check_bias (bool, optional): Check for structural bias using the bias toolbox. Defaults to False.
            bias_folder (string, optional): The folder to store the bias visualisations. Defaults to "bias_folder"
            full_run (bool, optional): Perform a re-run with ioh analyser attached. Defaults to False.
            full_run_folder (string, optional): The folder to store the IOH logs. Defaults to "/data/".
            reps (integer, optional): number of random runs. Defaults to 10.
            grid_effect (bool, optional): to include the effect of turning modules off or not. Defaults to False.\
            
        Returns:
            DataFrame
        """
        hall_of_fame = []
        configs_to_rerun = []
        for dim in self.dims:
            dim_df = self.df[self.df["dim"] == dim].copy()

            conf, aucs = self._get_average_best(dim_df)
            configs_to_rerun.append(conf.copy())

            # check effect of each configuration option (if grid was used)
            if grid_effect:
                conf = self.get_grid_effect(conf, aucs, dim)

            if check_bias:
                conf["bias"] = self.check_bias(
                    conf, dim, file_prefix=f"{bias_folder}ab_cma"
                )
            conf["dim"] = dim
            conf["fid"] = "All"
            conf["auc"] = aucs["auc"].mean()
            hall_of_fame.append(conf)

            for fid in tqdm.tqdm(self.fids):
                fid_df = dim_df[dim_df["fid"] == fid]

                # get single best (average best over all instances)
                conf, aucs = self._get_single_best(fid_df)
                configs_to_rerun.append(conf.copy())

                if grid_effect:
                    conf = self.get_grid_effect(conf, aucs, dim, fid)

                if check_bias:
                    conf["bias"] = self.check_bias(
                        conf,
                        dim,
                        num_runs=100,
                        file_prefix=f"{bias_folder}{fid}_{dim}",
                    )
                conf["dim"] = dim
                conf["fid"] = fid
                conf["auc"] = aucs["auc"].mean()

                hall_of_fame.append(conf)
        # now replace fid, iid with features instead,
        # build multiple decision trees .. visualise -- multi-output tree vs single output trees

        hall_of_fame = pd.DataFrame.from_records(hall_of_fame)
        if filename != None:
            cols = ["dim", "fid", *self.config_space.keys(), "auc"]
            if check_bias:
                cols = ["dim", "fid", *self.config_space.keys(), "auc", "bias"]
            hall_of_fame[cols].to_latex(filename, index=False)
        if full_run:  # do as last step as it will take time
            temp_reps = self.reps
            self.reps = reps
            self.run(True, 0, None, True, full_run_folder, configs_to_rerun)
            self.reps = temp_reps
        return hall_of_fame

    def _create_grid(self):
        """Generate the configurations to evaluate."""
        if self.sampling_method == "grid":
            grid = generate_grid(self.config_space, self.grid_steps_dict)
        else:
            grid = self.config_space.sample_configuration(self.sample_size)
        if self.verbose:
            print(f"Evaluating {len(grid)} configurations.")
        return grid

    def run(
        self,
        paralell=True,
        start_index=0,
        checkpoint_file="intermediate.csv",
        full_ioh=False,
        folder_root="/data/",
        grid=None,
    ):
        """Run the evaluation of all configurations.

        Args:
            paralell (bool, optional): Use multiple threads or not. Defaults to True.
            start_index (integer, optional) : Use to restart / continue a stopped run.
            checkpoint_file (string, optional): used for storing intermediate results.
        """
        # create the configuration grid
        if grid == None:
            grid = self._create_grid()
        # run all the optimizations
        for i in tqdm.tqdm(range(start_index, len(grid))):
            alg_name = f"{self.algname}-{i}"
            if paralell:
                partial_run = partial(run_verification)
                args = product(
                    self.dims,
                    self.fids,
                    self.iids,
                    [grid[i]],
                    [self.budget],
                    [self.reps],
                    [self.optimizer],
                    [full_ioh],
                    [folder_root],
                    [alg_name],
                )
                res = runParallelFunction(partial_run, args)
                for tab in res:
                    if checkpoint_file != None:
                        df_tab = pd.DataFrame(
                            tab,
                            columns=[
                                "fid",
                                "iid",
                                "dim",
                                "seed",
                                *self.config_space.keys(),
                                "auc",
                                "aucLarge",
                            ],
                        )
                        df_tab.to_csv(
                            checkpoint_file,
                            mode="a",
                            header=not os.path.exists(checkpoint_file),
                        )
                    else:
                        for row in tab:
                            self.df.loc[len(self.df)] = row
            else:
                for dim in self.dims:
                    for fid in self.fids:
                        for iid in self.iids:
                            tab = run_verification(
                                [
                                    dim,
                                    fid,
                                    iid,
                                    grid[i],
                                    self.budget,
                                    self.reps,
                                    self.optimizer,
                                    full_ioh,
                                    folder_root,
                                    alg_name,
                                ]
                            )
                            if checkpoint_file != None:
                                df_tab = pd.DataFrame(
                                    tab,
                                    columns=[
                                        "fid",
                                        "iid",
                                        "dim",
                                        "seed",
                                        *self.config_space.keys(),
                                        "auc",
                                        "aucLarge",
                                    ],
                                )
                                df_tab.to_csv(
                                    checkpoint_file,
                                    mode="a",
                                    header=not os.path.exists(checkpoint_file),
                                )
                            else:
                                for row in tab:
                                    self.df.loc[len(self.df)] = row
        if checkpoint_file != None:
            self.df = pd.read_csv(checkpoint_file)  # read results to process.
        if self.verbose:
            print(self.df)

    def get_results_for_config(self, config: dict, dim: int, fid=None, iid=None):
        """Get the AUC result from a specific configuration in dictorionary form.

        Args:
            config (dict): The dictionairy to ask.
            dim (int): The dimensionality.
            fid (int, optional): The function id. Defaults to None (is all).
            iid (int, optional): The instance id. Defaults to None (is all).

        Returns:
            array: auc scores
        """
        dim_df = self.df[(self.df["dim"] == dim)]
        if fid != None:
            fid_df = dim_df[(dim_df["fid"] == fid)]
        else:
            fid_df = dim_df
        if iid != None:
            iid_df = fid_df[(fid_df["iid"] == iid)]
        else:
            iid_df = fid_df
        return iid_df.query(get_query_string_from_dict(config))["auc"]

    def get_results_for_other_configs(
        self, config: dict, feature: str, dim: int, fid=None, iid=None
    ):
        """Get the AUC result from variations in one feature for a specific configuration in dictorionary form.

        Args:
            config (dict): The dictionairy to ask.
            feature (string): The feature to complement (vary from the original).
            dim (int): The dimensionality.
            fid (int, optional): The function id. Defaults to None (is all).
            iid (int, optional): The instance id. Defaults to None (is all).

        Returns:
            array: auc scores
        """
        dim_df = self.df[(self.df["dim"] == dim)]
        if fid != None:
            fid_df = dim_df[(dim_df["fid"] == fid)]
        else:
            fid_df = dim_df
        if iid != None:
            iid_df = fid_df[(fid_df["iid"] == iid)]
        else:
            iid_df = fid_df
        return iid_df.query(
            get_query_string_from_dict_for_others(config.copy(), feature)
        )["auc"]

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

    def check_bias(
        self,
        config,
        dim,
        num_runs=100,
        method="both",
        return_preds=False,
        file_prefix=None,
    ):
        wrap_f0()
        """Runs the bias result on the given configuration .

        Args:
            config (dict): Configuration of an optimzer.
            dim (int): Dimensionality
            num_runs (int): number of runs on f0, should be either 30,50,100,200,500 or 600 (600 gives highest precision)
            method (string): either "deep", "stat" or "both" to use the deep extension or not. Defautls to deep.
            return_preds (boolean): To also return the predicted class probabilities or not.
            file_prefix (string): prefix to store the image, if None it will show instead of save. Defaults to None.
        """

        samples = []
        if self.verbose:
            print(f"Running {num_runs} evaluations on f0 for bias detection..")
        f0 = get_f0(dim)
        for i in np.arange(num_runs):
            self.optimizer(f0, config, budget=self.budget, dim=dim, seed=i)
            scaled_x = (f0.state.current_best.x + 5) / 10.0
            samples.append(scaled_x)
            f0.reset()

        samples = np.array(samples)
        if self.biastest == None:
            from BIAS import BIAS

            self.biastest = BIAS()
        filename = None
        filename2 = None
        if file_prefix != None:
            config_str = "_".join(f"{value}" for value in config.values())
            config_str = config_str.replace("1/2^lambda", "hp-lambda")
            filename = f"{file_prefix}_bias_deep-{dim}.png"
            filename2 = f"{file_prefix}_bias-{dim}.png"
        y = ""
        if method == "stat" or method == "both":
            preds, y = self.biastest.predict(
                samples, show_figure=True, filename=filename2
            )
        if y != "none" and (method == "deep" or method == "both"):
            y, preds = self.biastest.predict_deep(samples)

        if (
            y != "unif"
            and y != "none"
            and (method == "deep" or method == "both")
            and file_prefix != None
        ):
            if self.verbose:
                print(f"Warning! Configuration shows structural bias of type {y}.")
            self.biastest.explain(samples, preds, filename=filename)
        if return_preds:
            return y, preds
        return y

    def behaviour_stats(self, fids=None, per_fid=False):
        behaviour = {}
        # Random Robustness single best = var / (a – b)**2/12
        # instance robustness single best
        # Global robustness avg best
        if fids == None:
            fids = self.fids
        if per_fid:
            fid_behaviour = {}
            for fid in fids:
                fid_behaviour[f"f{fid}"] = self.behaviour_stats(fids=[fid])

            return pd.concat(fid_behaviour, axis=0)
        for dim in self.dims:
            uniform_std = math.sqrt(1**2 / 12)

            dim_df = self.df[(self.df["dim"] == dim)]
            stat_index = f"d={dim}"
            df = pd.DataFrame(columns=["Measure", "value"])
            # calculate statistics for all parameters
            var_all = dim_df["auc"].std()
            mean_performance = dim_df["auc"].mean()
            df.loc[len(df)] = {
                "Measure": "Algorithm stability",
                "value": 1 - (var_all / uniform_std),
            }

            # calculate statistics for avg best
            _, df_best_mean = self._get_average_best(dim_df)
            df_best_mean = df_best_mean[
                df_best_mean["fid"].isin(fids)
            ]  # filter on fids for auc
            var_avg_best = df_best_mean["auc"].std()
            avg_best_performance = df_best_mean["auc"].mean()
            df.loc[len(df)] = {
                "Measure": "Invar. avg. best",
                "value": 1 - (var_avg_best / uniform_std),
            }

            if len(self.df["iid"].unique()) > 1:
                iid_vars_avg_best = []
                for iid in list(self.df["iid"].unique()):
                    # calculate variance per iid
                    iid_vars_avg_best.append(
                        df_best_mean[df_best_mean["iid"] == iid]["auc"].std()
                    )
                iid_var_avg_best = np.mean(iid_vars_avg_best)
                df.loc[len(df)] = {
                    "Measure": "S-inv. avg. best",
                    "value": 1 - (iid_var_avg_best / uniform_std),
                }

            if len(self.df["seed"].unique()) > 1:
                seed_vars_avg_best = []
                for seed in list(self.df["seed"].unique()):
                    # calculate variance per iid
                    seed_vars_avg_best.append(
                        df_best_mean[df_best_mean["seed"] == seed]["auc"].std()
                    )
                seed_var_avg_best = np.mean(seed_vars_avg_best)
                df.loc[len(df)] = {
                    "Measure": "I-inv. avg. best",
                    "value": 1 - (seed_var_avg_best / uniform_std),
                }

            # calculate statistics for single best per function
            vars_single_best = []
            single_best_performances = []
            all_single_best_auc = []
            for fid in fids:
                fid_df = dim_df[dim_df["fid"] == fid]
                _, df_single_best = self._get_single_best(fid_df)
                vars_single_best.append(df_single_best["auc"].std())
                single_best_performances.append(df_single_best["auc"].mean())
                all_single_best_auc.extend(df_single_best["auc"].values)
                seed_vars_single_best = []
                for seed in list(self.df["seed"].unique()):
                    # calculate variance per iid
                    seed_vars_single_best.append(
                        df_single_best[df_single_best["seed"] == seed]["auc"].std()
                    )
                iid_vars_single_best = []
                for iid in list(self.df["iid"].unique()):
                    # calculate variance per iid
                    iid_vars_single_best.append(
                        df_single_best[df_single_best["iid"] == iid]["auc"].std()
                    )
            all_single_best_auc = np.array(all_single_best_auc)
            single_best_performance = np.mean(single_best_performances)

            mean_var_single_best = np.mean(vars_single_best)
            df.loc[len(df)] = {
                "Measure": "Invar. single best",
                "value": 1 - (mean_var_single_best / uniform_std),
            }

            if len(self.df["iid"].unique()) > 1:
                iid_var_single_best = np.mean(iid_vars_avg_best)
                df.loc[len(df)] = {
                    "Measure": "S-inv. single best",
                    "value": 1 - (iid_var_single_best / uniform_std),
                }

            if len(self.df["seed"].unique()) > 1:
                seed_var_single_best = np.mean(seed_vars_avg_best)
                df.loc[len(df)] = {
                    "Measure": "I-inv. single best",
                    "value": 1 - (seed_var_single_best / uniform_std),
                }

            # gains for avg best and single best
            df.loc[len(df)] = {
                "Measure": "Average norm. perf.",
                "value": mean_performance,
            }
            df.loc[len(df)] = {
                "Measure": "Gain avg. best",
                "value": (avg_best_performance - mean_performance),
            }
            df.loc[len(df)] = {
                "Measure": "Gain single best",
                "value": (single_best_performance - mean_performance),
            }
            sig_avg = 0
            res = stats.ttest_ind(dim_df["auc"].values, df_best_mean["auc"].values)
            if res.pvalue < 0.05:
                sig_avg = 1

            df.loc[len(df)] = {"Measure": "sig. impr. avg best", "value": sig_avg}
            sig_single = 0
            res = stats.ttest_ind(
                df_best_mean["auc"].values.flatten(), all_single_best_auc.flatten()
            )
            if res.pvalue < 0.05:
                sig_single = 1

            df.loc[len(df)] = {
                "Measure": "sig. impr. s. best vs avg best",
                "value": sig_single,
            }
            behaviour[stat_index] = df

            # df.loc[len(df)] = {"Measure": "Exp. Max Gain of ELA", "value": single_best_performance - avg_best_performance}
            # self.behaviour[stat_index] = df

        return pd.concat(behaviour, axis=1)

    def _get_single_best(self, fid_df, use_median=False):
        name_list = [*self.config_space.keys()]
        if use_median:
            single_best = fid_df.groupby(name_list)["auc"].median().idxmax()
        else:  # use mean
            single_best = fid_df.groupby(name_list)["auc"].mean().idxmax()
        sing_best_conf = {}
        df_single_best = fid_df
        for i in range(len(name_list)):
            df_single_best = df_single_best[
                df_single_best[name_list[i]] == single_best[i]
            ]
            sing_best_conf[name_list[i]] = single_best[i]
        return sing_best_conf, df_single_best

    def get_single_best(self, fid, dim, use_median=False):
        subdf = self.df
        dim_df = subdf[subdf["dim"] == dim]
        fid_df = dim_df[dim_df["fid"] == fid]
        return self._get_single_best(fid_df, use_median)

    def get_single_best_for_iid(self, fid, iid, dim, use_median=False):
        subdf = self.df
        dim_df = subdf[subdf["dim"] == dim]
        fid_df = dim_df[dim_df["fid"] == fid]
        iid_df = fid_df[fid_df["iid"] == iid]
        return self._get_single_best(iid_df, use_median)

    def _get_average_best(self, dim_df, use_median=False):
        name_list = [*self.config_space.keys()]
        if use_median:
            best_mean = dim_df.groupby(name_list)["auc"].median().idxmax()
        else:  # use mean
            best_mean = dim_df.groupby(name_list)["auc"].mean().idxmax()
        df_best_mean = dim_df
        average_best_conf = {}
        for i in range(len(name_list)):
            df_best_mean = df_best_mean[df_best_mean[name_list[i]] == best_mean[i]]
            average_best_conf[name_list[i]] = best_mean[i]
        return average_best_conf, df_best_mean

    def get_average_best(self, dim, use_median=False):
        """Returns average best configuration for given dimensionality and the data belonging to it."""
        dim_df = self.df[self.df["dim"] == dim]
        return self._get_average_best(dim_df, use_median)

    def performance_stats(self, normalize=False, latex=False):
        """Show the performance of the algorithm, average best per dimension and single-best per fid.

        Args:
            normalize (bool, optional): Normalize the auc by using the budget. Defaults to False as it should already be normalized.
            latex (bool, optional): Formats the table for latex output. Defaults to False.

        Returns:
            [type]: [description]
        """
        self.stats = {}
        include_function_name = True
        for dim in self.dims:
            dim_df = self.df[self.df["dim"] == dim]
            stat_index = f"d={dim}"
            if latex:
                if include_function_name:
                    self.stats[stat_index] = pd.DataFrame(
                        columns=["Function", "single-best", "avg-best", "all"]
                    )
                else:
                    self.stats[stat_index] = pd.DataFrame(
                        columns=["single-best", "avg-best", "all"]
                    )
            else:
                self.stats[stat_index] = pd.DataFrame(
                    columns=[
                        "Function",
                        "single-best mean",
                        "single-best std",
                        "avg-best mean",
                        "avg-best std",
                        "all mean",
                        "all std",
                    ]
                )
            # split df per function
            # get avg best config
            _, df_best_mean = self._get_average_best(dim_df)
            for fid in self.fids:
                func = ioh.get_problem(fid, dimension=dim, instance=1)
                fid_df = dim_df[dim_df["fid"] == fid]
                _, df_single_best = self.get_single_best(fid, dim)

                # Define the new row to be added
                avg_best_avg = df_best_mean[df_best_mean["fid"] == fid]["auc"].mean()
                avg_best_var = df_best_mean[df_best_mean["fid"] == fid]["auc"].std()
                avg_avg = fid_df["auc"].mean()
                avg_var = fid_df["auc"].std()
                avg_single = df_single_best["auc"].mean()
                var_single = df_single_best["auc"].std()
                if normalize:
                    avg_single = avg_single / self.budget
                    var_single = var_single / self.budget
                    avg_best_avg = avg_best_avg / self.budget
                    avg_best_var = avg_best_var / self.budget
                    avg_avg = avg_avg / self.budget
                    avg_var = avg_var / self.budget

                # single best significance
                single_sig = False
                avg_sig = False
                res = stats.ttest_ind(
                    df_single_best["auc"].values,
                    df_best_mean[df_best_mean["fid"] == fid]["auc"].values,
                )
                if res.pvalue < 0.05:
                    single_sig = True
                # avg best significance
                res = stats.ttest_ind(
                    df_best_mean[df_best_mean["fid"] == fid]["auc"].values,
                    fid_df["auc"].values,
                )
                if res.pvalue < 0.05:
                    avg_sig = True

                if latex:
                    if include_function_name:
                        new_row = {
                            "Function": f"f{fid} {func.meta_data.name}",
                            "single-best": f"{avg_single:.2f} ({var_single:.2f})",
                            "avg-best": f"{avg_best_avg:.2f} ({avg_best_var:.2f})",
                            "all": f"{avg_avg:.2f} ({avg_var:.2f})",
                        }
                    else:
                        new_row = {
                            "single-best": f"{avg_single:.2f} ({var_single:.2f})",
                            "avg-best": f"{avg_best_avg:.2f} ({avg_best_var:.2f})",
                            "all": f"{avg_avg:.2f} ({avg_var:.2f})",
                        }
                    if single_sig:
                        new_row["single-best"] = (
                            "\\textbf{" + f"{avg_single:.2f} ({var_single:.2f})" + "}"
                        )
                    if avg_sig:
                        new_row["avg-best"] = (
                            "\\textbf{"
                            + f"{avg_best_avg:.2f} ({avg_best_var:.2f})"
                            + "}"
                        )
                else:
                    new_row = {
                        "Function": f"f{fid} {func.meta_data.name}",
                        "single-best mean": avg_single,
                        "single-best std": var_single,
                        "avg-best mean": avg_best_avg,
                        "avg-best std": avg_best_var,
                        "all mean": avg_avg,
                        "all std": avg_var,
                    }
                # Use the loc method to add the new row to the DataFrame
                self.stats[stat_index].loc[len(self.stats[stat_index])] = new_row
                # check if the single best is significantly better than the avg best
            include_function_name = False

        return pd.concat(self.stats, axis=1)

    def to_latex_report(
        self,
        include_behaviour=True,
        include_explain=True,
        include_hall_of_fame=True,
        include_bias=False,
        filename=None,
        img_dir=None,
    ):
        """Generate a latex report including tables and figures

        Args:
            include_behaviour (bool, optional): Include alg stability stats or not. Defaults to True.
            include_explain (bool, optional): Include explainable images or not. Defaults to True.
            include_hall_of_fame (bool, optional): Include single best configurations and force plots. Defaults to True.
            include_bias (bool, optional): Include structural bias indicators for the single best solutions, can only be set to True if hall of fame is True. Defaults to False.
            filename (string, optional): To store to file, when None returns string. Defaults to None.
            img_dir (string, optional): Where to store the images, if None it will store in the base directory. Defaults to None.

        Returns:
            string: Latex string or none when writing to a file.
        """
        self.performance_stats(latex=True)
        file_content = f"% Performance stats per dimension and function for {self.algname}. Boldface for the single-best configuration indicates a significant improvement over the average best configuration (for that dimension), Boldface for the average best configuration indicates a significant improvement over the average AUC of all configurations.\n"

        concat_df = pd.concat(self.stats, axis=1)
        file_content = file_content + concat_df.to_latex(
            index=False,
            multicolumn_format="c",
            caption="Performance of single-best, average best and average algorithm performance over all configurations per function and dimension.",
        )

        if include_behaviour:
            file_content += (
                f"% Behaviour stats per dimension and function for {self.algname}\n"
            )
            behaviour_df = self.behaviour_stats(per_fid=False)
            file_content = file_content + behaviour_df.to_latex(
                index=False,
                multicolumn_format="c",
                float_format="%.2f",
                caption=f"Algorithm stability of {self.algname}",
            )
        # generate files and latex code for the shap summary plots
        figures_text = ""
        if img_dir != None:
            img_dir = f"{img_dir}/img_"
        if include_explain:
            self.explain(
                partial_dependence=False,
                best_config=include_hall_of_fame,
                file_prefix=img_dir,
                check_bias=include_bias,
                keep_order=True,
            )

            num_cols = 4
            if len(self.fids) % 4 == 0:
                num_cols = 4
            else:
                num_cols = 1
            for dim in self.dims:
                figures_text += "\\begin{figure}[t]\n\\centering\n"
                for fid_i in range(0, len(self.fids), num_cols):
                    if num_cols == 4:
                        figures_text += (
                            "\t\\includegraphics[height=0.15\\textheight,trim=0mm 0mm 30mm 0mm,clip]{"
                            + f"{img_dir}img_summary_f{self.fids[fid_i]}_d{dim}.png"
                            + "}\n"
                            + "\t\\includegraphics[height=0.15\\textheight,trim=60mm 0mm 30mm 0mm,clip]{"
                            + f"{img_dir}img_summary_f{self.fids[fid_i+1]}_d{dim}.png"
                            + "}\n"
                            + "\t\\includegraphics[height=0.15\\textheight,trim=60mm 0mm 30mm 0mm,clip]{"
                            + f"{img_dir}img_summary_f{self.fids[fid_i+2]}_d{dim}.png"
                            + "}\n"
                            + "\t\\includegraphics[height=0.15\\textheight,trim=60mm 0mm 0mm 0mm,clip]{"
                            + f"{img_dir}img_summary_f{self.fids[fid_i+3]}_d{dim}.png"
                            + "}\n"
                        )
                    else:
                        figures_text += (
                            "\t\\includegraphics[width=0.3\\textwidth,trim=0mm 0mm 0mm 0mm,clip]{"
                            + f"{img_dir}img_summary_f{self.fids[fid_i]}_d{dim}.png"
                            + "}\n"
                        )
                # caption
                figures_text += (
                    "\\caption{Hyper-parameter contributions per benchmark function for d="
                    + str(dim)
                    + ". \\label{fig:shapxplaind"
                    + str(dim)
                    + "}}\n\n"
                )
                figures_text += "\\end{figure}\n\n"

        if filename != None:
            with open(f"{filename}.tex", "w") as fh:
                fh.write(figures_text)
                fh.write(file_content)
        else:
            return figures_text + file_content

    def explain(
        self,
        partial_dependence=False,
        best_config=True,
        file_prefix=None,
        check_bias=False,
        keep_order=False,
        catboost_params=None,
    ):
        """Plots the explainations for the evaluated algorithm and set of hyper-parameters.

        Args:
            partial_dependence (bool, optional): Show partial dependence plots. Defaults to False.
            best_config (bool, optional): Show force plot of the best single optimizer. Defaults to True.
            file_prefix (str, optional): Prefix for the file-name when saving figures. Defaults to None, meaning figures are not saved.
            check_bias (bool, optional): Check the best configuration for structural bias. Defaults to False.
            keep_order (bool, optional): Uses a fixed order for the features, handy if you want to plot multiple next to each other.
        """
        use_matplotlib = True
        if file_prefix == None and hasattr(sys, "ps1"):
            # interactive mode
            use_matplotlib = False
            shap.initjs()

        if catboost_params == None:
            catboost_params = {
                "iterations": 100,
                "depth": 14,
            }  # probably a bit overkill.. this will take time.

        df = self.df.copy(True)
        df = df.rename(
            columns={"iid": "Instance variance", "seed": "Stochastic variance"}
        )
        df_display = df.copy(True)
        categorical_columns = df.dtypes[
            (df.dtypes == "object") | (df.dtypes == "category")
        ].index.to_list()
        df[categorical_columns] = df[categorical_columns].apply(
            lambda col: pd.Categorical(col).codes
        )
        df_display[categorical_columns] = df_display[categorical_columns].astype(
            "category"
        )
        # for c in categorical_columns:
        # df[c] = df[c].astype('str')
        # df[c] = df[c].astype("category")

        categorical_columns = df.dtypes[df.dtypes == "category"].index.to_list()
        for dim in self.dims:
            for fid in self.fids:
                print(f"Processing d{dim} f{fid}..")
                subdf_display = df_display[
                    (df_display["fid"] == fid) & (df_display["dim"] == dim)
                ]
                subdf_display = subdf_display.reset_index()
                subdf_display = subdf_display[
                    [
                        *self.config_space.keys(),
                        "Instance variance",
                        "Stochastic variance",
                    ]
                ]
                subdf = df[(df["fid"] == fid) & (df["dim"] == dim)]
                subdf = subdf.reset_index()
                X = subdf[
                    [
                        *self.config_space.keys(),
                        "Instance variance",
                        "Stochastic variance",
                    ]
                ]

                y = subdf["auc"].values
                if (
                    False and self.sampling_method == "grid"
                ):  # this takes waay to long to calculate all shap values.
                    # we can use a knn regressor with k=1
                    bst = KNeighborsRegressor(n_neighbors=1)
                    bst.fit(X, y)
                    print(
                        "fitted model R2 train:", bst.score(X, y)
                    )  # when using a grid, we don't care for overfitting.
                    explainer = shap.KernelExplainer(bst.predict, shap.sample(X, 10))
                else:
                    bst = cb.CatBoostRegressor(**catboost_params)
                    bst.fit(X, y, cat_features=categorical_columns, verbose=False)
                    print("fitted model R2 train:", bst.score(X, y))
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
                plt.tight_layout()
                plt.xlabel(
                    f"Hyper-parameter contributions on $f_{{{fid}}}$ in $d={dim}$"
                )
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
                    best_config_name, _ = self.get_single_best(fid, dim)
                    best_config, aucs = self._get_single_best(subdf)
                    all_confs = X.query(get_query_string_from_dict(best_config))
                    best_config_index = all_confs.index[0]
                    all_indexes = all_confs.index.to_numpy()
                    if self.verbose:
                        print(
                            "single best config ",
                            best_config_name,
                            "with mean auc ",
                            aucs["auc"].mean(),
                        )
                    if check_bias:
                        self.check_bias(
                            best_config_name,
                            dim=dim,
                            file_prefix=file_prefix,
                        )
                    shap.force_plot(
                        explainer.expected_value,
                        shap_values[best_config_index, :],
                        subdf_display.loc[best_config_index],
                        matplotlib=use_matplotlib,
                        out_names="",
                        show=(not use_matplotlib),
                        plot_cmap="viridis",
                    )
                    if use_matplotlib:
                        plt.xlabel(
                            f"Single best configuration on $f_{{{fid}}}$ in $d={dim}$"
                        )
                        # plt.tight_layout()
                        if file_prefix != None:
                            plt.savefig(
                                f"{file_prefix}singlebest_f{fid}_d{dim}.png",
                                dpi=1200,
                                transparent=True,
                                bbox_inches="tight",
                                format="png",
                            )
                        else:
                            plt.show()
                        plt.clf()

                    shap.decision_plot(
                        explainer.expected_value,
                        shap_values[all_indexes, :],
                        subdf_display.loc[all_indexes],
                        show=False,
                    )
                    plt.xlabel(
                        f"Single best configuration with all random seeds and on all instances of $f_{{{fid}}}$ in $d={dim}$"
                    )
                    plt.tight_layout()
                    if file_prefix != None:
                        plt.savefig(f"{file_prefix}singlebest_dec_f{fid}_d{dim}.png")
                    else:
                        plt.show()
                    plt.clf()


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
            res = stats.ttest_ind(
                df_single_best1["auc"].values, df_single_best2["auc"].values
            )

            if res.pvalue < 0.05:
                if avg_single1 > avg_single2:
                    single_best1 = "\\textbf{" + single_best1 + "}"
                else:
                    single_best2 = "\\textbf{" + single_best2 + "}"

            # avg best significance
            res = stats.ttest_ind(fid_df1["auc"].values, fid_df2["auc"].values)
            if res.pvalue < 0.05:
                if avg_best1 > avg_best2:
                    avg_best1 = "\\textbf{" + avg_best1 + "}"
                else:
                    avg_best2 = "\\textbf{" + avg_best2 + "}"

            # avg significance
            res = stats.ttest_ind(dim_df1["auc"].values, dim_df2["auc"].values)
            if res.pvalue < 0.05:
                if avg1 > avg2:
                    avg1 = "\\textbf{" + avg1 + "}"
                else:
                    avg2 = "\\textbf{" + avg2 + "}"

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

    return pd.concat(comparison_stats, axis=1)

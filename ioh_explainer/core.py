
from ConfigSpace import ConfigurationSpace
from ConfigSpace.util import generate_grid
from functools import partial
from multiprocessing import cpu_count
from multiprocessing import Pool
import matplotlib.pyplot as plt
import ioh
import numpy as np
import pandas as pd
import tqdm
from itertools import product
from .utils import runParallelFunction, auc_logger
import xgboost
import shap


class explainer(object):
    
    def __init__(self, optimizer, 
                 config_space , 
                 optimizer_args = None,
                 dims = [5, 10, 20, 40], 
                 fids = [1,5], 
                 iids=5, 
                 reps=5, 
                 sampling_method = "grid",  #or random
                 grid_steps_dict = None, #used for grid sampling
                 sample_size = None,  #only used with random method
                 budget = 10000,
                 seed = 1,
                 verbose = False):
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
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
        self.df = pd.DataFrame(columns = ['fid', 'iid', 'dim', 'seed', *config_space.keys() , 'auc'])
        np.random.seed(seed)

    def _create_grid(self):
        if self.sampling_method == "grid":
            self.configuration_grid = generate_grid(self.config_space, self.grid_steps_dict)
        else:
            self.configuration_grid = self.config_space.sample(self.sample_size)
        if self.verbose:
            print(f"Evaluating {len(self.configuration_grid)} configurations.")

    def _run_verification(self, args):
        dim, fid, iid, config_i = args
        config = self.configuration_grid[config_i]
        #func = auc_func(fid, dimension=dim, instance=iid, budget=self.budget)
        func = ioh.get_problem(fid, dimension=dim, instance=iid)
        myLogger = auc_logger(self.budget, func, triggers=[ioh.logger.trigger.ALWAYS])
        func.attach_logger(myLogger)
        return_list = []
        for seed in range(self.reps):
            np.random.seed(seed)
            self.optimizer(func, config, budget=self.budget, dim=dim, seed=seed, **self.optimizer_args)
            auc = myLogger.auc
            func.reset()
            myLogger.reset(func)
            return_list.append({'fid' : fid, 'iid': iid, 'dim' : dim, 'seed' : seed, **config, 'auc': auc})
        return return_list
            

    def run(self, paralell=False):
        self._create_grid()
        #execute all runs
        #run all the optimizations
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
                            tab = self._run_verification([dim,fid,iid,i])
                            for row in tab: 
                                self.df.loc[len(self.df)] = row
        if self.verbose:
            print(self.df)
    
    def save_results(self, filename="results.pkl"):
        self.df.to_pickle(filename)  

    def load_results(self, filename="results.pkl"):
        self.df = pd.read_pickle(filename)

    def train_model(self, df, fid, dim):
        if f"{fid}-{dim}" in self.models.keys():
            return self.models[f"{fid}-{dim}"]
        subdf = df[(df['fid'] == fid) & (df['dim'] == dim)]
        X = subdf[[*self.config_space.keys(), 'Instance variance', 'Stochastic variance']]
        y = subdf['auc'].values
        
        # train xgboost model on experiments data (TODO show accuracy with 5-fold or something similar)
        bst = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)
        # explain the model's prediction using SHAP values on the first 1000 training data samples
        explainer = shap.TreeExplainer(bst)
        shap_values = explainer.shap_values(X)

        self.models[f"{fid}-{dim}"] = {"model": bst, "explainer": explainer, "shap": shap_values, "X":X}
        return self.models[f"{fid}-{dim}"]
        

    def plot(self, partial_dependence=True, best_config=True, save_figs=False, prefix="res_"):
        df = self.df
        df = df.rename(columns={"iid": "Instance variance", "seed": "Stochastic variance"})
        for fid in self.fids:
            for dim in self.dims:
                model_dict = self.train_model(df, fid, dim)

                shap.summary_plot(model_dict['shap'], model_dict['X'], show=False, plot_type='dot', cmap=plt.get_cmap("viridis")) #layered_violin
                axes = plt.gcf().axes
                axes[0].invert_xaxis()
                plt.xlabel(f'Hyper-parameter contributions on $f_{fid}$ in $d={dim}$')
                if save_figs:
                    plt.savefig(f"{prefix}summary_f{fid}_d{dim}.png")
                else:
                    plt.show()

                if partial_dependence:
                    #show dependency plots for all features
                    for hyper_parameter in range(len(self.config_space.keys())):
                        shap.dependence_plot(hyper_parameter, model_dict['shap'], model_dict['X'], show=False, cmap=plt.get_cmap("viridis"))
                        if save_figs:
                            plt.savefig(f"{prefix}pdp_{hyper_parameter}_f{fid}_d{dim}.png")
                        else:
                            plt.show()
                
                if best_config:
                    #show force plot of best configuration
                    #get best configuration from subdf
                    best_config = np.argmin(y)
                    print("best config ", model_dict['X'].iloc[best_config], "with auc ", y[best_config])
                    shap.force_plot(explainer.expected_value, model_dict['shap'][best_config,:], model_dict['X'].iloc[best_config], matplotlib=True, show=False, plot_cmap="PkYg")
                    plt.title(f'Best configuration {model_dict["X"].iloc[best_config][self.config_space.keys()]}')
                    if save_figs:
                        plt.savefig(f"{prefix}bestconfig_f{fid}_d{dim}.png")
                    else:
                        plt.show()
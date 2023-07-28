
from ConfigSpace import ConfigurationSpace
from ConfigSpace.util import generate_grid
from functools import partial
from multiprocessing import cpu_count
from multiprocessing import Pool
import ioh
import numpy as np
import pandas as pd
import tqdm
from itertools import product
from .utils import runParallelFunction, auc_func
import xgboost
import shap

class explainer(object):

    def __init__(self, optimizer, 
                 config_space , 
                 optimizer_args = None,
                 dims = [5, 10, 20, 40], 
                 fids = [1,5], 
                 iids=20, 
                 reps=5, 
                 sampling_method = "grid",  #or random
                 grid_steps_dict = None, #if none uses 10 steps for each
                 sample_size = None,  #only used with random method
                 budget = 10000,
                 seed = 1,
                 verbose = False):
        self.optimizer = optimizer
        self.config_space = config_space
        self.dims = dims
        self.fids = fids
        self.iids = iids
        self.reps = reps
        self.sampling_method = sampling_method
        self.grid_steps_dict = grid_steps_dict
        if (self.grid_steps_dict == None):
            pass #TODO
        self.verbose = verbose
        self.budget = budget
        self.df = pd.DataFrame(columns = ['fid', 'iid', 'dim', 'seed', *config_space.keys() , 'auc'])
        np.random.seed(seed)

    def _create_grid(self):
        self.configuration_grid = generate_grid(self.config_space, self.grid_steps_dict)
        if self.verbose:
            print(f"Evaluating {len(self.configuration_grid)} configurations.")

    def _run_verification(self, args):
        dim, fid, iid, config_i = args
        config = self.configuration_grid[config_i]
        #func = auc_func(fid, dimension=dim, instance=iid, budget=self.budget)
        func = ioh.get_problem(fid, dimension=dim, instance=iid)
        #func.attach_logger(logger)
        for seed in range(self.reps):
            self.optimizer(func, config, budget=self.budget, dim=dim)
            y = func.state.current_best_internal.y
            func.reset()
            return {'fid' : fid, 'iid': iid, 'dim' : dim, 'seed' : seed, **config, 'auc':y} #func.auc
            

    def run(self, paralell=False):
        self._create_grid()
        #execute all runs
        #run all the optimizations
        for i in tqdm.tqdm(range(len(self.configuration_grid))):
            if paralell:
                partial_run = partial(self._run_verification)
                args = product(self.dims, self.fids, np.arange(self.iids), [i])
                res = runParallelFunction(partial_run, args)
                for row in res:
                    self.df.loc[len(self.df)] = row
            else:
                for dim in self.dims:
                    for fid in self.fids:
                        for iid in range(self.iids):
                            row = self._run_verification([dim,fid,iid,i])
                            self.df.loc[len(self.df)] = row
            
        self.df.to_pickle("df.pkl")  
        if self.verbose:
            print(self.df)

    def plot(self):
        df = self.df
        df = df.rename(columns={"iid": "Instance variance", "seed": "Stochastic variance"})
        for fid in self.fids:
            for dim in self.dims:
                subdf = df[(df['fid'] == fid) & (df['dim'] == dim)]
                X = subdf[[*self.config_space.keys(), 'Instance variance', 'Stochastic variance']]
                y = subdf['auc'].values
                    
                # train xgboost model on diabetes data:
                bst = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

                # explain the model's prediction using SHAP values on the first 1000 training data samples
                shap_values = shap.TreeExplainer(bst).shap_values(X)
                shap.summary_plot(shap_values, X)
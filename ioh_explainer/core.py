
from ConfigSpace import ConfigurationSpace
from ConfigSpace.util import generate_grid
from functools import partial
from multiprocessing import cpu_count
from multiprocessing import Pool
import ioh
import numpy as np
import pandas as pd
import progressbar
from .utils import runParallelFunction, auc_func


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
        print(self.df)
        np.random.seed(seed)


    

    def run(self):
        #execute all runs
 
        ConfigurationGrid = generate_grid(self.config_space, self.grid_steps_dict)
        if self.verbose:
            print(len(ConfigurationGrid))

        #run all the optimizations
        for cs in ConfigurationGrid:
            for dim in self.dims:
                for fid in self.fids:
                    for iid in range(self.iids):
                        func = auc_func(fid, dimension=dim, instance=iid, budget=self.budget)
                        #func.attach_logger(logger)
                        for seed in range(self.reps):
                            self.optimizer(func.f, cs, budget=self.budget, dim=dim)
                            auc = func.auc
                            self.df.loc[len(self.df)] = {'fid' : fid, 'iid': iid, 'dim' : dim, 'seed' : seed, **cs, 'auc':auc}
                            func.reset()
        self.df.to_pickle("df.pkl")  
        print(self.df)
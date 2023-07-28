
from ConfigSpace import ConfigurationSpace
from ConfigSpace.util import generate_grid
from functools import partial
from multiprocessing import cpu_count
from multiprocessing import Pool
import ioh
import numpy as np
import pandas as pd
import progressbar

def runParallelFunction(runFunction, arguments):
        """
            Return the output of runFunction for each set of arguments,
            making use of as much parallelization as possible on this system

            :param runFunction: The function that can be executed in parallel
            :param arguments:   List of tuples, where each tuple are the arguments
                                to pass to the function
            :return:
        """
        arguments = list(arguments)
        p = Pool(min(cpu_count(), len(arguments)))
        results = p.map(runFunction, arguments)
        p.close()
        return results

class auc_func():
    def __init__(self, *args, **kwargs):
        budget = kwargs.pop('budget')
        self.f = ioh.get_problem(*args, **kwargs)
        self.auc = budget
        self.budget = budget
        powers = np.round(np.linspace(8, -8, 81), decimals=1)
        self.target_values = np.power([10] * 81, powers)

    def __call__(self, x):
        if self.f.state.evaluations >= self.budget:
            return np.infty
        y = self.f(x)
        self.auc -= sum(self.f.state.current_best_internal.y > self.target_values) / 81
        return y

    def reset(self):
        self.auc = self.budget
        self.f.reset()  

class explainer(object):

    def __init__(self, optimizer, 
                 config_space , 
                 optimizer_args = None,
                 dims = [5, 10, 20, 40], 
                 fids = [1,5], 
                 iids=20, 
                 reps=5, 
                 sampling_method = "grid",  #or random
                 grid_steps = 20,
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
        self.grid_steps = grid_steps
        self.verbose = verbose
        self.budget = budget
        self.df = pd.DataFrame(columns = ['fid', 'iid', 'dim', 'seed', *config_space.keys , 'auc'])
        print(self.df)
        np.random.seed(seed)


    

    def run(self):
        #execute all runs
 
        ConfigurationGrid = generate_grid(cs, self.grid_steps)
        if self.verbose:
            print(len(ConfigurationGrid))

        #run all the optimizations
        for cs in progressbar.ProgressBar(ConfigurationGrid):
            for dim in self.dims:
                for fid in self.fids:
                    for iid in range(self.iids):
                        func = auc_func(fid, dimension=dim, instance=iid)
                        #func.attach_logger(logger)
                        for seed in range(self.reps):
                            self.optimizer(func.f, cs, budget=self.budget, dim=dim)
                            auc = func.auc
                            self.df.append({'fid' : fid, 'iid': iid, 'dim' : dim, 'seed' : seed, **cs, 'auc':auc}, ignore_index = True)
                            func.reset()
        self.df.to_pickle("df.pkl")  
        print(self.df)
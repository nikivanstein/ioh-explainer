
import sys
import argparse
import warnings
import ioh
from modde import ModularDE, Parameters
import numpy as np
import pickle
import pandas as pd
from functools import partial
from multiprocessing import cpu_count
from multiprocessing import Pool
import os.path
from os import path
from itertools import product
import copy


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
#     local_func = partial(func_star, func=runFunction)
    results = p.map(runFunction, arguments)
    p.close()
    return results


            
def run_de(func, seed, parameters, budget=None, dim=5, penalty_factor=4,
                      fixed_budget=False, target_precision=0, verbose=True, fid=1, iid=1):
    """
    Function to automatically tune a ccmaes version for a specific optimization problem

    Parameters
    ----------
    func: function
        the function to optimize, in the form of an IOH_funcion
    parameters:
        A dictionary containing the configuration of the ccmaes
    verbose:
        Whether to show extra output or not. Currently only a boolean toggle
    seed:
        The random seed for numpy
    budget:
        Maximum number of allowed function evaluations. Defaults to dimension of the problem * 1000
    target_precision:
        To what precision the function should be optimized (if optimum is known)
    penalty_factor:
        If failed to optimize target to target_precision, return budget times this factor to penalize poor runs
    fixed_budget:
        Indicated that the run is fixed-budget, meaning that splitpoint will be interpreted as a budget, and returns best-so-far precision
    Notes
    -----
    To transform any function into an IOH_function object, you can use the custom_IOH_function wrapper.
    """

    # Initialization
    np.random.seed(seed)
    if budget is None:
        budget = 10e4 * dim

    parameters['budget'] = int(budget)

    c = ModularDE(func, **parameters)
    # c.parameters.target = func.f.objective.y

    try:
        c.run()
        if verbose:
            print(f"At target: {func.state.evaluations} used, target_hit={func.state.current_best.y})")
        auc = func.auc
        return [fid, iid, seed, parameters['F'][0], parameters['CR'][0], auc]
    except Exception as e:
        if verbose:
            print(f"Found target {func.state.current_best.y} target, but exception ({e}), so run failed")
        return []


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
                
def run_verification(args):
    
    fid, dim = args
    
    folder = "data/"
    
    items = {
        #'L-SHADE' : {'mutation_base':'target', 'mutation_reference':'pbest',  'lpsr':True, 'lambda_' : 18*dim, 'use_archive' : True, 'adaptation_method_F' :'shade', 'adaptation_method_CR' : 'shade'},
        #'SHADE' : {'mutation_base':'target', 'mutation_reference':'pbest',  'lambda_' : 10*dim, 'use_archive' : True, 'adaptation_method_F' :'shade', 'adaptation_method_CR' : 'shade',  'lambda_' : 10*dim},
        #'DAS1' : {'F' : np.array([0.8]), 'CR' : np.array([0.9]),  'lambda_' : 10*dim},
        #'DAS2' : {'F' : np.array([0.8]), 'CR' : np.array([0.9]), 'mutation_base' : 'target', 'mutation_reference' : 'best',  'lambda_' : 10*dim},
        #'Qin1' : {'F' : np.array([0.9]), 'CR' : np.array([0.9]),  'lambda_' : 50},
        #'Qin2' : {'F' : np.array([0.5]), 'CR' : np.array([0.3]),  'lambda_' : 50},
        #Qin3' : {'F' : np.array([0.5]), 'CR' : np.array([0.3]), 'mutation_reference' : 'best',  'lambda_' : 50},
        #Qin4' : {'F' : np.array([0.5]), 'CR' : np.array([0.3]), 'mutation_reference' : 'best', 'mutation_n_comps' : 2,  'lambda_' : 50},
        #'Gamperle1' : {'F' : np.array([0.45]), 'CR' : np.array([0.4]), 'mutation_base' : 'best', 'mutation_n_comps' : 2, 'lambda_' : 2*dim},
        #'Gamperle2' : {'F' : np.array([0.6]), 'CR' : np.array([0.9]), 'mutation_base' : 'best', 'mutation_n_comps' : 2, 'lambda_' : 2*dim},
        #'jDE' : {'lambda_' : 10*dim, 'adaptation_method_F' :'jDE', 'adaptation_method_CR' : 'jDE',  'lambda_' : 100},
    }
    results = []
    for F in range(1,20):
        for CR in range(1,20):
   
            key = f"DAS-F{F}-CR{CR}"
            item = {'F' : np.array([F*0.05]), 'CR' : np.array([CR*0.05]),  'lambda_' : 10*dim}
            
            #lets not use a logger for now, only capture AUC    
            #logger = ioh.logger.Analyzer(root=folder, folder_name=f"F{fid}_{dim}D_{key}", algorithm_name=f"{key}")
            budget = 10000
            for iid in range(10):
                func = auc_func(fid, dimension=dim, instance=iid, budget=budget)
                #func.attach_logger(logger)
                for seed in range(5):
                    fb = True
                    res = run_de(func, seed, item, fixed_budget = fb,
                                    budget=budget, dim=dim, verbose=True, fid=fid, iid=iid)
                    results.append(copy.deepcopy(res))
                    func.reset()
    results = np.array(results)
    np.save("de-results500.npy", results)
        
if __name__=='__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    folder_loc = "data/"

    partial_run = partial(run_verification)
    args = product([1,5], [5,20,40])
    runParallelFunction(partial_run, args)


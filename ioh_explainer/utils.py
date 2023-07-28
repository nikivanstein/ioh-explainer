from multiprocessing import cpu_count
from multiprocessing import Pool
import ioh
import numpy as np
import pandas as pd
import progressbar
from itertools import product

"""
Utility functions
"""

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
        self.meta_data = self.f.meta_data
        self.state = self.f.state
        self.auc = budget
        self.budget = budget
        powers = np.round(np.linspace(8, -8, 81), decimals=1)
        self.target_values = np.power([10] * 81, powers)

    def __call__(self, x):
        if self.f.state.evaluations >= self.budget:
            return np.infty
        y = self.f(x)
        self.state = self.f.state
        self.auc -= sum(self.f.state.current_best_internal.y > self.target_values) / 81
        return y

    def reset(self):
        self.auc = self.budget
        self.f.reset()  
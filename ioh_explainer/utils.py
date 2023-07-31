from multiprocessing import cpu_count
from multiprocessing import Pool
import ioh
import numpy as np
import pandas as pd
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

class auc_logger(ioh.logger.AbstractLogger):
    def __init__(self, budget, func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auc = budget
        self.budget = budget
        self.func = func
        powers = np.round(np.linspace(8, -8, 81), decimals=1)
        self.target_values = np.power([10] * 81, powers)

    def __call__(self, log_info: ioh.LogInfo):
        self.auc -= sum(self.func.state.current_best_internal.y > self.target_values) / 81
    
    def reset(self):
        self.auc = self.budget
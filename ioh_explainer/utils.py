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

class auc_logger(ioh.logger.AbstractLogger):
    def __init__(self, budget, problem, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.problem = problem
        self.auc = budget
        self.budget = budget

    def __call__(self, log_info: ioh.LogInfo):
        print(f"triggered! y: {log_info.y}")
        self.auc -= sum(self.problem.f.state.current_best_internal.y > self.problem.target_values) / 81
        print(self.auc)
        print(log_info)
    
    def reset(self):
        self.auc = self.budget
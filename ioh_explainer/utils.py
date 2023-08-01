from itertools import product
from multiprocessing import Pool, cpu_count

import ioh
import numpy as np
import pandas as pd
from BIAS import f0

"""
Utility functions
"""


def runParallelFunction(runFunction, arguments):
    """Return the output of runFunction for each set of arguments .

    Args:
        runFunction (function): The function that can be executed in parallel
        arguments (list): List of tuples, where each tuple are the arguments to pass to the function

    Returns:
        list: Results returned by all runFunctions
    """
    arguments = list(arguments)
    p = Pool(min(cpu_count(), len(arguments)))
    results = p.map(runFunction, arguments)
    p.close()
    return results


def ioh_f0():
    """Wrapped version of the f0 objective function.

    Args:
        dim (integer): dimensionality

    Returns:
        function: ioh problem class
    """
    return ioh.wrap_problem(f0, name="f0", lb=0.0, ub=1.0)


class auc_logger(ioh.logger.AbstractLogger):
    """Auc_logger class implementing the logging module for ioh."""

    def __init__(self, budget, *args, **kwargs):
        """Initialize the logger.

        Args:
            budget (int): Evaluation budget for calculating AUC.
        """
        super().__init__(*args, **kwargs)
        self.auc = budget
        self.budget = budget
        powers = np.round(np.linspace(8, -8, 81), decimals=1)
        self.target_values = np.power([10] * 81, powers)

    def __call__(self, log_info: ioh.LogInfo):
        """Subscalculate the auc.

        Args:
            log_info (ioh.LogInfo): info about current values.
        """
        if log_info.evaluations >= self.budget:
            return
        self.auc -= sum(log_info.y > self.target_values) / 81

    def reset(self, func):
        self.auc = self.budget

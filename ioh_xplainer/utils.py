from itertools import product
from multiprocessing import Pool, cpu_count

import ioh
import numpy as np
import pandas as pd

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
    from BIAS import f0
    return ioh.wrap_problem(f0, name="f0", lb=0.0, ub=1.0)


class auc_logger(ioh.logger.AbstractLogger):
    """Auc_logger class implementing the logging module for ioh."""

    def __init__(self, budget, lower = 1e-8, upper = 1e2, scale_log = True, *args, **kwargs):
        """Initialize the logger.

        Args:
            budget (int): Evaluation budget for calculating AUC.
        """
        super().__init__(*args, **kwargs)
        self.auc = 0
        self.lower = lower
        self.upper = upper
        self.budget = budget
        self.transform = (lambda x : np.log10(x) if scale_log else (lambda x : x)) 

    def __call__(self, log_info: ioh.LogInfo):
        """Subscalculate the auc.

        Args:
            log_info (ioh.LogInfo): info about current values.
        """
        if log_info.evaluations >= self.budget:
            return
        y_value = np.clip(log_info.raw_y_best, self.lower, self.upper)
        self.auc += (self.transform(y_value) - self.transform(self.lower))/(self.transform(self.upper)-self.transform(self.lower))

    def reset(self, func):
        super().reset()
        self.auc = 0
        
def correct_auc(ioh_function, logger, budget):
    """Correct AUC values in case a run stopped before the budget was exhausted

        Args:
            ioh_function: The function in its final state (before resetting!)
            logger: The logger in its final state, so we can ensure the settings for auc calculation match
            budget: The intended maximum budget

        Returns:
            float: The normalized AUC of the run, corrected for stopped runs
        """
    fraction = (logger.transform(np.clip(ioh_function.state.current_best_internal.y, logger.lower, logger.upper)) - logger.transform(logger.lower))/(logger.transform(logger.upper)-logger.transform(logger.lower))
    return (logger.auc + np.clip(budget - ioh_function.state.evaluations, 0, budget) * fraction)/budget

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def run_verification(args):
        """Run validation on the given configurations for multiple random seeds.

        Args:
            args (list): List of [dim, fid, iid, config, budget, reps, optimizer], including all information to run one configuration.

        Returns:
            list: A list of dictionaries containing the auc scores of each random repetition.
        """
        dim, fid, iid, config, budget, reps, optimizer = args
        # func = auc_func(fid, dimension=dim, instance=iid, budget=self.budget)
        func = ioh.get_problem(fid, dimension=dim, instance=iid)
        myLogger = auc_logger(budget, triggers=[ioh.logger.trigger.ALWAYS])
        myLoggerLarge = auc_logger(budget, upper=1e8, triggers=[ioh.logger.trigger.ALWAYS])
        func.attach_logger(myLogger)
        return_list = []
        for seed in range(reps):
            np.random.seed(seed)
            optimizer(func, config, budget=budget, dim=dim, seed=seed)
            auc1 = correct_auc(func, myLogger, budget)
            auc2 = correct_auc(func, myLoggerLarge, budget)            
            func.reset()
            myLogger.reset(func)
            myLoggerLarge.reset(func)
            return_list.append(
                {"fid": fid, "iid": iid, "dim": dim, "seed": seed, **config, "auc": auc1, "aucLarge" : auc2}
            )
        return return_list

def get_query_string_from_dict(filter):
    """Get a query string from a dictionary filter to apply to a pandas Dataframme.

    Args:
        filter (dict): Dictionary with the columns and values to filter on.

    Returns:
        string: Query string.
    """
    return ' and '.join(
        [f'({key} == "{val}")' if type(val) == str else f'({key} == {val})' for key, val in filter.items()]
    )
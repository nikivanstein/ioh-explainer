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


def wrap_f0():
    from BIAS import f0

    return ioh.wrap_problem(f0, name="f0", lb=0.0, ub=1.0)


def get_f0(dim=5):
    """Wrapped version of the f0 objective function.

    Args:
        dim (integer): dimensionality

    Returns:
        function: ioh problem class
    """
    return ioh.get_problem("f0", dimension=dim)


class aoc_logger(ioh.logger.AbstractLogger):
    """aoc_logger class implementing the logging module for ioh."""

    def __init__(
        self,
        budget,
        lower=1e-8,
        upper1=1e2,
        upper2=1e8,
        scale_log=True,
        *args,
        **kwargs,
    ):
        """Initialize the logger.

        Args:
            budget (int): Evaluation budget for calculating aoc.
        """
        super().__init__(*args, **kwargs)
        self.aoc1 = 0
        self.aoc2 = 0
        self.lower = lower
        self.upper1 = upper1
        self.upper2 = upper2
        self.budget = budget
        self.transform = lambda x: np.log10(x) if scale_log else (lambda x: x)

    def __call__(self, log_info: ioh.LogInfo):
        """Subscalculate the aoc.

        Args:
            log_info (ioh.LogInfo): info about current values.
        """
        if log_info.evaluations >= self.budget:
            return
        y_value = np.clip(log_info.raw_y_best, self.lower, self.upper1)
        self.aoc1 += (self.transform(y_value) - self.transform(self.lower)) / (
            self.transform(self.upper1) - self.transform(self.lower)
        )
        y_value = np.clip(log_info.raw_y_best, self.lower, self.upper2)
        self.aoc2 += (self.transform(y_value) - self.transform(self.lower)) / (
            self.transform(self.upper2) - self.transform(self.lower)
        )

    def reset(self, func):
        super().reset()
        self.aoc1 = 0
        self.aoc2 = 0


def correct_aoc(ioh_function, logger, budget):
    """Correct aoc values in case a run stopped before the budget was exhausted

    Args:
        ioh_function: The function in its final state (before resetting!)
        logger: The logger in its final state, so we can ensure the settings for aoc calculation match
        budget: The intended maximum budget

    Returns:
        float: The normalized aoc of the run, corrected for stopped runs
    """
    fraction = (
        logger.transform(
            np.clip(
                ioh_function.state.current_best_internal.y, logger.lower, logger.upper1
            )
        )
        - logger.transform(logger.lower)
    ) / (logger.transform(logger.upper1) - logger.transform(logger.lower))
    aoc1 = (
        logger.aoc1
        + np.clip(budget - ioh_function.state.evaluations, 0, budget) * fraction
    ) / budget
    fraction = (
        logger.transform(
            np.clip(
                ioh_function.state.current_best_internal.y, logger.lower, logger.upper2
            )
        )
        - logger.transform(logger.lower)
    ) / (logger.transform(logger.upper2) - logger.transform(logger.lower))
    aoc2 = (
        logger.aoc2
        + np.clip(budget - ioh_function.state.evaluations, 0, budget) * fraction
    ) / budget

    return 1 - aoc1, 1 - aoc2


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
    (
        dim,
        fid,
        iid,
        config,
        budget,
        reps,
        optimizer,
        full_ioh,
        folder_root,
        alg_name,
    ) = args
    # func = auc_func(fid, dimension=dim, instance=iid, budget=self.budget)
    func = ioh.get_problem(fid, dimension=dim, instance=iid)
    myLoggerLarge = aoc_logger(
        budget, upper1=1e2, upper2=1e8, triggers=[ioh.logger.trigger.ALWAYS]
    )
    func.attach_logger(myLoggerLarge)
    if full_ioh:
        logger = ioh.logger.Analyzer(
            root=folder_root,
            folder_name=f"{alg_name}-{dim}-{fid}-{iid}",
            algorithm_name=alg_name,
        )
        func.attach_logger(logger)
    return_list = []
    for seed in range(reps):
        np.random.seed(seed)
        optimizer(func, config, budget=budget, dim=dim, seed=seed)
        auc1, auc2 = correct_aoc(func, myLoggerLarge, budget)
        func.reset()
        myLoggerLarge.reset(func)
        if full_ioh:
            logger.reset()
        return_list.append(
            {
                "fid": fid,
                "iid": iid,
                "dim": dim,
                "seed": seed,
                **config,
                "auc": auc1,
                "aucLarge": auc2,
            }
        )
    return return_list


def get_query_string_from_dict(filter):
    """Get a query string from a dictionary filter to apply to a pandas Dataframme.

    Args:
        filter (dict): Dictionary with the columns and values to filter on.

    Returns:
        string: Query string.
    """
    return " and ".join(
        [
            f'({key} == "{val}")' if type(val) == str else f"({key} == {val})"
            for key, val in filter.items()
        ]
    )


def get_query_string_from_dict_for_others(filter, column):
    """Get a query string from a dictionary filter to apply to a pandas Dataframme where one column is negated.

    Args:
        filter (dict): Dictionary with the columns and values to filter on.
        column (string): Column that is negated to get all other configurations.

    Returns:
        string: Query string.
    """
    to_negate_val = filter.pop(column)

    normal_items = " and ".join(
        [
            f'({key} == "{val}")' if type(val) == str else f"({key} == {val})"
            for key, val in filter.items()
        ]
    )
    negated_item = (
        f'({column} != "{to_negate_val}")'
        if type(to_negate_val) == str
        else f"({column} != {to_negate_val})"
    )
    return f"{normal_items} and {negated_item}"

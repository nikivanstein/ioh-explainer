import os

import pandas as pd
import pytest
from ConfigSpace import ConfigurationSpace
from ioh_xplainer import __version__, explainer
from scipy.optimize import differential_evolution

"""
Configuration space for test functions
"""

import traceback

import numpy as np

test_space = ConfigurationSpace(
    {
        "strategy": ["best1bin", "rand1exp"],
        "popsize": [5, 20],
    }
)  # 20k+
test_features = [
    "strategy",
    "popsize",
]
steps_dict = {}


def run_test_de(func, config, budget, dim, seed):
    result = differential_evolution(
        func=func,
        bounds=[(-5, 5)] * dim,
        strategy=config.get("strategy"),
        popsize=config.get("popsize"),
        maxiter=budget,
    )
    return result


# The main explainer object for modular CMA
test_explainer = explainer(
    run_test_de,
    test_space,
    algname="Test-DE",
    dims=[5],  # , 10, 20, 40
    fids=np.arange(1, 2),  # ,5
    iids=[1, 2],
    reps=2,
    sampling_method="grid",  # or random
    grid_steps_dict={},
    sample_size=None,  # only used with random method
    budget=1000,  # 10000
    seed=1,
    verbose=True,
)
data_file = "test_de.pkl"


def test_version():
    assert __version__ == "0.1.0"


def test_de_run():
    test_explainer.run(
        paralell=True, start_index=0, checkpoint_file="test_paralell.csv"
    )
    assert os.path.exists("test_paralell.csv")
    os.remove("test_paralell.csv")
    test_explainer.save_results(data_file)
    assert os.path.exists(data_file)
    # test loading
    test_explainer.load_results(data_file)
    assert "aucLarge" in test_explainer.df.columns

    df = test_explainer.performance_stats()
    assert len(df) == 1

    # test report
    test_explainer.to_latex_report(
        filename="test_de",
        include_behaviour=True,
        include_explain=True,
        include_bias=False,
        include_hall_of_fame=False,
    )
    assert os.path.exists("test_de.tex")
    os.remove("test_de.tex")

"""Run all CMA configurations in paraalell using the configurationspace"""
from config import run_cma, run_de
import numpy as np
import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace
from tqdm import tqdm

from iohxplainer import explainer

cma_cs = ConfigurationSpace(
    {
        "covariance": [False, True],
        "elitist": [False, True],
        "mirrored": ["nan", "mirrored", "mirrored pairwise"],
        "base_sampler": ["halton", "sobol", "gaussian"],
        "weights_option": ["default", "equal", "1/2^lambda"],
        "local_restart": ["nan", "IPOP", "BIPOP"],
        "active": [False, True],
        "step_size_adaptation": ["csa", "psr"],
        "lambda_": (1,200), #times dim
        "mu": (0.0, 1.0),  # Uniform float
    }
)  # 20k+

# The main explainer object for modular CMA
cmaes_explainer = explainer(
    run_cma,
    cma_cs,
    algname="mod-CMA",
    dims=[5, 30],  # , 10, 20, 40
    fids=np.arange(1, 25),  # ,5
    iids=[1, 2, 3, 4, 5],
    reps=3,
    sampling_method="random",  # or random
    grid_steps_dict={},
    sample_size=1000,  # only used with random method
    budget=10000,  # 10000
    seed=1,
    verbose=True,
)

cmaes_explainer.run(paralell=True, start_index=0, checkpoint_file="cma-checkpoint-random.csv")
cmaes_explainer.save_results("cma_random.pkl")

de_cs = ConfigurationSpace(
    {
        "F": (0.05, 2.0),
        "CR": (0.00001, 1.0),
        "lambda_": (1,200), #times dim
        "mutation_base": ["target", "best", "rand"],
        "mutation_reference": ["pbest", "rand", "nan", "best"],
        "mutation_n_comps": [1, 2],
        "use_archive": [False, True],
        "crossover": ["exp", "bin"],
        "adaptation_method": ["nan", "jDE", "shade"],
        "lpsr": [False, True],
    }
)

# The main explainer object for DE
de_explainer = explainer(
    run_de,
    de_cs,
    algname="mod-de",
    dims=[5, 30],  # ,10,40],#, 10, 20, 40  ,15,30
    fids=np.arange(1, 25),  # ,5
    iids=[1, 2, 3, 4, 5],  # ,5
    reps=3,  # maybe later 10? = 11 days processing time
    sampling_method="random",  # or random
    grid_steps_dict={},
    sample_size=1000,  # only used with random method
    budget=10000,  # 10000
    seed=1,
    verbose=False,
)

de_explainer.run(paralell=True, start_index=0, checkpoint_file="de-checkpoint-random.csv")
de_explainer.save_results("de_random.pkl")
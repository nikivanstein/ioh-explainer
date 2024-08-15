"""
Configuration of the CMA space
"""

import traceback

import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace
from ConfigSpace.util import generate_grid
from IPython.display import display
from modcma.c_maes import (
    ModularCMAES,
    Parameters,
    Population,
    mutation,
    options,
    parameters,
    utils,
)
from modde import ModularDE
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
        "lambda_": ["nan", "5", "10", "20", "200"],
        "mu": ["nan", "5", "10", "20"],  # Uniform float
    }
)  # 20k+

cma_default_config = {
    "covariance": False,
    "elitist": False,
    "mirrored": "nan",
    "base_sampler": "gaussian",
    "weights_option":"default",
    "local_restart": "nan",
    "active": False,
    "step_size_adaptation": "csa",
    "lambda_": 10.0,
    "mu": 5.0,  # Uniform float
}

cma_cs_bias = ConfigurationSpace(
    {
        "covariance": [False, True],
        "elitist": [False, True],
        "orthogonal": [False, True],
        "sequential": [False, True],
        "threshold": [False, True],
        "sigma": [False, True],
        "mirrored": ["nan", "mirrored", "mirrored pairwise"],
        "base_sampler": ["sobol", "gaussian", "halton"],
        "weights_option": ["default", "equal", "1/2^lambda"],
        "local_restart": ["nan", "IPOP", "BIPOP"],
        "active": [False, True],
        "step_size_adaptation": ["csa", "psr", "tpa", "msr", "xnes", "mxnes", "lpxnes"],
        "bound_correction": [
            "nan",
            "saturate",
            "mirror",
            "cotn",
            "toroidal",
            "uniform",
        ],
        "lambda_": ["20"],
        "mu": ["5"],  # Uniform float
    }
)  # 20k+

cma_features = [
    "active",
    "covariance",
    "elitist",
    "mirrored",
    "base_sampler",
    "weights_option",
    "local_restart",
    "step_size_adaptation",
    "lambda_",
    "mu",
]

cma_features_bias = [
    "active",
    "covariance",
    "elitist",
    "orthogonal",
    "sequential",
    "threshold",
    "sigma",
    "bound_correction",
    "mirrored",
    "base_sampler",
    "weights_option",
    "local_restart",
    "step_size_adaptation",
    "lambda_",
    "mu",
]
steps_dict = {}


def config_to_cma_parameters(config, dim, budget):
    # modules first
    modules = parameters.Modules()
    active = bool(config.get("active"))
    if config.get("active") == "True":
        active = True
    if config.get("active") == "False":
        active = False
    modules.active = active

    elitist = bool(config.get("elitist"))
    if config.get("elitist") == "True":
        elitist = True
    if config.get("elitist") == "False":
        elitist = False
    modules.elitist = elitist

    if "orthogonal" in config.keys():
        orthogonal = bool(config.get("orthogonal"))
        if config.get("orthogonal") == "True":
            orthogonal = True
        if config.get("orthogonal") == "False":
            orthogonal = False
        modules.orthogonal = orthogonal

    if "sigma" in config.keys():
        sigma = bool(config.get("sigma"))
        if config.get("sigma") == "True":
            sigma = True
        if config.get("sigma") == "False":
            sigma = False
        modules.sample_sigma = sigma

    if "sequential" in config.keys():
        sequential = bool(config.get("sequential"))
        if config.get("sequential") == "True":
            sequential = True
        if config.get("sequential") == "False":
            sequential = False
        modules.sequential_selection = sequential

    if "threshold" in config.keys():
        threshold = bool(config.get("threshold"))
        if config.get("threshold") == "True":
            threshold = True
        if config.get("threshold") == "False":
            threshold = False
        modules.threshold_convergence = threshold

    if "bound_correction" in config.keys():
        correction_mapping = {
            "cotn": options.CorrectionMethod.COTN,
            "mirror": options.CorrectionMethod.MIRROR,
            "nan": options.CorrectionMethod.NONE,
            "saturate": options.CorrectionMethod.SATURATE,
            "toroidal": options.CorrectionMethod.TOROIDAL,
            "uniform": options.CorrectionMethod.UNIFORM_RESAMPLE,
        }
        modules.bound_correction = correction_mapping[config.get("bound_correction")]

    mirrored_mapping = {
        "mirrored": options.Mirror.MIRRORED,
        "nan": options.Mirror.NONE,
        "mirrored pairwise": options.Mirror.PAIRWISE,
    }
    modules.mirrored = mirrored_mapping[config.get("mirrored")]

    restart_strategy_mapping = {
        "IPOP": options.RestartStrategy.IPOP,
        "nan": options.RestartStrategy.NONE,
        "BIPOP": options.RestartStrategy.BIPOP,
    }
    modules.restart_strategy = restart_strategy_mapping[config.get("local_restart")]

    sampler_mapping = {
        "sobol": options.BaseSampler.SOBOL,
        "gaussian": options.BaseSampler.GAUSSIAN,
        "halton": options.BaseSampler.HALTON,
    }
    modules.sampler = sampler_mapping[config.get("base_sampler")]

    ssa_mapping = {
        "csa": options.StepSizeAdaptation.CSA,
        "psr": options.StepSizeAdaptation.PSR,
        "lpxnes": options.StepSizeAdaptation.LPXNES,
        "msr": options.StepSizeAdaptation.MSR,
        "mxnes": options.StepSizeAdaptation.MXNES,
        "tpa": options.StepSizeAdaptation.TPA,
        "xnes": options.StepSizeAdaptation.XNES,
    }

    modules.ssa = ssa_mapping[config.get("step_size_adaptation")]

    weights_mapping = {
        "default": options.RecombinationWeights.DEFAULT,
        "equal": options.RecombinationWeights.EQUAL,
        "1/2^lambda": options.RecombinationWeights.HALF_POWER_LAMBDA,
    }
    modules.weights = weights_mapping[config.get("weights_option")]

    covariance = bool(config.get("covariance"))
    if config.get("covariance") == "True":
        covariance = True
    if config.get("covariance") == "False":
        covariance = False
    if covariance:
        modules.matrix_adaptation = options.MatrixAdaptationType.COVARIANCE
    else:
        modules.matrix_adaptation = options.MatrixAdaptationType.MATRIX

    # settings
    lam = config.get("lambda_")
    if config.get("lambda_") == "nan":
        lam = None
    else:
        lam = int(config.get("lambda_"))

    mu = config.get("mu")
    if config.get("mu") == "nan":
        mu = None
    else:
        mu = int(config.get("mu"))
    if mu == 0:
        mu = 1

    if mu != None and lam != None and mu > lam:
        # do not run, instead return
        return False
    settings = parameters.Settings(dim, modules, budget=budget, lambda0=lam, mu0=mu)
    return Parameters(settings)


def run_cma(func, config, budget, dim, *args, seed=0, **kwargs):
    utils.set_seed(seed)
    # print(seed)
    par = config_to_cma_parameters(config, dim, int(budget))
    if par == False:
        return []  # wrong mu/lambda

    # modules = parameters.Modules()
    # settings = parameters.Settings(2, modules)
    # par = Parameters(settings)
    c = ModularCMAES(par)
    try:
        c.run(func)

        return []
    except Exception as e:
        print(
            f"Found target {func.state.current_best.y} target, but exception ({e}), so run failed"
        )
        traceback.print_exc()
        print(config)
        return []


# The main explainer object for modular CMA
cmaes_explainer = explainer(
    run_cma,
    cma_cs,
    algname="mod-CMA",
    dims=[5,30],  # 5 , 10, 20, 40
    fids=np.arange(1, 25),  # ,5
    iids=[1, 2, 3, 4, 5],
    reps=3,
    sampling_method="grid",  # or random
    grid_steps_dict={},
    sample_size=None,  # only used with random method
    budget=60000,  # 10000
    seed=1,
    verbose=True,
)

bias_cmaes_explainer = explainer(
    run_cma,
    cma_cs_bias,
    algname="mod-CMA",
    dims=[30],  # , 10, 20, 40
    fids=[0],  # ,5
    iids=[1],
    reps=1,
    sampling_method="grid",  # or random
    grid_steps_dict={},
    sample_size=None,  # only used with random method
    budget=10000,  # 10000
    seed=1,
    verbose=True,
)

de_cs = ConfigurationSpace(
    {
        "F": [0.25, 0.5, 0.75, 1.25, 1.75],
        "CR": [0.05, 0.25, 0.5, 0.75, 1.0],
        "lambda_": ["nan", "2", "10"],
        "mutation_base": ["target", "best", "rand"],
        "mutation_reference": ["pbest", "rand", "nan", "best"],
        "mutation_n_comps": [1, 2],
        "use_archive": [False, True],
        "crossover": ["exp", "bin"],
        "adaptation_method": ["nan", "jDE", "shade"],
        "lpsr": [False, True],
    }
)

de_features = [
    "F",
    "CR",
    "lambda_",
    "mutation_base",
    "mutation_reference",
    "mutation_n_comps",
    "use_archive",
    "crossover",
    "adaptation_method",
    "lpsr",
]


def run_de(func, config, budget, dim, *args, **kwargs):
    """Run Modular Differential evolution with the given config and budget."""
    lam = config.get("lambda_")
    if config.get("lambda_") == "nan":
        lam = None
    else:
        lam = int(config.get("lambda_")) * dim

    mut = config.get("mutation_reference")
    if config.get("mutation_reference") == "nan":
        mut = None

    archive = config.get("use_archive")
    if config.get("use_archive") == "False":
        archive = False
    elif config.get("use_archive") == "True":
        archive = True

    lpsr = config.get("lpsr")
    if config.get("lpsr") == "False":
        lpsr = False
    elif config.get("lpsr") == "True":
        lpsr = True

    cross = config.get("crossover")
    if config.get("crossover") == "nan":
        cross = None

    adaptation_method = config.get("adaptation_method")
    if config.get("adaptation_method") == "nan":
        adaptation_method = None

    item = {
        "F": np.array([float(config.get("F"))]),
        "CR": np.array([float(config.get("CR"))]),
        "lambda_": lam,
        "mutation_base": config.get("mutation_base"),
        "mutation_reference": mut,
        "mutation_n_comps": int(config.get("mutation_n_comps")),
        "use_archive": archive,
        "crossover": cross,
        "adaptation_method_F": adaptation_method,
        "adaptation_method_CR": adaptation_method,
        "lpsr": lpsr,
    }
    item["budget"] = int(budget)
    c = ModularDE(func, **item)
    try:
        c.run()
        return []
    except Exception as e:
        print(
            f"Found target {func.state.current_best.y} target, but exception ({e}), so run failed"
        )
        traceback.print_exc()
        print(item)
        return []


# The main explainer object for DE
de_explainer = explainer(
    run_de,
    de_cs,
    algname="mod-de",
    dims=[5,30],  # 5,30],#, 10, 20, 40  ,15,30
    fids=np.arange(1, 25),  # ,5
    iids=[1, 2, 3, 4, 5],  # ,5
    reps=3,  # maybe later 10? = 11 days processing time
    sampling_method="grid",  # or random
    grid_steps_dict=steps_dict,
    sample_size=None,  # only used with random method
    budget=60000,  # 10000
    seed=1,
    verbose=False,
)

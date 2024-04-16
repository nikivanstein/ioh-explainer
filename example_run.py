import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.util import generate_grid
from IPython.display import display
from modcma.c_maes import (ModularCMAES, Parameters, Population, mutation,
                           options, parameters, utils)
from modde import ModularDE

from iohxplainer import explainer


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

    if mu != None and lam != None and mu > lam:
        # do not run, instead return
        return False
    settings = parameters.Settings(dim, modules, budget=budget, lambda0=lam, mu0=mu)
    return Parameters(settings)


def run_cma(func, config, budget, dim, *args, seed=0, **kwargs):
    utils.set_seed(seed)
    par = config_to_cma_parameters(config, dim, int(budget))
    if par == False:
        return []  # wrong mu/lambda

    # modules = parameters.Modules()
    # settings = parameters.Settings(2, modules)
    # par = Parameters(settings)
    c = ModularCMAES(par)

    try:
        # print(config)
        c(func)
        return []
    except Exception as e:
        print(
            f"Found target {func.state.current_best.y} target, but exception ({e}), so run failed"
        )
        print(config)
        return []

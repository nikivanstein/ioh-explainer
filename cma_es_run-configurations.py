"""Run all CMA configurations in paraalell using the configurationspace"""

from ConfigSpace import ConfigurationSpace
from ConfigSpace.util import generate_grid
from ioh_xplainer import explainer
from modcma.c_maes import (
    mutation,
    Population,
    Parameters,
    parameters,
    options,
    ModularCMAES,
)


import numpy as np
import traceback

"""Attributes
----------
lambda_: int = None
    The number of offspring in the population
mu: int = None
    The number of parents in the population
sigma0: float = .5
    The initial value of sigma (step size)

"""
cs = ConfigurationSpace({
    'covariance' : [False, True], 
    'elitist' : [False, True], 
    'mirrored': ['nan', 'mirrored', 'mirrored pairwise'], 
    'base_sampler': ['sobol', 'gaussian', 'halton'], 
    'weights_option': ['default', 'equal', '1/2^lambda'], 
    'local_restart': ['nan', 'IPOP', 'BIPOP'], 
    'active': [False, True],
    'step_size_adaptation': ['csa', 'psr'],
    "lambda_": ['nan',  '5', '10', '20'],
    "mu": ['nan', '5', '10', '20']             # Uniform float
}) #20k+

steps_dict = {
}


def config_to_cma_parameters(config, dim, budget):
    #modules first
    modules = parameters.Modules()
    active = bool(config.get('active'))
    if config.get('active')=="True":
        active = True
    if config.get('active')=="False":
        active = False
    modules.active = active

    elitist = bool(config.get('elitist'))
    if config.get('elitist')=="True":
        elitist = True
    if config.get('elitist')=="False":
        elitist = False
    modules.elitist = elitist
    #modules.orthogonal = config.get('orthogonal') #Not in use for me
    #modules.sample_sigma = config.get('sample_sigma') #Not in use for me
    #modules.sequential_selection  = config.get('sequential_selection') #Not in use for me
    #modules.threshold_convergence  = config.get('threshold_convergence') #Not in use for me
    # bound_correction_mapping = {'COTN': options.CorrectionMethod.COTN,
    #                             'count': options.CorrectionMethod.COUNT,
    #                             'mirror':  options.CorrectionMethod.MIRROR,
    #                             'nan':  options.CorrectionMethod.NONE,
    #                             'saturate':  options.CorrectionMethod.SATURATE,
    #                             'toroidal':  options.CorrectionMethod.TOROIDAL,
    #                             'uniform resample':  options.CorrectionMethod.UNIFORM_RESAMPLE
    #                             } #Not used for me.
    # modules.bound_correction = bound_correction_mapping[config.get('bound_correction')]
    mirrored_mapping = {'mirrored': options.Mirror.MIRRORED,
                        'nan': options.Mirror.NONE,
                        'mirrored pairwise':  options.Mirror.PAIRWISE
                        }
    modules.mirrored = mirrored_mapping[config.get('mirrored')]

    restart_strategy_mapping = {'IPOP': options.RestartStrategy.IPOP,
                        'nan': options.RestartStrategy.NONE,
                        'BIPOP':  options.RestartStrategy.BIPOP
                        }
    modules.restart_strategy = restart_strategy_mapping[config.get('local_restart')]

    sampler_mapping = {'sobol': options.BaseSampler.SOBOL,
                        'gaussian': options.BaseSampler.GAUSSIAN,
                        'halton':  options.BaseSampler.HALTON
                        }
    modules.sampler = sampler_mapping[config.get('base_sampler')]

    ssa_mapping = {'csa': options.StepSizeAdaptation.CSA,
                    'psr': options.StepSizeAdaptation.PSR,
                    'lpxnes': options.StepSizeAdaptation.LPXNES,
                    'msr': options.StepSizeAdaptation.MSR,
                    'mxnes': options.StepSizeAdaptation.MXNES,
                    'tpa': options.StepSizeAdaptation.TPA,
                    'xnes': options.StepSizeAdaptation.XNES
                    }
    modules.ssa = ssa_mapping[config.get('step_size_adaptation')]

    weights_mapping = {'default': options.RecombinationWeights.DEFAULT,
                    'equal': options.RecombinationWeights.EQUAL,
                    '1/2^lambda': options.RecombinationWeights.HALF_POWER_LAMBDA,
                    }
    modules.weights = weights_mapping[config.get('weights_option')]

    covariance = bool(config.get("covariance"))
    if config.get('covariance')=="True":
        covariance = True
    if config.get('covariance')=="False":
        covariance = False
    if covariance:
        modules.matrix_adaptation = options.MatrixAdaptationType.COVARIANCE
    else:
        modules.matrix_adaptation = options.MatrixAdaptationType.MATRIX

    #settings
    lam = config.get('lambda_')
    if config.get('lambda_') == 'nan':
        lam = None
    else:
        lam = int(config.get('lambda_'))

    mu = config.get('mu')
    if config.get('mu') == 'nan':
        mu = None
    else:
        mu = int(config.get('mu'))

    if mu != None and lam != None and mu > lam:
        #do not run, instead return
        return False
    settings = parameters.Settings(dim, modules, budget=budget, lambda0=lam, mu0=mu)
    return Parameters(settings)


def run_cma(func, config, budget, dim, *args, **kwargs):

    par = config_to_cma_parameters(config, dim, int(budget))
    if par == False:
        return [] #wrong mu/lambda

    #modules = parameters.Modules()
    #settings = parameters.Settings(2, modules)
    #par = Parameters(settings)
    c = ModularCMAES(par)
    
    try:
        #print(config)
        c(func)
        return []
    except Exception as e:
        print(f"Found target {func.state.current_best.y} target, but exception ({e}), so run failed")
        traceback.print_exc()
        print(config)
        return []




cma_explainer = explainer(run_cma, 
                 cs , 
                 algname="mod-cma",
                 dims = [5,30],#,10,40],#, 10, 20, 40  ,15,30
                 fids = np.arange(1,25), #,5
                 iids = [1,2,3,4,5], #,5 
                 reps = 3, #maybe later 10? = 11 days processing time
                 sampling_method = "grid",  #or random
                 grid_steps_dict = steps_dict,
                 sample_size = None,  #only used with random method
                 budget = 10000, #10000
                 seed = 1,
                 verbose = False)


cma_explainer.run(paralell=True, start_index = 0, checkpoint_file="intermediate_cma_cpp.csv")
#cma_explainer.run(paralell=True, )
cma_explainer.save_results("cma_results_cpp.pkl")


#de_explainer.load_results("de_results.pkl")
#x = de_explainer.df[(de_explainer.df['fid'] == 1) & (de_explainer.df['dim'] == 5)][["F","CR","lambda_"]].to_numpy()

#y = de_explainer.df[(de_explainer.df['fid'] == 1) & (de_explainer.df['dim'] == 5)]["auc"].to_numpy()
#np.savetxt("sobol/x.csv", x)
#np.savetxt("sobol/y.csv", y)

#de_explainer.plot(save_figs = True, prefix = "img/de_")

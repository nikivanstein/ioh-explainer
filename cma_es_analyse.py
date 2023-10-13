import os
import pandas as pd
import re
import numpy as np
from tqdm import tqdm
from ioh_xplainer import explainer
import pandas as pd

from ConfigSpace import ConfigurationSpace
from ConfigSpace.util import generate_grid
from IPython.display import display

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

data_file = "cma_results_cpp.pkl"
features = ['covariance','elitist', 'mirrored', 'base_sampler', 'weights_option', 'local_restart', 'step_size_adaptation', 'lambda_', 'mu']
df = pd.read_pickle(data_file)


config_dict = {}
for f in features:
    config_dict[f] = list(map(str, df[f].unique()))

config_dict['elitist'] = [False, True]
config_dict['active'] = [False, True]
config_dict['covariance'] = [False, True]

print(config_dict)
print( df['dim'].unique())
print( df['seed'].unique())
print( df['iid'].unique())

cs = ConfigurationSpace(config_dict)

print(cs)
print( df['dim'].unique())
cmaes_explainer = explainer(None, 
                 cs , 
                 algname="mod-CMA",
                 dims = [5,30],#, 10, 20, 40 
                 fids = np.arange(1,25), #,5
                 iids = df['iid'].unique(), #20 
                 reps = len( df['seed'].unique()), 
                 sampling_method = "grid",  #or random
                 grid_steps_dict = {},
                 sample_size = None,  #only used with random method
                 budget = 10000, #10000
                 seed = 1,
                 verbose = True)


cmaes_explainer.load_results(data_file)
#for feature in features:
#    cmaes_explainer.df[feature] = cmaes_explainer.df[feature].astype("category")
df = cmaes_explainer.performance_stats()
#display(df.style.bar(cmap='viridis'))


#Hall of fame plots
cmaes_explainer.explain(partial_dependence=False,
            best_config=True,
            file_prefix="cma_img_new/",
            check_bias=True,
            keep_order=True)


#cmaes_explainer.plot(partial_dependence=False, best_config=False)
#print(de_explainer.stats[5])
#df = de_explainer.stats[5]
#print(df.to_latex(index=False,
#                  float_format="{:.2f}".format,
#                  multicolumn_format = "c"
#))  
cmaes_explainer.to_latex_report(filename="mod_cma_new", img_dir="cma_img_new/")

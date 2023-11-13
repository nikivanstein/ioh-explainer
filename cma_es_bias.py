"""Check for the hall of fame if they have any structural bias.
"""

import os
import pandas as pd
import re
import numpy as np
from tqdm import tqdm
from ioh_xplainer import explainer
import pandas as pd
import ioh
from scipy.stats import qmc
from ConfigSpace import ConfigurationSpace
from ConfigSpace.util import generate_grid
from IPython.display import display
import matplotlib.pyplot as plt
from modcma.c_maes import (
    mutation,
    Population,
    Parameters,
    parameters,
    options,
    ModularCMAES,
)


data_file = "cma_final_processed.pkl"
features = ['covariance', 'elitist', 'mirrored', 'base_sampler', 'weights_option', 'local_restart', 'step_size_adaptation', 'lambda_', 'mu']
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
        print(config)
        return []

print(cs)
print( df['dim'].unique())
cmaes_explainer = explainer(run_cma, 
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

#use aucLarge for D30
cmaes_explainer.df.loc[cmaes_explainer.df["dim"] == 30,'auc'] = cmaes_explainer.df.loc[cmaes_explainer.df["dim"] == 30,'aucLarge']

hall_of_fame = []
for dim in cmaes_explainer.dims:
    dim_df = cmaes_explainer.df[cmaes_explainer.df['dim'] == dim].copy()

    conf, aucs = cmaes_explainer._get_average_best(dim_df)
    conf['bias'] = cmaes_explainer.check_bias(conf, dim, file_prefix=f"ab_cma")
    conf['dim'] = dim
    conf['fid'] = 'All'
    conf['auc'] = aucs['auc'].mean()
    hall_of_fame.append(conf)
    
    for fid in tqdm(cmaes_explainer.fids):
        fid_df = dim_df[dim_df['fid'] == fid]

        #get single best (average best over all instances)
        conf, aucs = cmaes_explainer._get_single_best(fid_df)
        conf['bias'] = cmaes_explainer.check_bias(conf, dim, num_runs=600, file_prefix=f"{fid}_cma")
        conf['dim'] = dim
        conf['fid'] = fid
        conf['mean auc'] = aucs['auc'].mean()

        
        hall_of_fame.append(conf)

#now replace fid, iid with features instead, 
#build multiple decision trees .. visualise -- multi-output tree vs single output trees

hall_of_fame = pd.DataFrame.from_records(hall_of_fame)
hall_of_fame.to_pickle("cma_es-hall_of_fame.pkl")

hall_of_fame[['dim','fid',*features, 'mean auc', 'bias']].to_latex("cma_es-hall-of-fame.tex",index=False)
pd.plotting.parallel_coordinates(
    hall_of_fame, class_column='dim', cols=features
)
plt.save("cma_es-hall-of-fame.png")
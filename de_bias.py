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
from modde import ModularDE, Parameters


data_file = "de_final_processed.pkl" #read in modular DE data
df = pd.read_pickle(data_file)

features= ['F','CR', 'lambda_','mutation_base', 'mutation_reference',
       'mutation_n_comps', 'use_archive', 'crossover', 'adaptation_method',
       'lpsr']

config_dict = {}
for f in features:
    config_dict[f] = list(map(str, df[f].unique()))

config_dict['mutation_n_comps'] = [1,2]
config_dict['use_archive'] = [False, True]
config_dict['lpsr'] = [False, True]

#for each fid, iid get the best configuration  (mean?)
cs = ConfigurationSpace(config_dict)

print(cs)

def run_de(func, config, budget, dim, *args, **kwargs):

    lam = config.get('lambda_')
    if config.get('lambda_') == 'nan':
        lam = None
    else:
        lam = int(config.get('lambda_')) * dim
        
    mut = config.get('mutation_reference')
    if config.get('mutation_reference') == 'nan':
        mut = None

    archive = config.get('use_archive')
    if config.get('use_archive') == "False":
        archive = False
    elif config.get('use_archive') == "True":
        archive = True

    lpsr = config.get('lpsr')
    if config.get('lpsr') == "False":
        lpsr = False
    elif config.get('lpsr') == "True":
        lpsr = True
    
    cross = config.get('crossover')
    if config.get('crossover') == 'nan':
        cross = None

    adaptation_method = config.get('adaptation_method')
    if config.get('adaptation_method') == 'nan':
        adaptation_method = None
    
    item = {'F': np.array([float(config.get('F'))]), 
        'CR':np.array([float(config.get('CR'))]),  
        'lambda_' : lam,
        "mutation_base": config.get('mutation_base'), 
        "mutation_reference" : mut, 
        "mutation_n_comps" : int(config.get('mutation_n_comps')), 
        "use_archive" : archive, 
        "crossover" : cross, 
        "adaptation_method_F" : adaptation_method,
        "adaptation_method_CR" : adaptation_method,
        "lpsr" : lpsr
         }
    item['budget'] = int(budget)
    c = ModularDE(func, **item)
    try:
        c.run()
        return []
    except Exception as e:
        print(f"Found target {func.state.current_best.y} target, but exception ({e}), so run failed")
        print(item)
        return []

de_explainer = explainer(run_de, 
                 cs , 
                 algname="mod-de",
                 dims = [5,30],#,10,40],#, 10, 20, 40  ,15,30
                 fids = np.arange(1,25), #,5
                 iids = [1,2,3,4,5], #,5 
                 reps = 3, #maybe later 10? = 11 days processing time
                 sampling_method = "grid",  #or random
                 grid_steps_dict = {},
                 sample_size = None,  #only used with random method
                 budget = 10000, #10000
                 seed = 1,
                 verbose = False)




de_explainer.load_results(data_file)

#use aucLarge for D30
de_explainer.df.loc[de_explainer.df["dim"] == 30,'auc'] = de_explainer.df.loc[de_explainer.df["dim"] == 30,'aucLarge']

hall_of_fame = []
for dim in de_explainer.dims:
    dim_df = de_explainer.df[de_explainer.df['dim'] == dim].copy()

    conf, aucs = de_explainer._get_average_best(dim_df)
    conf['bias'] = de_explainer.check_bias(conf, dim, file_prefix=f"ab_de")
    conf['dim'] = dim
    conf['fid'] = 'All'
    conf['mean auc'] = aucs['auc'].mean()
    hall_of_fame.append(conf)
    
    for fid in tqdm(de_explainer.fids):
        fid_df = dim_df[dim_df['fid'] == fid]

        #get single best (average best over all instances)
        conf, aucs = de_explainer._get_single_best(fid_df)
        conf['bias'] = de_explainer.check_bias(conf, dim, num_runs=600, file_prefix=f"{fid}_de")
        conf['dim'] = dim
        conf['fid'] = fid
        conf['auc'] = aucs['auc'].mean()

        
        hall_of_fame.append(conf)

#now replace fid, iid with features instead, 
#build multiple decision trees .. visualise -- multi-output tree vs single output trees

hall_of_fame = pd.DataFrame.from_records(hall_of_fame)
hall_of_fame.to_pickle("de-hall_of_fame.pkl")

hall_of_fame[['dim','fid',*features, 'mean auc', 'bias']].to_latex("de-hall-of-fame.tex",index=False)
pd.plotting.parallel_coordinates(
    hall_of_fame, class_column='dim', cols=features
)
plt.save("de-hall-of-fame.png")
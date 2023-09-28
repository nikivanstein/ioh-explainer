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



data_file = "cma_final.pkl"
features = ['elitist', 'mirrored', 'base_sampler', 'weights_option', 'local_restart', 'step_size_adaptation', 'lambda_', 'mu']
df = pd.read_pickle(data_file)


config_dict = {}
for f in features:
    config_dict[f] = list(map(str, df[f].unique()))

config_dict['elitist'] = [False, True]
config_dict['active'] = [False, True]

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

#cmaes_explainer.plot(partial_dependence=False, best_config=False)
#print(de_explainer.stats[5])
#df = de_explainer.stats[5]
#print(df.to_latex(index=False,
#                  float_format="{:.2f}".format,
#                  multicolumn_format = "c"
#))  
cmaes_explainer.to_latex_report(filename="mod_cma_new", img_dir="cma_img_new/")
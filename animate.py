import logging
import os
import sys
from functools import partial
from itertools import product
import numpy as np
import pandas as pd
import tqdm
from ConfigSpace import ConfigurationSpace, Configuration
import matplotlib.pyplot as plt

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from config import bias_cmaes_explainer, cma_features_bias, cma_cs_bias
from iohxplainer.utils import runParallelFunction
import matplotlib.animation as animation
from config import bias_cmaes_explainer, cma_features_bias

features = [
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
]
bias = pd.read_csv("cma-bias.csv")
bias = bias.drop(columns=["Unnamed: 0"])
#bias_sub = bias[(bias["elitist"] == False)].copy()
#bias = bias[(bias["bound_correction"].isna()) & (bias["elitist"] == False) & (bias["covariance"] == True)].copy()

bias_names = ["unif", "centre", "bounds"]
bias_colors = ["#f98e09", "#bc3754", "#57106e", "#000004"]
df = pd.DataFrame(
    columns=[
        *cma_features_bias,
        *bias_names,
    ]
)
bias[features] = bias[features].fillna("nan")

def scatter_hist(x, y, ax, ax_histx, ax_histy, c):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    # the scatter plot:
    ax.scatter(x, y, s=1.5, color=c)

    # now determine nice limits by hand:
    binwidth = 0.05
    
    bins = np.arange(0.0, 1.0 + binwidth, binwidth)
    ax_histx.hist(x, bins=bins, color=c)
    ax_histx.set_xlim(-0.05, 1.05)
    ax_histx.set_ylim(0, 200)
    ax_histy.hist(y, bins=bins, orientation='horizontal', color=c)
    ax_histy.set_xlim(0, 200)
    ax_histy.set_ylim(-0.05, 1.05)

config = None
configuration_index = 0
def updatefig(i, c):
    if i%10 == 0:
        print("#", end="", flush=True)
    bias_cmaes_explainer.budget = i
    samples = bias_cmaes_explainer.get_bias_samples(
        config, 2, num_runs=500
    )
    ax.clear()
    ax_histx.clear()
    ax_histy.clear()
    #add histogram to both axes!
    scatter_hist(samples[:,0],samples[:,1], ax, ax_histx, ax_histy, c)


    plt.draw()
    #if i == 0 or i == 299:
    ci = bias_colors.index(c)
    plt.savefig(f"bias_videos/{bias_type}_config_{configuration_index}_frame_{i}_{ci}.png")
    

if True:
    for bias_type in bias_names:
        #let's explore the bias_type bias
        
        topconfs_with_bias = bias.sort_values(by=[bias_type], ascending=False).head(20).copy()

        configs = topconfs_with_bias
        config_list = []
        


        for index, c in configs[features].iterrows():
            c["lambda_"] = "20"
            c["mu"] = "5"
            conf = Configuration(cma_cs_bias, dict(c))
            print(index, dict(c))
            configuration_index = index

            #fig = plt.figure()
            # Start with a square Figure.
            fig = plt.figure(figsize=(10, 10))
            # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
            # the size of the marginal axes and the main axes in both directions.
            # Also adjust the subplot parameters for a square plot.
            gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                                left=0.1, right=0.9, bottom=0.1, top=0.9,
                                wspace=0.05, hspace=0.05)
            # Create the Axes.
            ax = fig.add_subplot(gs[1, 0])
            plt.xlim((0.0,1.0))
            plt.ylim((0.0,1.0))
            ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
            ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
            # Draw the scatter plot and marginals.
            
            bias_cmaes_explainer.verbose = False
            config = conf
            #updatefig(0)
            for c in bias_colors:
                updatefig(1000, c)
            plt.clf()
            #anim = animation.FuncAnimation(fig, updatefig, 300)
            #anim.save(f"bias_videos/{bias_type}_config_{index}_.mp4", fps=30)
            
            
# 29163 {'active': False, 'covariance': False, 'elitist': True, 'orthogonal': True, 'sequential': True, 'threshold': True, 'sigma': False, 'bound_correction': 'mirror', 'mirrored': 'mirrored pairwise', 'base_sampler': 'sobol', 'weights_option': 'default', 'local_restart': 'IPOP', 'step_size_adaptation': 'tpa', 'lambda_': '20', 'mu': '5'}
# 157412 {'active': False, 'covariance': False, 'elitist': False, 'orthogonal': False, 'sequential': True, 'threshold': False, 'sigma': True, 'bound_correction': 'saturate', 'mirrored': 'nan', 'base_sampler': 'halton', 'weights_option': '1/2^lambda', 'local_restart': 'nan', 'step_size_adaptation': 'lpxnes', 'lambda_': '20', 'mu': '5'}
# 157370 {'active': False, 'covariance': False, 'elitist': False, 'orthogonal': False, 'sequential': True, 'threshold': False, 'sigma': False, 'bound_correction': 'saturate', 'mirrored': 'nan', 'base_sampler': 'halton', 'weights_option': '1/2^lambda', 'local_restart': 'nan', 'step_size_adaptation': 'lpxnes', 'lambda_': '20', 'mu': '5'}
# 319331 {'active': True, 'covariance': False, 'elitist': True, 'orthogonal': False, 'sequential': True, 'threshold': True, 'sigma': True, 'bound_correction': 'mirror', 'mirrored': 'mirrored pairwise', 'base_sampler': 'gaussian', 'weights_option': '1/2^lambda', 'local_restart': 'IPOP', 'step_size_adaptation': 'csa', 'lambda_': '20', 'mu': '5'}
# 166529 {'active': False, 'covariance': True, 'elitist': True, 'orthogonal': True, 'sequential': False, 'threshold': True, 'sigma': False, 'bound_correction': 'saturate', 'mirrored': 'nan', 'base_sampler': 'halton', 'weights_option': '1/2^lambda', 'local_restart': 'nan', 'step_size_adaptation': 'lpxnes', 'lambda_': '20', 'mu': '5'}
# 392020 {'active': True, 'covariance': False, 'elitist': True, 'orthogonal': True, 'sequential': False, 'threshold': True, 'sigma': True, 'bound_correction': 'mirror', 'mirrored': 'mirrored pairwise', 'base_sampler': 'halton', 'weights_option': 'equal', 'local_restart': 'IPOP', 'step_size_adaptation': 'mxnes', 'lambda_': '20', 'mu': '5'}
# 123526 {'active': False, 'covariance': False, 'elitist': False, 'orthogonal': True, 'sequential': False, 'threshold': True, 'sigma': True, 'bound_correction': 'toroidal', 'mirrored': 'mirrored', 'base_sampler': 'gaussian', 'weights_option': 'equal', 'local_restart': 'BIPOP', 'step_size_adaptation': 'csa', 'lambda_': '20', 'mu': '5'}
# 424102 {'active': True, 'covariance': False, 'elitist': False, 'orthogonal': False, 'sequential': False, 'threshold': True, 'sigma': True, 'bound_correction': 'uniform', 'mirrored': 'mirrored pairwise', 'base_sampler': 'halton', 'weights_option': 'equal', 'local_restart': 'nan', 'step_size_adaptation': 'xnes', 'lambda_': '20', 'mu': '5'}
# 122518 {'active': False, 'covariance': False, 'elitist': False, 'orthogonal': True, 'sequential': False, 'threshold': True, 'sigma': True, 'bound_correction': 'toroidal', 'mirrored': 'mirrored', 'base_sampler': 'gaussian', 'weights_option': 'equal', 'local_restart': 'IPOP', 'step_size_adaptation': 'csa', 'lambda_': '20', 'mu': '5'}
# 283909 {'active': True, 'covariance': False, 'elitist': True, 'orthogonal': True, 'sequential': True, 'threshold': False, 'sigma': True, 'bound_correction': 'uniform', 'mirrored': 'mirrored', 'base_sampler': 'sobol', 'weights_option': 'equal', 'local_restart': 'BIPOP', 'step_size_adaptation': 'mxnes', 'lambda_': '20', 'mu': '5'}
# 147287 {'active': False, 'covariance': False, 'elitist': False, 'orthogonal': False, 'sequential': True, 'threshold': True, 'sigma': False, 'bound_correction': 'nan', 'mirrored': 'nan', 'base_sampler': 'halton', 'weights_option': '1/2^lambda', 'local_restart': 'BIPOP', 'step_size_adaptation': 'mxnes', 'lambda_': '20', 'mu': '5'}
# 11866 {'active': False, 'covariance': True, 'elitist': True, 'orthogonal': False, 'sequential': True, 'threshold': True, 'sigma': False, 'bound_correction': 'nan', 'mirrored': 'mirrored pairwise', 'base_sampler': 'sobol', 'weights_option': 'equal', 'local_restart': 'BIPOP', 'step_size_adaptation': 'msr', 'lambda_': '20', 'mu': '5'}
# 153743 {'active': False, 'covariance': True, 'elitist': False, 'orthogonal': True, 'sequential': False, 'threshold': True, 'sigma': False, 'bound_correction': 'nan', 'mirrored': 'mirrored', 'base_sampler': 'halton', 'weights_option': '1/2^lambda', 'local_restart': 'BIPOP', 'step_size_adaptation': 'msr', 'lambda_': '20', 'mu': '5'}
# 7462 {'active': False, 'covariance': True, 'elitist': False, 'orthogonal': False, 'sequential': False, 'threshold': True, 'sigma': True, 'bound_correction': 'nan', 'mirrored': 'mirrored', 'base_sampler': 'sobol', 'weights_option': 'equal', 'local_restart': 'IPOP', 'step_size_adaptation': 'xnes', 'lambda_': '20', 'mu': '5'}
# 1840 {'active': False, 'covariance': False, 'elitist': False, 'orthogonal': False, 'sequential': True, 'threshold': True, 'sigma': True, 'bound_correction': 'nan', 'mirrored': 'mirrored pairwise', 'base_sampler': 'sobol', 'weights_option': 'equal', 'local_restart': 'IPOP', 'step_size_adaptation': 'mxnes', 'lambda_': '20', 'mu': '5'}
# 16257 {'active': False, 'covariance': False, 'elitist': True, 'orthogonal': False, 'sequential': True, 'threshold': True, 'sigma': True, 'bound_correction': 'saturate', 'mirrored': 'nan', 'base_sampler': 'sobol', 'weights_option': 'default', 'local_restart': 'IPOP', 'step_size_adaptation': 'csa', 'lambda_': '20', 'mu': '5'}
# 161719 {'active': False, 'covariance': False, 'elitist': True, 'orthogonal': False, 'sequential': True, 'threshold': False, 'sigma': False, 'bound_correction': 'saturate', 'mirrored': 'mirrored', 'base_sampler': 'halton', 'weights_option': 'equal', 'local_restart': 'IPOP', 'step_size_adaptation': 'msr', 'lambda_': '20', 'mu': '5'}
# 89773 {'active': False, 'covariance': False, 'elitist': True, 'orthogonal': False, 'sequential': False, 'threshold': False, 'sigma': True, 'bound_correction': 'saturate', 'mirrored': 'nan', 'base_sampler': 'gaussian', 'weights_option': 'equal', 'local_restart': 'BIPOP', 'step_size_adaptation': 'msr', 'lambda_': '20', 'mu': '5'}
# 89772 {'active': False, 'covariance': False, 'elitist': True, 'orthogonal': False, 'sequential': False, 'threshold': False, 'sigma': True, 'bound_correction': 'saturate', 'mirrored': 'nan', 'base_sampler': 'gaussian', 'weights_option': 'default', 'local_restart': 'BIPOP', 'step_size_adaptation': 'msr', 'lambda_': '20', 'mu': '5'}
# 89771 {'active': False, 'covariance': False, 'elitist': True, 'orthogonal': False, 'sequential': False, 'threshold': True, 'sigma': True, 'bound_correction': 'saturate', 'mirrored': 'nan', 'base_sampler': 'gaussian', 'weights_option': '1/2^lambda', 'local_restart': 'BIPOP', 'step_size_adaptation': 'tpa', 'lambda_': '20', 'mu': '5'}
# 247171 {'active': True, 'covariance': False, 'elitist': True, 'orthogonal': True, 'sequential': False, 'threshold': False, 'sigma': True, 'bound_correction': 'mirror', 'mirrored': 'nan', 'base_sampler': 'sobol', 'weights_option': 'equal', 'local_restart': 'BIPOP', 'step_size_adaptation': 'csa', 'lambda_': '20', 'mu': '5'}
# 292390 {'active': True, 'covariance': False, 'elitist': False, 'orthogonal': False, 'sequential': False, 'threshold': True, 'sigma': True, 'bound_correction': 'nan', 'mirrored': 'nan', 'base_sampler': 'gaussian', 'weights_option': 'equal', 'local_restart': 'BIPOP', 'step_size_adaptation': 'xnes', 'lambda_': '20', 'mu': '5'}
# 292396 {'active': True, 'covariance': False, 'elitist': False, 'orthogonal': False, 'sequential': False, 'threshold': True, 'sigma': True, 'bound_correction': 'nan', 'mirrored': 'nan', 'base_sampler': 'gaussian', 'weights_option': 'equal', 'local_restart': 'BIPOP', 'step_size_adaptation': 'mxnes', 'lambda_': '20', 'mu': '5'}
# 292395 {'active': True, 'covariance': False, 'elitist': False, 'orthogonal': False, 'sequential': False, 'threshold': True, 'sigma': True, 'bound_correction': 'nan', 'mirrored': 'nan', 'base_sampler': 'gaussian', 'weights_option': 'default', 'local_restart': 'BIPOP', 'step_size_adaptation': 'mxnes', 'lambda_': '20', 'mu': '5'}
# 292394 {'active': True, 'covariance': False, 'elitist': False, 'orthogonal': False, 'sequential': False, 'threshold': False, 'sigma': True, 'bound_correction': 'nan', 'mirrored': 'nan', 'base_sampler': 'gaussian', 'weights_option': '1/2^lambda', 'local_restart': 'BIPOP', 'step_size_adaptation': 'mxnes', 'lambda_': '20', 'mu': '5'}
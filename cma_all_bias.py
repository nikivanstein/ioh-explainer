import logging
import os
import sys
from functools import partial
from itertools import product

import numpy as np
import pandas as pd
import tqdm

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from config import bias_cmaes_explainer, cma_features_bias
from iohxplainer.utils import runParallelFunction

# data_file = "cma_bias_processed.pkl"
# df = pd.read_pickle(data_file)

# cmaes_explainer.load_results(data_file)
# use aucLarge for D30
bias_names = ["unif", "centre", "disc", "bounds", "clusters"]
df = pd.DataFrame(
    columns=[
        *cma_features_bias,
        *bias_names,
    ]
)

grid = bias_cmaes_explainer._create_grid()
bias_cmaes_explainer.budget = 5000
bias_cmaes_explainer.verbose = False


def check_bias(args):
    stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")
    (
        config,
        alg_name,
    ) = args
    y, preds = bias_cmaes_explainer.check_bias(
        config, 5, num_runs=100, method="deep", return_preds=True
    )
    pred_mean = np.mean(np.array(preds), axis=0).flatten()
    return pred_mean


stepsize = 40
for i in tqdm.tqdm(range(0, len(grid), stepsize)):
    alg_name = f"ModCMA-{i}"
    configs = grid[i : min(i + stepsize, len(grid))]
    partial_run = partial(check_bias)
    args = product(
        configs,
        [alg_name],
    )
    res = runParallelFunction(partial_run, args)

    for i in range(len(configs)):
        pred = res[i]
        c = dict(configs[i])  # change to dict
        for p in range(len(pred)):
            c[bias_names[p]] = pred[p]

        df.loc[len(df)] = c

    df.to_csv(
        "cma-bias.csv",
        mode="a",
        header=not os.path.exists("cma-bias.csv"),
    )
    df = pd.DataFrame(
        columns=[
            *cma_features_bias,
            *bias_names,
        ]
    )

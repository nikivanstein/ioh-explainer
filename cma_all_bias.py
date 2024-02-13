
import pandas as pd
import tqdm
import os
import numpy as np
import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from config import bias_cmaes_explainer, cma_features_bias

#data_file = "cma_bias_processed.pkl"
#df = pd.read_pickle(data_file)

#cmaes_explainer.load_results(data_file)
# use aucLarge for D30
bias_names = ["unif","centre","disc","bounds","clusters"]
df = pd.DataFrame(
            columns=[
                *cma_features_bias,
                *bias_names,
            ]
        )

grid = bias_cmaes_explainer._create_grid()
bias_cmaes_explainer.budget = 5000
bias_cmaes_explainer.verbose = False
for i in tqdm.tqdm(range(0, len(grid))):
    alg_name = f"ModCMA-{i}"
    row = grid[i].get_array() #change to dict

    y,preds = bias_cmaes_explainer.check_bias(grid[i], 5, num_runs=100, method="deep", return_preds=True)
    #print(y,preds)
    
    pred_mean = np.mean(np.array(preds), axis=0).flatten()
    row = np.hstack((row,pred_mean))

    df.loc[len(df)] = row
    if (i%100==0):
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
"""Extract ELA features for each instance and each best configuration.
To be used with cma_es_AAC (auto algorithm configuration)
"""


## different models we can make:
# ELA/DOE + config --> AUC
# ELA ---> config
# high-level --> config
# DOE --> config

import ioh
import numpy as np

# first collect all the data
import pandas as pd
from ConfigSpace import ConfigurationSpace
from pflacco.classical_ela_features import (
    calculate_cm_grad,
    calculate_dispersion,
    calculate_ela_distribution,
    calculate_ela_level,
    calculate_ela_meta,
    calculate_information_content,
    calculate_nbc,
)
from pflacco.local_optima_network_features import (
    calculate_lon_features,
    compute_local_optima_network,
)
from pflacco.misc_features import calculate_fitness_distance_correlation
from pflacco.sampling import create_initial_sample
from tqdm import tqdm

from config import cma_features, cmaes_explainer
from iohxplainer import explainer

data_file = "cma_final_processed.pkl"
features = cma_features
df = pd.read_pickle(data_file)

cmaes_explainer.load_results(data_file)
sample_size = 1000  # fixed

new_doe_df = []
new_ela_df = []
new_df_fidonly = []
for dim in cmaes_explainer.dims:
    dim_df = cmaes_explainer.df[cmaes_explainer.df["dim"] == dim].copy()

    X = create_initial_sample(
        dim, lower_bound=-5, upper_bound=5, n=sample_size, seed=42
    )

    for fid in tqdm(cmaes_explainer.fids):
        fid_df = dim_df[dim_df["fid"] == fid]

        for iid in fid_df["iid"].unique():
            iid_df = fid_df[fid_df["iid"] == iid]
            # get best performing conf
            conf, aucs = cmaes_explainer._get_single_best(iid_df)
            conf["dim"] = dim
            conf["fid"] = fid
            conf["iid"] = iid
            conf["auc"] = aucs["auc"].mean()

            # ELA
            func = ioh.get_problem(fid, dimension=dim, instance=iid)
            y = X.apply(lambda x: func(x), axis=1)
            y = (y - np.min(y)) / (np.max(y) - np.min(y))

            # doe = (y.flatten() - np.min(y)) / (
            #    np.max(y) - np.min(y)
            # )
            conf2 = conf.copy()
            conf.update(y)
            new_doe_df.append(conf)

            conf2.update(calculate_ela_meta(X, y))
            conf2.update(calculate_ela_distribution(X, y))
            conf2.update(calculate_ela_level(X, y))
            conf2.update(calculate_nbc(X, y))
            conf2.update(calculate_dispersion(X, y))
            conf2.update(calculate_information_content(X, y, seed=100))

            # all dictionairies! yeaa
            new_ela_df.append(conf2)

# now replace fid, iid with features instead,
# build multiple decision trees .. visualise -- multi-output tree vs single output trees

new_ela_df = pd.DataFrame.from_records(new_ela_df)
new_ela_df.to_pickle("ela-features-cma.pkl")

new_doe_df = pd.DataFrame.from_records(new_doe_df)
new_doe_df.to_pickle("doe-features-cma.pkl")

print(new_ela_df)
print(new_doe_df)

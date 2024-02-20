"""Processing the samples for GSA report
Afterwards run using Docker the following commands: 
 > docker run --rm -v ./de-d5/output:/output -v ./de-d5/:/data ghcr.io/basvanstein/gsareport:main -p /data/problem.json -d /data -o /output
 > docker run --rm -v ./de-d30/output:/output -v ./de-d30/:/data ghcr.io/basvanstein/gsareport:main -p /data/problem.json -d /data -o /output
 > docker run --rm -v ./cma-d5/output:/output -v ./cma-d5/:/data ghcr.io/basvanstein/gsareport:main -p /data/problem.json -d /data -o /output
 > docker run --rm -v ./cma-d30/output:/output -v ./de-d30/:/data ghcr.io/basvanstein/gsareport:main -p /data/problem.json -d /data -o /output

The reports will be in the output directories.
"""

import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
from SALib.sample import latin, saltelli
from SALib.sample.morris import sample
from sklearn.model_selection import cross_val_score

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def generateSamples(sample_size=10000, problem={}):
    if sample_size > 50 or problem < 64:
        x_lhs = latin.sample(problem, sample_size * problem["num_vars"])
    else:
        x_lhs = None
    x_morris = sample(problem, sample_size)
    if problem["num_vars"] < 64:
        x_sobol = saltelli.sample(problem, sample_size)
    else:
        x_sobol = None
    return x_lhs, x_morris, x_sobol


for folder in ["cma-d5", "cma-d30", "de-d5", "de-d30"]:  # "cma-d5", "cma-d30"
    # folder = "d30"

    with open(f"{folder}/problem.json") as json_file:
        problem = json.load(json_file)
    print(problem)

    X = np.loadtxt(f"{folder}/x-space.csv")

    print("X shape", X.shape)
    y = np.loadtxt(f"{folder}/y-space.csv")
    print("y shape", y.shape)

    regr = RandomForestRegressor(n_estimators=20, max_depth=9)
    model_score = cross_val_score(regr, X, y, cv=3)

    # regr = make_pipeline(StandardScaler(), MLPRegressor(random_state=1, max_iter=50))
    # model_score = cross_val_score(regr, X, y, cv=3)
    print(model_score)

    regr.fit(X, y)

    x_lhs, x_morris, x_sobol = generateSamples(10000, problem)
    y_lhs = regr.predict(x_lhs)
    y_morris = regr.predict(x_morris)
    y_sobol = regr.predict(x_sobol)

    np.savetxt(f"{folder}/x_lhs.csv", x_lhs)
    np.savetxt(f"{folder}/x_morris.csv", x_morris)
    np.savetxt(f"{folder}/x_sobol.csv", x_sobol)

    np.savetxt(f"{folder}/y_lhs.csv", y_lhs)
    np.savetxt(f"{folder}/y_morris.csv", y_morris)
    np.savetxt(f"{folder}/y_sobol.csv", y_sobol)

<h1><img src="IOH-xplainer-logo.png" width="256" style="float:left;">IOHxplainer</h1>

<br/><br/><br/>

![Build](https://github.com/nikivanstein/iohxplainer/actions/workflows/test.yml/badge.svg)
[![codecov](https://codecov.io/gh/nikivanstein/iohxplainer/graph/badge.svg?token=SYBOLV6H44)](https://codecov.io/gh/nikivanstein/iohxplainer)
[![PyPI version](https://badge.fury.io/py/iohxplainer.svg)](https://badge.fury.io/py/iohxplainer) 
![Python versions](https://img.shields.io/pypi/pyversions/iohxplainer)
![license](https://img.shields.io/pypi/l/iohxplainer)

eXplainable benchmarking using XAI methods for iterative optimization heuristics.

IOHxplainer builds on top of the `IOH` package, including IOHexperimenter and IOHprofiler.

# Installation

Either install the package from source or via Pypi:

`pip install iohxplainer`

## Quick Start

Import the package

```python
from iohxplainer import explainer
```

Define the `search space` and the algorithm runner, basically the hyper-parameters and ranges you want to explore for a given optimization algorithm or modular algorithm. In this example we use the Differential Evolution implementation by [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html) and we take a limited set of hyper-parameter options to explore.

```python
from scipy.optimize import differential_evolution
from ConfigSpace import ConfigurationSpace
import numpy as np

confSpace = ConfigurationSpace(
    {
        "strategy": ["best1bin", "best1exp", "rand1exp", "randtobest1exp"],
        "popsize": [1, 2, 5, 10],
    }
) 

features = ["strategy", "popsize"]

def run_de(func, config, budget, dim, *args, **kwargs):
    bounds = [(-5,5)] * dim #define the boundaries
    result = differential_evolution(func, bounds, strategy=config.get("strategy"), popsize=config.get("popsize"))

```

Inside the algorithm runner (`run_de`) we define the boundaries of the to be optimized function. Since we will be using the BBOB benchmark suite for this example, we set the boundaries to -5,5 for each dimension of the problem.

Next we create the `explainer` object and run the experiments.

```python
de_explainer = explainer(
    run_de,
    confSpace,
    algname="Differential Evolution",
    dims=[2],  # we test in 2D for this example.
    fids=np.arange(1, 25),  # we use all 24 BBOB functions.
    iids=[1, 2, 3, 4, 5], # we use the first 5 instances for each BBOB function.
    reps=3, # we repeat each run 3 times with different random seeds.
    sampling_method="grid",  # can also be set to random, since we have a few options we can make a full enumeration of the space.
    grid_steps_dict={}, # only needed if we have continuous parameters.
    sample_size=None,  # only used with random sampling method
    budget=10000,  # evaluation budget of an optimization run.
    seed=1, # starting random seed for reproducability.
    verbose=True,
)

#we start running the experiments in paralell and store intermediate results in a csv (to allow a restart when crashing)
de_explainer.run(paralell=True, start_index=0, checkpoint_file="checkpoint.csv")
#store the final results as pkl file.
de_explainer.save_results("de.pkl")
```

Finally we can analyze the run data and make various plots and reports.

```python
import os
os.mkdir("de_report")
de_explainer.to_latex_report(filename="de_report", img_dir="de_report/")
```

Also see the [python notebook file](example.ipynb) for additional demo material.

# Experimental setup

All experiments and setup from the scientific paper can be found in the (experiments)[experiments/] folder.  
The Modular CMA and Modular DE setup are specified in the `config.py` file.

## Reproducing the experiments

Steps to reproduce the experiments from the paper.
Make sure you checkout the ModularCMAES framework from Github and install the cpp version.
See https://github.com/IOHprofiler/ModularCMAES for installation instructions.

The experiments are run on two modular frameworks, Modular DE and Modular CMAES. Both can be easily installed `poetry add modde modcma`.

1. Run all Modular DE or Modular CMA configurations using the *(de|cma_es)_run-configurations.py* file, writes a pkl file as result. 
(This step takes a few days on a supercomputer with 120 cores). The processed results of this step and the step 2 can also be downloaded (since it takes roughly a month to run on a CPU cluster): (cma_final_processed.pkl)[https://www.dropbox.com/scl/fi/ry9b1nnn7681o3b08o073/cma_final_processed.pkl?rlkey=zi9kjjs8t870ldzx9iw87fpm9&dl=0] and (de_final_processed.pkl)[https://www.dropbox.com/scl/fi/f46q2tuhylupm7vgth948/de_final_processed.pkl?rlkey=j87etm66ilvglue0l35mvw8n7&dl=0].
2. Pre-process the pickle files with *(de|cma_es)_process_pkl.py*.
3. Analyse the performance data of all configurations using IOH-Xplainer *(de|cma_es)_analyse.py*.
4. Compare the two frameworks using *compare_de_cma.py*. Writes the result as latex file (compare-new.tex).
5. Perform automated algorithm configuration experiment using *(de|cma_es)_AAC-notebook.ipynb* files.


# Setting up the dev environment

- Checkout this code.
- Make sure `pipx` (https://github.com/pypa/pipx) is installed with python 3.8+
- Install Poetry with pipx `pipx install poetry`
- Install dependencies `poetry install`
- Run tests `poetry run pytest`


# Cite us

TBA

# ioh-xplainer
eXplainable benchmarking using XAI for iterative optimization heuristics

# Experimental setup

The Modular CMA and Modular DE setup are specified in the `config.py` file.

# scripts

Steps to reproduce the experiments from the paper.

1. Run all Modular DE or Modular CMA configurations using the *(de|cma_es)_run-configurations.py* file, writes a pkl file as result.
2. Pre-process the pickle files with *(de|cma_es)_process_pkl.py*.
3. Analyse the performance data of all configurations using IOH-Xplain *(de|cma_es)_analyse.py*.
4. Compare the two frameworks using *compare_de_cma.py*. Writes the result as latex file (compare-new.tex).
5. Perform automated algorithm configuration experiment using *(de|cma_es)_AAC-notebook.ipynb* files.


# Setting up the dev environment

- Checkout this code.
- Make sure `pipx` (https://github.com/pypa/pipx) is installed with python 3.8+
- Install Poetry with pipx `pipx install poetry`
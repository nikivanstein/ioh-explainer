# ioh-xplainer
eXplainable benchmarking using XAI for iterative optimization heuristics


# scripts

Steps to reproduce the experiments from the paper.

1. Run all Modular DE or Modular CMA configurations using the *(de|cma_es)_run-configurations.py* file, writes a pkl file as result.
2. Pre-process the pickle files with *(de|cma_es)_process_pkl.py*.
3. Analyse the performance data of all configurations using IOH-Xplain *(de|cma_es)_analyse.py*.
4. Compare the two frameworks using *compare_de_cma.py*. Writes the result as latex file (compare-new.tex).
5. Perform automated algorithm configuration experiment using *(de|cma_es)_AAC-notebook.ipynb* files.

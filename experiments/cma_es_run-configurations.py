"""Run all CMA configurations in paraalell using the configurationspace"""

from config import cmaes_explainer

cmaes_explainer.run(paralell=True, start_index=0, checkpoint_file="cma-checkpoint.csv")
cmaes_explainer.save_results("cma_final.pkl")

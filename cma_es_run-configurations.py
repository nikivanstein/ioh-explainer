"""Run all CMA configurations in paraalell using the configurationspace"""

from config import cma_explainer

cma_explainer.run(paralell=True, start_index = 0, checkpoint_file="cma-checkpoint.csv")
cma_explainer.save_results("cma_final.pkl")

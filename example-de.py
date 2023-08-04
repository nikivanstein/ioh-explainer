from ConfigSpace import ConfigurationSpace
from ConfigSpace.util import generate_grid
from ioh_xplainer import explainer
from modde import ModularDE, Parameters
import numpy as np


cs = ConfigurationSpace({
    "F": (0.05, 2.0),              # Uniform float
    "CR" : (0.05, 1.0),            # Uniform float
    "lambda_": (1, 20)             # Uniform int
})

steps_dict = {
    "F": 20, 
    "CR" : 20,
    "lambda_": 10
}


def run_de(func, config, budget, dim, *args, **kwargs):
    item = {'F': np.array([float(config.get('F'))]), 'CR':np.array([float(config.get('CR'))]),  'lambda_' : int(config.get('lambda_'))*dim }
    item['budget'] = int(budget)
    c = ModularDE(func, **item)
    try:
        c.run()
        return []
    except Exception as e:
        print(f"Found target {func.state.current_best.y} target, but exception ({e}), so run failed")
        return []

de_explainer = explainer(run_de, 
                 cs , 
                 dims = [5,10,20],#,10,40],#, 10, 20, 40 
                 fids = [1,5,7,13,18,20,23], #,5
                 iids = 5, #20 
                 reps = 5, 
                 sampling_method = "grid",  #or random
                 grid_steps_dict = steps_dict,
                 sample_size = None,  #only used with random method
                 budget = 10000, #10000
                 seed = 1,
                 verbose = False)


de_explainer.run(paralell=True)
de_explainer.save_results("de_results.pkl")


de_explainer.load_results("de_results.pkl")
#x = de_explainer.df[(de_explainer.df['fid'] == 1) & (de_explainer.df['dim'] == 5)][["F","CR","lambda_"]].to_numpy()

#y = de_explainer.df[(de_explainer.df['fid'] == 1) & (de_explainer.df['dim'] == 5)]["auc"].to_numpy()
#np.savetxt("sobol/x.csv", x)
#np.savetxt("sobol/y.csv", y)

de_explainer.plot(save_figs = True, prefix = "img/de_")

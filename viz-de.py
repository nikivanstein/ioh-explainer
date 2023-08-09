from ConfigSpace import ConfigurationSpace
from ConfigSpace.util import generate_grid
from ioh_xplainer import explainer
from modde import ModularDE, Parameters
import numpy as np


cs = ConfigurationSpace({
#    "F": (0., 2.0),              # Uniform float
#    "CR" : (0., 1.0),            # Uniform float
#    "lambda_": ['nan', '1', '10'],    # 1 or 10xdim
    "mutation_base": ['target', 'best', 'rand'], 
    "mutation_reference" : ['pbest', 'rand', 'nan', 'best'], 
    "mutation_n_comps" : [1,2], 
    "use_archive" : [False, True], 
    "crossover" : ['exp', 'bin'], 
    "adaptation_method" : ['nan', 'jDE', 'shade'],
    "lpsr" : [False, True]
})

steps_dict = {
    #"F": 10, 
    #"CR" : 5,
}


def run_de(func, config, budget, dim, *args, **kwargs):

    # lam = config.get('lambda_')
    # if config.get('lambda_') == 'nan':
    #     lam = None
    # else:
    #     lam = int(config.get('lambda_')) * dim
        
    mut = config.get('mutation_reference')
    if config.get('mutation_reference') == 'nan':
        mut = None

    archive = config.get('use_archive')
    if config.get('use_archive') == "False":
        archive = False
    elif config.get('use_archive') == "True":
        archive = True

    lpsr = config.get('lpsr')
    if config.get('lpsr') == "False":
        lpsr = False
    elif config.get('lpsr') == "True":
        lpsr = True
    
    cross = config.get('crossover')
    if config.get('crossover') == 'nan':
        cross = None

    adaptation_method = config.get('adaptation_method')
    if config.get('adaptation_method') == 'nan':
        adaptation_method = None
    
    item = {
        "mutation_base": config.get('mutation_base'), 
        "mutation_reference" : mut, 
        "mutation_n_comps" : int(config.get('mutation_n_comps')), 
        "use_archive" : archive, 
        "crossover" : cross, 
        "adaptation_method_F" : adaptation_method,
        "adaptation_method_CR" : adaptation_method,
        "lpsr" : lpsr
         }
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
                 algname="mod-de",
                 dims = [5,30],#,10,40],#, 10, 20, 40  ,15,30
                 fids = np.arange(1,25), #,5
                 iids = 3, #20 
                 reps = 3, 
                 sampling_method = "grid",  #or random
                 grid_steps_dict = steps_dict,
                 sample_size = None,  #only used with random method
                 budget = 10000, #10000
                 seed = 1,
                 verbose = True)


#de_explainer.run(paralell=True)
#de_explainer.save_results("de_results_huge5d.pkl")


de_explainer.load_results("mod_de_all.pkl")
#x = de_explainer.df[(de_explainer.df['fid'] == 1) & (de_explainer.df['dim'] == 5)][["F","CR","lambda_"]].to_numpy()

#y = de_explainer.df[(de_explainer.df['fid'] == 1) & (de_explainer.df['dim'] == 5)]["auc"].to_numpy()
#np.savetxt("sobol/x.csv", x)
#np.savetxt("sobol/y.csv", y)

de_explainer.explain(file_prefix = "img/img_", keep_order=True)

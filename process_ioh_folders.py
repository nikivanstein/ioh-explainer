import os
import numpy as np
import matplotlib.pyplot as plt
import json

def get_json_files(path):
    items = os.listdir(path)
    json_files = [os.path.join(path, x) for x in items if x.endswith("json")]
    return json_files

def get_run_data(scenario, budget=10000, target=0):
    runs = []
    targets_reached = []
    t = np.unique(np.geomspace(1, budget, 250).astype(int))
    with open(os.path.join(scenario["dirname"], scenario["path"])) as f:
        run = []
        target_reached = None
        next(f)
        for line in f:
            if line.startswith("ev"):
                x, y = zip(*run)
                idx = (np.searchsorted(x, t, side="right") - 1).clip(0)
                y = np.minimum.accumulate(np.array(y)[idx])
                runs.append(y)
                targets_reached.append(target_reached)
                run = []
                target_reached = None
                continue
            ev, y, *a = line.strip().split()
            try:
                ev, y = int(ev), float(y)
            except:
                continue
            if y < target:
                target_reached = target_reached or ev
            run.append((ev, y))
    return t, runs

def geo_mean_overflow(iterable):
    return np.exp(np.log(iterable).mean(axis=0))

def plot_runs(t, runs, label, ax, color=None):
    data = geo_mean_overflow(runs.clip(1e-16))
    for run in runs:
        #ax.loglog(t, data, color="white", linewidth=3)
        if color != None:
            ax.loglog(t, run, linewidth=1, color=color)
        else:
            ax.loglog(t, run, linewidth=1)
    
    if color != None:
        ax.loglog(t, data, linewidth=2, color=color, label=label)
    else:
        ax.loglog(t, data, linewidth=2, label=label)

def get_scenario(path, fid, dimension):
    json_files = get_json_files(path)
    try:
        json_file, *_ = [x for x in json_files if f"{fid}_" in x]
    except Exception as e:
        return None, None

    with open(json_file) as f:
        meta = json.load(f)
    try:
        scenario, *_ = [x for x in meta["scenarios"] if x["dimension"] == dimension]
        scenario["dirname"] = os.path.dirname(json_file)
        return scenario, meta["algorithm"]["name"]
    except Exception as e:
        return None, None

dim = 5,30

for framework in ["de", "cma"]:
    for fid in range(1, 25):
        f, axes = plt.subplots(1, 2, figsize=(20, 10))
        f.suptitle(f"F{fid}")
        for ax, d in zip(axes.ravel(), dim):
            path = f"/data/neocortex/{framework}_data/"
            
            for config in range(0,25):
                config_nr = config
                if (d == 30):
                    config_nr += 25
                for iid in range(1,6):
                    iid_runs = []
                    folder = f"mod-{framework}-{config_nr}-{d}-{fid}-{iid}"
                    if os.path.isdir(os.path.join(path, folder)):
                        algpath = os.path.join(path, folder)
                        scenario, name = get_scenario(algpath, fid, d)
                        if not scenario:
                            continue
                        t, runs = get_run_data(scenario)
                        iid_runs.extend(runs)
                if (config_nr == 0 and d == 5) or (config_nr == 25 and d == 30):
                    plot_runs(t, np.array(iid_runs), "Average-best", ax, "green")
                elif config == fid:
                    plot_runs(t, np.array(iid_runs), "Single-best", ax, "blue")
                #else:
                #    plot_runs(t, np.array(iid_runs), "Other", ax, "gray")
            ax.set_title(f"{d}D")
            ax.grid()
            ax.legend()
            ax.set_xlabel("evals")
            ax.set_xlabel("best so far")
        plt.tight_layout()
        plt.savefig(f"output/convergence_plots/{framework}-{fid}.png")
        plt.clf()
        plt.close()

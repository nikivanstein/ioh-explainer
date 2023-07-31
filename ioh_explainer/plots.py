import pandas as pd
import matplotlib.pyplot as plt
import string as str
import numpy as np

def strength_weakness_plt(df, baseline="RS", algname="IOH"):
    #assuming input is a dataframe created by the Analyse algorithm function of ioh-explainer.
    #should include at least fid 1 and 5.
    #baseline can be RS (random Search) or...

    df = pd.DataFrame(columns=["Instance Robustness", 
                               "Seed Robustness", 
                               "Hyper-parameter Robustness",
                               "Global structure exploitation",
                               "Local optima evasion",
                               "Low-dim performance",
                               "High-dim performance",
                               "Avg performance default",
                               "Performance global best"])
    
    fig, ax = plt.subplots()

# Example data
    y_ticks = df.keys()
    y_pos = np.arange(len(y_ticks))
    stats = [20,30,-20,45,66,88,65, 44, 70]
    error = np.random.rand(len(stats))

    colors = ["#880000" if x < 0 else "#008800" for x in stats]

    ax.barh(y_pos, stats, xerr=error, color=colors, align='center')
    ax.set_xbound([-100,100])
    ax.set_yticks(y_pos, labels=y_ticks)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Relative strengths against baseline (%)')
    ax.set_title('CMA-ES')
    plt.tight_layout()
    plt.savefig("example.png")

strength_weakness_plt(None)
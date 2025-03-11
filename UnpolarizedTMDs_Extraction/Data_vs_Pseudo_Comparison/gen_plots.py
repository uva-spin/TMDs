import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import lhapdf


# Load Real Data
E288 = pd.read_csv("../Data/E288.csv")
E605 = pd.read_csv("../Data/E605.csv")
E772 = pd.read_csv("../Data/E772.csv")
data = pd.concat([E288])
#data = pd.read_csv("pseudodataBQM.csv")

# Load Pseudo Data
pseudoE288 = pd.read_csv("pseudodata_E288.csv")


import matplotlib.pyplot as plt

def gen_plots(df1, df2, filename):
    df1["unique_group"] = df1["QM"].astype(str) + "_" + df1["x1"].astype(str) + "_" + df1["x2"].astype(str)
    df2["unique_group"] = df2["QM"].astype(str) + "_" + df2["x1"].astype(str) + "_" + df2["x2"].astype(str)

    groups_df1 = df1.groupby("unique_group")
    groups_df2 = df2.groupby("unique_group")

    n_groups = groups_df1.ngroups
    ncols = 3
    nrows = (n_groups + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
    axes = axes.flatten()

    for idx, (group_name, group_df1) in enumerate(groups_df1):
        qT1 = group_df1['qT'].values
        A1 = group_df1['A'].values
        A1_err = group_df1['dA'].values
        QM = group_df1['QM'].values[0]

        axes[idx].errorbar(qT1, A1, yerr=A1_err, fmt='o', color='blue', label='df1: $E\\frac{d^3\\sigma}{dp^3}$')

        # If df2 has the same group, plot it too
        if group_name in groups_df2.groups:
            group_df2 = groups_df2.get_group(group_name)
            qT2 = group_df2['qT'].values
            A2 = group_df2['A'].values
            A2_err = group_df2['dA'].values

            axes[idx].errorbar(qT2, A2, yerr=A2_err, fmt='s', color='red', label='df2: $E\\frac{d^3\\sigma}{dp^3}$')

        axes[idx].set_title(f'$Q_M$ = {QM:.2f} GeV')
        axes[idx].set_xlabel('qT')
        axes[idx].set_ylabel('A')
        axes[idx].legend()
        axes[idx].grid(True)

    plt.tight_layout()
    plt.savefig(str(filename) + ".pdf")


testdf = gen_plots(E288,pseudoE288,"E288_Comparison")
testdf

#gen_plots(E288,pseudoE288,"E288_Comparison")
#gen_plots(E605,"E605")
#gen_plots(E772,"E772")


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import lhapdf


# Load and Preprocess Data
E288 = pd.read_csv("E288.csv")
E605 = pd.read_csv("E605.csv")
E772 = pd.read_csv("E772.csv")
data = pd.concat([E772])
#data = pd.read_csv("pseudodataBQM.csv")


x1_values = tf.constant(data['x1'].values, dtype=tf.float32)
x2_values = tf.constant(data['x2'].values, dtype=tf.float32)
qT_values = tf.constant(data['qT'].values, dtype=tf.float32)
QM_values = tf.constant(data['QM'].values, dtype=tf.float32)
A_true_values = tf.constant(data['A'].values, dtype=tf.float32)



def gen_plots(df,filename):
    df["unique_group"] = (
    df["QM"].astype(str) + "_" + df["x1"].astype(str) + "_" + df["x2"].astype(str)
    )
    groups = df.groupby("unique_group")
    n_groups = groups.ngroups
    ncols = 3
    nrows = (n_groups + ncols - 1) // ncols


    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
    axes = axes.flatten()

    for idx, (group_name, group_df) in enumerate(groups):
        qT = group_df['qT'].values
        QM = group_df['QM'].values
        x1 = group_df['x1'].values
        x2 = group_df['x2'].values
        A_true = group_df['A'].values
        A_err = group_df['dA'].values
        
        axes[idx].errorbar(qT, A_true, yerr=A_err, fmt='o', color='blue', label='$E\\frac{d^3\\sigma}{dp^3}$')
        axes[idx].set_title(f'$Q_M$ = {QM[0]:.2f} GeV')
        axes[idx].set_xlabel('qT')
        axes[idx].set_ylabel('A')
        axes[idx].legend()
        axes[idx].grid(True)

    plt.savefig(str(filename)+".pdf")


gen_plots(E288,"E288")
gen_plots(E605,"E605")
gen_plots(E772,"E772")

# data["unique_group"] = (
#     data["QM"].astype(str) + "_" + data["x1"].astype(str) + "_" + data["x2"].astype(str)
# )
# groups = data.groupby("unique_group")
# n_groups = groups.ngroups
# ncols = 3
# nrows = (n_groups + ncols - 1) // ncols


# fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
# axes = axes.flatten()

# for idx, (group_name, group_df) in enumerate(groups):
#     qT = group_df['qT'].values
#     QM = group_df['QM'].values
#     x1 = group_df['x1'].values
#     x2 = group_df['x2'].values
#     A_true = group_df['A'].values
#     A_err = group_df['dA'].values
    
#     axes[idx].errorbar(qT, A_true, yerr=A_err, fmt='o', color='blue', label='A_pred')
#     axes[idx].set_title(f'A vs qT for $Q_M$ = {QM[0]:.2f} GeV')
#     axes[idx].set_xlabel('qT')
#     axes[idx].set_ylabel('A')
#     axes[idx].legend()
#     axes[idx].grid(True)

# plt.tight_layout()
# plt.savefig(f"Data_E772.pdf")

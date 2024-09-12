import tensorflow as tf
import pandas as pd
import numpy as np
#import lhapdf
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.backends.backend_pdf import PdfPages

k_upper = 6
kBins = 100
#phiBins = 10

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")
        

########### Import pseudodata file 
df = pd.read_csv('E288_pseudo_data.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')


####### Here we define a function that can sample cross-section within errA ###
def GenerateReplicaData(df):
    pseudodata_df = {'x1': [],
                     'x2': [],
                     'pT': [],
                     'QM': [],
                     'A': [],
                     'errA':[]}
    #pseudodata_df = pd.DataFrame(pseudodata_df)
    pseudodata_df['x1'] = df['x1']
    pseudodata_df['x2'] = df['x2']
    pseudodata_df['pT'] = df['pT']
    pseudodata_df['QM'] = df['QM']
    pseudodata_df['A'] = df['A']
    tempA = df['A']
    tempAerr = np.abs(np.array(df['errA'])) 
    pseudodata_df['errA'] = np.random.normal(loc=tempA, scale=tempAerr)
    pseudodata_df['fsbar_xA']=df['fsbar_xA']
    pseudodata_df['fubar_xA']=df['fubar_xA']
    pseudodata_df['fdbar_xA']=df['fdbar_xA']
    pseudodata_df['fd_xA']=df['fd_xA']
    pseudodata_df['fu_xA']=df['fu_xA']
    pseudodata_df['fs_xA']=df['fs_xA']
    pseudodata_df['fsbar_xB']=df['fsbar_xB']
    pseudodata_df['fubar_xB']=df['fubar_xB']
    pseudodata_df['fdbar_xB']=df['fdbar_xB']
    pseudodata_df['fd_xB']=df['fd_xB']
    pseudodata_df['fu_xB']=df['fu_xB']
    pseudodata_df['fs_xB']=df['fs_xB']
    return pd.DataFrame(pseudodata_df)


### Let's look at the values of k1 and k2 for a given kinematic set (per line)

def kB(qT, k, phi):
    k = tf.convert_to_tensor(k, dtype=qT.dtype)
    phi = tf.convert_to_tensor(phi, dtype=qT.dtype)
    return tf.sqrt(qT**2 + k**2 - 2*qT*k*tf.cos(phi))



# Consider first 24 pT values
pT_values = df['pT'][:20]


phi_values = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi] 
k_values = np.linspace(0, 6, 24)  


with PdfPages('kB_vs_k_plots_first_24_pT_with_phi.pdf') as pdf:
    fig, axes = plt.subplots(5, 4, figsize=(12, 18))  
    axes = axes.flatten()  
    
    for idx, pT in enumerate(pT_values):
        qT = tf.constant(pT, dtype=tf.float32)
        
        for phi_value in phi_values:
            phi = tf.constant(phi_value, dtype=tf.float32)
            kB_values = [kB(qT, tf.constant(k, dtype=tf.float32), phi).numpy() for k in k_values]
            ax = axes[idx]
            ax.plot(k_values, kB_values, marker='o', label=f'phi = {phi_value:.2f} rad')
        
        ax.set_title(f'pT = {pT:.2f}')
        ax.set_xlabel('k')
        ax.set_ylabel('kB')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig) 
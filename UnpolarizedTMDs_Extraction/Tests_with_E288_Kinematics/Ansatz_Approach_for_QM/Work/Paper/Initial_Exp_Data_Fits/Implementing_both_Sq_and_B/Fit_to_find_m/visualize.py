import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# Define the model function S(q_T) = 1/(2π m) * exp(-q_T^2/(2m))
def model_function(qT, m):
    return 1/(2*np.pi*m) * np.exp(-qT**2/(2*m))


functions_results = pd.read_csv("comparison_results.csv")

    
qT = np.array(functions_results['qT'])
SqTmean = np.array(functions_results['SqT_mean'])
SqTstd = np.array(functions_results['SqT_std'])

SqT_model = np.array(model_function(qT,0.38))


# ####### Plot SqT ############
plt.figure(1,figsize=(10, 6))
plt.plot(qT, SqT_model, label='$\mathcal{S}(q_T)$ DNN model (mean)', linestyle='-', color='blue')
plt.plot(qT, SqTmean, label='$\mathcal{S}(q_T)$ DNN model (mean)', linestyle='-', color='red')
plt.fill_between(qT, SqTmean - SqTstd, SqTmean + SqTstd, color='red', alpha=0.2, label='$\mathcal{S}(q_T)$ DNN model (std)')
plt.xlabel(r'$q_T$', fontsize=14)
plt.ylabel(r'$\mathcal{S}(q_T)$', fontsize=14)
plt.title('$\mathcal{S}(q_T)$ vs $q_T$', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig('SqT_plot.pdf')
plt.close()

import pandas as pd
import numpy as np
from iminuit import Minuit
import matplotlib.pyplot as plt

def model_function(qT, m):
    return 1/(2*np.pi*m) * np.exp(-qT**2/(2*m))


functions_results = pd.read_csv("comparison_results.csv")


qT = np.array(functions_results['qT'])
SqTmean = np.array(functions_results['SqT_mean'])
SqTstd = np.array(functions_results['SqT_std'])


def chi2(m):
    yth = model_function(qT,m)
    yexp = SqTmean
    yerr = SqTstd
    tempChi2=np.sum(((yth-yexp))**2)
    return tempChi2

m_initial_guess = 0.4  
minuit = Minuit(chi2, m=m_initial_guess)


minuit.limits["m"] = (0, 1.0)  # m should be positive


minuit.migrad()
minuit.hesse()

m_solution = minuit.values["m"]
print(minuit.values["m"])


ms = np.linspace(0.1, 3.0, 100)
chi2_vals = [chi2(m) for m in ms]
plt.plot(ms, chi2_vals)
plt.xlabel("m")
plt.ylabel("Chi2")
plt.title("Chi2 vs m")
plt.grid()
plt.show()


SqT_model = np.array(model_function(qT,m_solution))


# ####### Plot SqT ############
plt.figure(1,figsize=(10, 6))
plt.plot(qT, SqT_model, label='$m$ from MINUIT Fit', linestyle='-', color='blue')
#plt.text(0.1,4.0,f'm={m_solution}')
plt.plot(qT, SqTmean, label='$\mathcal{S}(q_T)$ DNN model (mean)', linestyle='-', color='red')
plt.fill_between(qT, SqTmean - SqTstd, SqTmean + SqTstd, color='red', alpha=0.2, label='$\mathcal{S}(q_T)$ DNN model (std)')
plt.xlabel(r'$q_T$', fontsize=14)
plt.ylabel(r'$\mathcal{S}(q_T)$', fontsize=14)
plt.title(f'm={m_solution}', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig('Fit_Results_for_Sqt.pdf')

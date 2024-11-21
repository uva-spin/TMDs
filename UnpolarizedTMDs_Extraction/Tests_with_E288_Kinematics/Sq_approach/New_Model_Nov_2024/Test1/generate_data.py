import numpy as np
from scipy.integrate import simpson
import pandas as pd

# Define S1 and S2
def S1(k):
    return np.exp(-4 * k**2 / (4 * k**2 + 4))

def S2(k):
    return np.exp(-4 * k**2 / (4 * k**2 + 1))

# Integration limits
k_min, k_max = 0, 2
pT_values = np.linspace(0, 10, 50)  # 50 pT values between 0 and 10
A_values = []

for pT in pT_values:
    k = np.linspace(k_min, k_max, 100)  # Discretize k
    integrand = S1(k) * S2(pT - k) + S1(pT - k) * S2(k)
    A_values.append(simpson(integrand, k))

# Save to CSV
data = pd.DataFrame({'pT': pT_values, 'A': A_values})
data.to_csv('A_pT_data.csv', index=False)
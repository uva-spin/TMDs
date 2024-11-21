import numpy as np
from scipy.integrate import simpson
import pandas as pd

# Define the true S1(k, QM) and S2(k, QM) functions
def S1_true(k, QM):
    Q0 = 100
    return np.exp(-4 * k**2 / (4 * k**2 + 4)) * np.log(Q0 / QM)

def S2_true(k, QM):
    Q0 = 100
    return np.exp(-4 * k**2 / (4 * k**2 + 1)) * np.log(Q0 / QM)

# Define A(qT, QM) function
def compute_A(qT, QM):
    k = np.linspace(0, 2, 100)
    phi = np.linspace(0, 2 * np.pi, 100)

    integrand = np.zeros((len(k), len(phi)))

    for i, k_val in enumerate(k):
        for j, phi_val in enumerate(phi):
            sqrt_term = np.sqrt(qT**2 + k_val**2 - 2 * qT * k_val * np.cos(phi_val))
            term1 = S1_true(k_val, QM) * S2_true(sqrt_term, QM)
            term2 = S1_true(sqrt_term, QM) * S2_true(k_val, QM)
            integrand[i, j] = term1 + term2

    # Integrate over phi and then k
    phi_integral = simpson(integrand, phi, axis=1)
    total_integral = simpson(phi_integral, k)
    return total_integral

# Generate pseudo-data
qT_values = np.linspace(0.1, 2, 50)
QM_values = np.linspace(10, 100, 50)
A_values = np.array([compute_A(qT, QM) for qT, QM in zip(qT_values, QM_values)])

# Save pseudo-data to CSV
data = pd.DataFrame({'qT': qT_values, 'QM': QM_values, 'A': A_values})
data.to_csv("A_qT_QM_data.csv", index=False)
print("Pseudo-data saved to A_qT_QM_data.csv")

import numpy as np
import pandas as pd
from scipy.integrate import simpson

# Define the true S1(k) and S2(k) functions
def S1_true(k):
    return np.exp(-4 * k**2 / (4 * k**2 + 4))

def S2_true(k):
    return np.exp(-4 * k**2 / (4 * k**2 + 1))

# Define the function to compute A(pT)
def A_qT(qT):
    # Parameters for integration
    k_values = np.linspace(0, 2, 100)      # Discretized k for integration
    phi_values = np.linspace(0, 2 * np.pi, 100)  # Discretized phi for integration
    dphi = phi_values[1] - phi_values[0]
    dk = k_values[1] - k_values[0]

    A_values = []
    for qT in qT_values:
        integrand_values = np.zeros((len(k_values), len(phi_values)))

        for i, k in enumerate(k_values):
            for j, phi in enumerate(phi_values):
                term1 = S1_true(k) * S2_true(np.sqrt(qT**2 + k**2 - 2 * qT * k * np.cos(phi)))
                term2 = S1_true(np.sqrt(qT**2 + k**2 - 2 * qT * k * np.cos(phi))) * S2_true(k)
                integrand_values[i, j] = term1 + term2

        # Integrate over phi first (axis=1), then over k (axis=0)
        phi_integral = simpson(integrand_values, phi_values, axis=1)
        total_integral = simpson(phi_integral, k_values)
        A_values.append(total_integral)

    return np.array(A_values)


# Generate A(pT) values
qT_values = np.linspace(0.1, 2.0, 50)  # Values of qT (50 points)
A_values = A_qT(qT_values)

# Save data to a CSV file
data = pd.DataFrame({'qT': qT_values, 'A': A_values})
data.to_csv("A_qT_data.csv", index=False)
print("Data generation completed and saved as 'A_qT_data.csv'.")

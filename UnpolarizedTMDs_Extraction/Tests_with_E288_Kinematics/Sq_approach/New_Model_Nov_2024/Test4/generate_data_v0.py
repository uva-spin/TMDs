import numpy as np
import pandas as pd
from scipy.integrate import simpson

# Define the true S1(k, QM) and S2(k, QM) functions
def S1_true(k, QM):
    Q0 = 100
    return np.exp(-4 * k**2 / (4 * k**2 + 4)) * np.log(Q0 / QM)

def S2_true(k, QM):
    Q0 = 100
    return np.exp(-4 * k**2 / (4 * k**2 + 1)) * np.log(Q0 / QM)

# Define the function to compute A(qT, QM)
def A_qT_QM(qT_values,QM_values):
    k_values = np.linspace(0, 2, 100)     # 100 points for k
    phi_values = np.linspace(0, 2 * np.pi, 100)  # 100 points for phi
    A_values = []

    for QM in QM_values:
        for qT in qT_values:
            integrand_values = np.zeros((len(k_values), len(phi_values)))

            for i, k in enumerate(k_values):
                for j, phi in enumerate(phi_values):
                    sqrt_term = np.sqrt(qT**2 + k**2 - 2 * qT * k * np.cos(phi))
                    term1 = S1_true(k, QM) * S2_true(sqrt_term, QM)
                    term2 = S1_true(sqrt_term, QM) * S2_true(k, QM)
                    integrand_values[i, j] = term1 + term2

            # Integrate over phi first, then over k
            phi_integral = simpson(integrand_values, phi_values, axis=1)
            total_integral = simpson(phi_integral, k_values)
            A_values.append([qT, QM, total_integral])

    return np.array(A_values)

# Parameters for integration
qT_values = np.linspace(0.1, 2.0, 50)  # 50 points for qT
QM_values = np.linspace(10, 100, 10)  # 10 points for QM

# Generate A(qT, QM) values
data = A_qT_QM(qT_values,QM_values)

# Save to CSV
df = pd.DataFrame(data, columns=["qT", "QM", "A"])
df.to_csv("A_qT_QM_data.csv", index=False)
print("Data generation completed and saved as 'A_qT_QM_data.csv'.")

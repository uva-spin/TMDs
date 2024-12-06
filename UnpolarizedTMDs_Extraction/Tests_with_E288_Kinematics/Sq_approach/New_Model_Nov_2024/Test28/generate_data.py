import numpy as np
from scipy.integrate import simpson
import pandas as pd
import lhapdf

# Load LHAPDF dataset
NNPDF4_nlo = lhapdf.mkPDF('NNPDF40_nlo_as_01180')

# Helper function for PDF values
def pdf(pdfset, flavor, x, QQ):
    return pdfset.xfxQ2(flavor, x, QQ)

# Define the true S1(k, QM) and S2(k, QM) functions
def S1_true(k, QM):
    Q0 = 100
    return np.exp(-4 * k**2 / (4 * k**2 + 4)) * np.log(Q0 / QM)

def S2_true(k, QM):
    Q0 = 100
    return np.exp(-4 * k**2 / (4 * k**2 + 1)) * np.log(Q0 / QM)

# Define A(x1, x2, pT, QM) function
def compute_A(x1, x2, pT, QM):
    k = np.linspace(0.0001, 10, 100)  # Discretize k
    phi = np.linspace(0, 2 * np.pi, 100)  # Discretize phi

    # Integrand values
    integrand = np.zeros((len(k), len(phi)))

    # Constants
    Q_scale = 2.4  # Energy scale for PDFs

    for i, k_val in enumerate(k):
        for j, phi_val in enumerate(phi):
            # Calculate sqrt(pT^2 + k^2 - 2 * pT * k * cos(phi))
            sqrt_term = np.sqrt(pT**2 + k_val**2 - 2 * pT * k_val * np.cos(phi_val))

            # Ensure x1 and x2 are valid for PDFs
            if x1 < 1e-7 or x1 > 1 or x2 < 1e-7 or x2 > 1:
                continue

            # Compute PDFs for each flavor
            f_u_x1 = pdf(NNPDF4_nlo, 2, x1, Q_scale)  # u-quark
            f_ubar_x2 = pdf(NNPDF4_nlo, -2, x2, Q_scale)  # anti-u quark
            f_u_x2 = pdf(NNPDF4_nlo, 2, x2, Q_scale)
            f_ubar_x1 = pdf(NNPDF4_nlo, -2, x1, Q_scale)

            # Compute S1 and S2
            S1_k_QM = S1_true(k_val, QM)
            S2_sqrt_QM = S2_true(sqrt_term, QM)
            S1_sqrt_QM = S1_true(sqrt_term, QM)
            S2_k_QM = S2_true(k_val, QM)

            # Compute the integrand
            term1 = f_u_x1 * f_ubar_x2 * S1_k_QM * S2_sqrt_QM
            term2 = f_u_x2 * f_ubar_x1 * S1_sqrt_QM * S2_k_QM
            integrand[i, j] = term1 + term2

    # Integrate over phi and then k
    phi_integral = simpson(integrand, phi, axis=1)
    total_integral = simpson(phi_integral, k)
    return total_integral

# Load data from E288.csv
data = pd.read_csv("E288.csv")

# Extract values from the CSV
x1_values = data['x1'].values
x2_values = data['x2'].values
pT_values = data['pT'].values
QM_values = data['QM'].values

# Compute A for each row
A_values = np.array([
    compute_A(x1, x2, pT, QM)
    for x1, x2, pT, QM in zip(x1_values, x2_values, pT_values, QM_values)
])

# Create a new DataFrame for the results
results_df = pd.DataFrame({
    'x1': x1_values,
    'x2': x2_values,
    'pT': pT_values,
    'QM': QM_values,
    'A': A_values
})

# Save results to a new CSV file
results_df.to_csv("A_for_E288kinematics.csv", index=False)
print("Computed A values saved to A_for_E288kinematics.csv")

import numpy as np
import matplotlib.pyplot as plt
import lhapdf
import pandas as pd


NNPDF4_nlo = lhapdf.mkPDF('NNPDF40_nlo_as_01180')
data = pd.read_csv("../../E288.csv")
alpha = 1/137

mm = 0.5

def pdf(pdfset, flavor, x, QQ):
    return pdfset.xfxQ(flavor, x, QQ)

def S(k):
    return ((k**2)/(mm*np.pi))*np.exp(-(k**2)/mm)

def fDNNQ(QM, b=0.5):
    return np.exp(-b * QM)

def compute_A(x1, x2, qT, QM):
    f_u_x1 = pdf(NNPDF4_nlo, 2, x1, QM) 
    f_ubar_x2 = pdf(NNPDF4_nlo, -2, x2, QM)
    f_u_x2 = pdf(NNPDF4_nlo, 2, x2, QM)
    f_ubar_x1 = pdf(NNPDF4_nlo, -2, x1, QM)
    f_d_x1 = pdf(NNPDF4_nlo, 1, x1, QM) 
    f_dbar_x2 = pdf(NNPDF4_nlo, -1, x2, QM)
    f_d_x2 = pdf(NNPDF4_nlo, 1, x2, QM)
    f_dbar_x1 = pdf(NNPDF4_nlo, -1, x1, QM)
    f_s_x1 = pdf(NNPDF4_nlo, 3, x1, QM) 
    f_sbar_x2 = pdf(NNPDF4_nlo, -3, x2, QM)
    f_s_x2 = pdf(NNPDF4_nlo, 3, x2, QM)
    f_sbar_x1 = pdf(NNPDF4_nlo, -3, x1, QM)

    # # Sk_contribution = (1/2)*(np.pi)*(np.exp(-qT*qT/2))
    Sk_contribution = (8*mm*mm + qT*qT*qT*qT)/(32*np.pi*mm)*(np.exp(-(qT*qT)/(2*mm)))

    fDNN_contribution = fDNNQ(QM)

    ux1ubarx2_term = f_u_x1*f_ubar_x2
    ubarx1ux2_term = f_u_x2*f_ubar_x1
    dx1dbarx2_term = f_d_x1*f_dbar_x2
    dbarx1dx2_term = f_d_x2*f_dbar_x1
    sx1sbarx2_term = f_s_x1*f_sbar_x2
    sbarx1sx2_term = f_s_x2*f_sbar_x1
    #FUU = (ux1ubarx2_term + ubarx1ux2_term + dx1dbarx2_term + dbarx1dx2_term + sx1sbarx2_term + sbarx1sx2_term) * fDNN_contribution * Sk_contribution
    #cross_section =  FUU*qT*((4*np.pi*alpha)**2)/(9*QM*QM*QM)/(2*np.pi*qT)
    PDFs = (ux1ubarx2_term + ubarx1ux2_term + dx1dbarx2_term + dbarx1dx2_term + sx1sbarx2_term + sbarx1sx2_term)
    fDNN = fDNN_contribution
    # factor = qT*((4*np.pi*alpha)**2)/(9*QM*QM*QM)/(2*np.pi*qT)
    factor = ((4*np.pi*alpha)**2)/(9*QM*QM*QM)/(2*np.pi)
    cross_section =  fDNN * factor * PDFs * Sk_contribution
    return cross_section


x1_values = data['xA'].values
x2_values = data['xB'].values
qT_values = data['PT'].values
QM_values = data['QM'].values


A_values = np.array([
    compute_A(x1, x2, qT, QM)
    for x1, x2, qT, QM in zip(x1_values, x2_values, qT_values, QM_values)
])

results_df = pd.DataFrame({
    'x1': x1_values,
    'x2': x2_values,
    'qT': qT_values,
    'QM': QM_values,
    'A': A_values
})

results_df.to_csv("pseudodataE288_BQM_B2.csv", index=False)
print("Computed A values saved to A_for_E288kinematics.csv")
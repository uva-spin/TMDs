import numpy as np
import matplotlib.pyplot as plt
import lhapdf
import pandas as pd


NNPDF4_nlo = lhapdf.mkPDF('NNPDF40_nlo_as_01180')
E288 = pd.read_csv("../Data/E288.csv")
E605 = pd.read_csv("../Data/E605.csv")
E772 = pd.read_csv("../Data/E772.csv")
data = pd.concat([E772])
alpha = 1/137

mm = 0.5

def pdf(pdfset, flavor, x, QQ):
    return pdfset.xfxQ(flavor, x, QQ)

def S(k):
    return ((k**2)/(mm*np.pi))*np.exp(-(k**2)/mm)

# def fDNNQ(QM, b=0.5):
#     return np.exp(-b * QM)

def fDNNQ(QM, b=0.5):
    return np.exp(-b * QM)

# def fDNNQ(QM, b=0.5):
#     return ((QM**2)/(b*np.pi))*np.exp(-(QM**2)/b)

# def fDNNQ(QM, b=0.5):
#     return b * np.exp(-b * QM)

# def fDNNQ(QM, A=1, mu=5.0, sigma=0.001):
#     return A * np.exp(-((QM - mu) ** 2) / (2 * sigma ** 2))

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
    hc_factor = 3.89*10**8 * 1000
    cross_section = cross_section * hc_factor 
    return cross_section


x1_values = data['x1'].values
x2_values = data['x2'].values
qT_values = data['qT'].values
QM_values = data['QM'].values
dA_values = data['dA'].values


A_values = np.array([
    compute_A(x1, x2, qT, QM)
    for x1, x2, qT, QM in zip(x1_values, x2_values, qT_values, QM_values)
])

results_df = pd.DataFrame({
    'x1': x1_values,
    'x2': x2_values,
    'qT': qT_values,
    'QM': QM_values,
    'A': A_values,
    'dA': dA_values
})


def gen_plots(df1, df2, filename):
    df1["unique_group"] = df1["QM"].astype(str) + "_" + df1["x1"].astype(str) + "_" + df1["x2"].astype(str)
    df2["unique_group"] = df2["QM"].astype(str) + "_" + df2["x1"].astype(str) + "_" + df2["x2"].astype(str)

    groups_df1 = df1.groupby("unique_group")
    groups_df2 = df2.groupby("unique_group")

    n_groups = groups_df1.ngroups
    ncols = 3
    nrows = (n_groups + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
    axes = axes.flatten()

    for idx, (group_name, group_df1) in enumerate(groups_df1):
        qT1 = group_df1['qT'].values
        A1 = group_df1['A'].values
        A1_err = group_df1['dA'].values
        QM = group_df1['QM'].values[0]

        axes[idx].errorbar(qT1, A1, yerr=A1_err, fmt='o', color='blue', label='Experiment')


        group_df2 = groups_df2.get_group(group_name)
        qT2 = group_df2['qT'].values
        A2 = group_df2['A'].values
        A2_err = group_df2['dA'].values

        axes[idx].errorbar(qT2, A2, yerr=A2_err, fmt='s', color='red', label='Pseudo Data')

        axes[idx].set_title(f'$Q_M$ = {QM:.2f} GeV')
        axes[idx].set_xlabel('qT')
        axes[idx].set_ylabel('A')
        axes[idx].legend()
        axes[idx].grid(True)

    plt.tight_layout()
    plt.savefig(str(filename) + ".pdf")
    plt.show()

results_df.to_csv("pseudodata_E772.csv", index=False)
print("Computed A values saved csv file")

gen_plots(data,results_df,"E772_Comparison")


######### B(QM) #################

# Generate QM Range for Comparison
QM_values = np.linspace(data['QM'].min(), data['QM'].max(), 200)
fDNNQ_values = fDNNQ(QM_values)

# Plot Analytical vs. Model Predictions
plt.figure(figsize=(10, 6))
plt.plot(QM_values, fDNNQ_values, label=r'Analytical $\mathcal{B}(Q_M)$', linestyle='--', color='blue')
plt.xlabel(r'$Q_M$', fontsize=14)
plt.ylabel(r'$f_{DNNQ}(Q_M)$', fontsize=14)
plt.title('Comparison of Analytical $\mathcal{B}(Q_M)$ and DNNQ Model', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig("QM_comparison_plot.pdf")
plt.show()
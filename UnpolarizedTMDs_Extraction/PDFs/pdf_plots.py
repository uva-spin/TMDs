import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


NNPDF4_lo = pd.read_csv('./PDFsets/NNPDF4_lo.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
NNPDF4_nlo = pd.read_csv('./PDFsets/NNPDF4_nlo.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
NNPDF4_nnlo = pd.read_csv('./PDFsets/NNPDF4_nnlo.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
CTEQ61_nlo = pd.read_csv('./PDFsets/CTEQ61_nlo.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')


fontsiz = 12


def pdf_comp_data(file1,flavor):
    if(flavor==2):
        tempPDF1 = np.array(file1['fu'])
    if(flavor==1):
        tempPDF1 = np.array(file1['fd'])
    if(flavor==3):
        tempPDF1 = np.array(file1['fs'])
    if(flavor==-2):
        tempPDF1 = np.array(file1['fubar'])
    if(flavor==-1):
        tempPDF1 = np.array(file1['fdbar'])
    if(flavor==-3):
        tempPDF1 = np.array(file1['fsbar'])
    return tempPDF1
    
def ff_plots(file1,file2,file3,file4,flavor,title):
    tempx = np.array(file1['x'])
    tempy1 = pdf_comp_data(file1,flavor)
    tempy2 = pdf_comp_data(file2,flavor)
    tempy3 = pdf_comp_data(file3,flavor)
    tempy4 = pdf_comp_data(file4,flavor)
    plt.plot(tempx, tempy1, 'b', label='NNPDF4_lo')
    plt.plot(tempx, tempy2, 'r', label='NNPDF4_nlo')
    plt.plot(tempx, tempy3, 'g', label='NNPDF4_nnlo')
    plt.plot(tempx, tempy4, 'orange', label='CTEQ61_nlo')
    plt.legend(loc=1,fontsize=fontsiz,handlelength=3)
    plt.xscale('log')
    plt.ylim(-1.0,2.0)
    plt.title(str(title))
    

fig1=plt.figure(1,figsize=(20,10))
plt.subplot(2,3,1)
ff_plots(NNPDF4_lo,NNPDF4_nlo,NNPDF4_nnlo,CTEQ61_nlo,2,'u-quark')
plt.subplot(2,3,2)
ff_plots(NNPDF4_lo,NNPDF4_nlo,NNPDF4_nnlo,CTEQ61_nlo,1,'d-quark')
plt.subplot(2,3,3)
ff_plots(NNPDF4_lo,NNPDF4_nlo,NNPDF4_nnlo,CTEQ61_nlo,3,'s-quark')
plt.subplot(2,3,4)
ff_plots(NNPDF4_lo,NNPDF4_nlo,NNPDF4_nnlo,CTEQ61_nlo,-2,'ubar-quark')
plt.subplot(2,3,5)
ff_plots(NNPDF4_lo,NNPDF4_nlo,NNPDF4_nnlo,CTEQ61_nlo,-1,'dbar-quark')
plt.subplot(2,3,6)
ff_plots(NNPDF4_lo,NNPDF4_nlo,NNPDF4_nnlo,CTEQ61_nlo,-3,'s-quark')
plt.savefig('./PDF_Plots/PDF_comparison.pdf', format='pdf', bbox_inches='tight')

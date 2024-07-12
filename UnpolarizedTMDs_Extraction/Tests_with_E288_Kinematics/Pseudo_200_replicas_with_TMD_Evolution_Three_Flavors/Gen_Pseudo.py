##########################################################################
###############  Written by Ishara Fernando                  #############
##############  Revised Date: 05/29/2024    ##############################
##### This code has the following featuers  ##############################
## 1) Flavor-dependend Sq mutliplied by the corresponding pdf from lhapdf#
## 2) SU(3)                                                      #########
## 3) Generates cross-section data for DY                        #########
##########################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lhapdf
import os
from scipy.integrate import simps


#NNPDF4_lo=lhapdf.mkPDF('NNPDF40_lo_as_01180')
NNPDF4_nlo=lhapdf.mkPDF('NNPDF40_nlo_as_01180')
#NNPDF4_nnlo=lhapdf.mkPDF('NNPDF40_nnlo_as_01180')
#CTQ61_nlo=lhapdf.mkPDF('cteq61')

def xFxQ2(dataset,flavor,x,QQ):
    temp_parton_dist_x=[(dataset.xfxQ2(flavor, x[i], QQ)) for i in range(len(x))]
    return temp_parton_dist_x
    
def pdf(pdfset, flavor, x, QQ):
    return np.array(pdfset.xfxQ2(flavor, x, QQ))
    
def PDFs(pdfset,flavor,tempX):
    tempPDFs=[pdf(pdfset,flavor,tempX[i],2.4) for i in range(0,len(tempX))]
    return tempPDFs


def DY_xFxQ2(dataset,flavor,x,QQ):
    # temp_parton_dist_x=np.array(dataset.xfxQ2(flavor, x, QQ),dtype=object)
    temp_parton_dist_x=np.array(dataset.xfxQ(flavor, x, QQ),dtype=object)
    return temp_parton_dist_x


E288_df = pd.read_csv('../E288.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')

#print(E288_df)


def generate_DY_PDFs(PDFset,df):
    tempQM=df['QM']
    ####### Generating grids for xA ###################
    tempx1=df['xA'] 
    df['fsbar_xA']=DY_xFxQ2(PDFset,-3,tempx1,tempQM)
    df['fubar_xA']=DY_xFxQ2(PDFset,-2,tempx1,tempQM)
    df['fdbar_xA']=DY_xFxQ2(PDFset,-1,tempx1,tempQM)
    df['fd_xA']=DY_xFxQ2(PDFset,1,tempx1,tempQM) 
    df['fu_xA']=DY_xFxQ2(PDFset,2,tempx1,tempQM)
    df['fs_xA']=DY_xFxQ2(PDFset,3,tempx1,tempQM)
    ####### Generating grids for x2 ###################
    tempx2=df['xB']
    df['fsbar_xB']=DY_xFxQ2(PDFset,-3,tempx2,tempQM)
    df['fubar_xB']=DY_xFxQ2(PDFset,-2,tempx2,tempQM)
    df['fdbar_xB']=DY_xFxQ2(PDFset,-1,tempx2,tempQM)
    df['fd_xB']=DY_xFxQ2(PDFset,1,tempx2,tempQM)
    df['fu_xB']=DY_xFxQ2(PDFset,2,tempx2,tempQM)
    df['fs_xB']=DY_xFxQ2(PDFset,3,tempx2,tempQM)
    return df

pdfs_df = generate_DY_PDFs(NNPDF4_nlo,E288_df)
#print(pdfs_df)



#### S(k) for pseudo-data ####

def Sku(k,Q):
    return np.exp(-4*k**2/(4*k**2 + 4))*(1/10)*np.log(100/Q)

def Skubar(k,Q):
    return np.exp(-4*k**2/(4*k**2 + 1))*(1/10)*np.log(100/Q)

def Skd(k,Q):
    return np.exp(-4*k**2/(4*k**2 + 6))*(1/10)*np.log(100/Q)

def Skdbar(k,Q):
    return np.exp(-4*k**2/(4*k**2 + 2))*(1/10)*np.log(100/Q)

def Sks(k,Q):
    return np.exp(-4*k**2/(4*k**2 + 3))*(1/10)*np.log(100/Q)

def Sksbar(k,Q):
    return np.exp(-4*k**2/(4*k**2 + 0.5))*(1/10)*np.log(100/Q)



fu_xA = np.array(pdfs_df['fu_xA'])
fubar_xA = np.array(pdfs_df['fubar_xA'])
fu_xB = np.array(pdfs_df['fu_xB'])
fubar_xB = np.array(pdfs_df['fubar_xB'])

fd_xA = np.array(pdfs_df['fd_xA'])
fdbar_xA = np.array(pdfs_df['fdbar_xA'])
fd_xB = np.array(pdfs_df['fd_xB'])
fdbar_xB = np.array(pdfs_df['fdbar_xB'])

fs_xA = np.array(pdfs_df['fs_xA'])
fsbar_xA = np.array(pdfs_df['fsbar_xA'])
fs_xB = np.array(pdfs_df['fs_xB'])
fsbar_xB = np.array(pdfs_df['fsbar_xB'])

xAvals = np.array(pdfs_df['xA'])
xBvals = np.array(pdfs_df['xB'])
pTvals = np.array(pdfs_df['PT'])
QMvals = np.array(pdfs_df['QM'])
Kvals = np.linspace(0.1,2,len(xAvals))
pT_k_vals = pTvals - Kvals

#print(pT_k_vals)

def f_map(x,f_array):
    mapping = dict(zip(x,f_array))
    return mapping

def fq_val(x,xarray,fq):
    map = f_map(xarray,fq)
    return map.get(x,None)

def fqbar_val(x,xarray,fq):
    map = f_map(xarray,fq)
    return map.get(x,None)

def fx1kx2k(xA,xB,pT,k,QM):
    uubar_contribution = fq_val(xA,xAvals,fu_xA)*Sku(k,QM)*fqbar_val(xB,xBvals,fubar_xB)*Skubar(pT-k,QM) + fqbar_val(xA,xAvals,fubar_xA)*Skubar(k,QM)*fq_val(xB,xBvals,fu_xB)*Sku(pT-k,QM)
    ddbar_contribution = fq_val(xA,xAvals,fd_xA)*Skd(k,QM)*fqbar_val(xB,xBvals,fdbar_xB)*Skdbar(pT-k,QM) + fqbar_val(xA,xAvals,fdbar_xA)*Skdbar(k,QM)*fq_val(xB,xBvals,fd_xB)*Skd(pT-k,QM)
    ssbar_contribution = fq_val(xA,xAvals,fu_xA)*Sks(k,QM)*fqbar_val(xB,xBvals,fubar_xB)*Sksbar(pT-k,QM) + fqbar_val(xA,xAvals,fubar_xA)*Sksbar(k,QM)*fq_val(xB,xBvals,fu_xB)*Sks(pT-k,QM)
    return uubar_contribution + ddbar_contribution + ssbar_contribution

#print(fq_val(x1vals[1]))

def Apseudo(x1,x2,pT,QM):
    tempx1, tempx2, temppT, tempQM, tempA = [], [], [], [], []
    kk = np.linspace(0.0,2.0,len(xAvals))
    for i in range(len(x1)):
        tempx1.append(x1[i])
        tempx2.append(x2[i])
        temppT.append(pT[i])
        tempQM.append(QM[i])
        tempfx1kfx2k = simps(fx1kx2k(x1[i],x2[i],pT[i],kk,QM[i]), dx=(kk[1]-kk[0]))
        tempA.append(tempfx1kfx2k)
    return np.array(tempx1), np.array(tempx2), np.array(temppT), np.array(tempQM), np.array(tempA)

x1Vals, x2Vals, pTVals, QMVals, Avals = Apseudo(xAvals,xBvals,pTvals,QMvals)

df = pd.DataFrame({'x1': x1Vals, 'x2': x2Vals, 'pT': pTVals, 'QM': QMVals, 'A': Avals})
df['frac_error'] = E288_df['frac_error']
df['errA'] = df['A']*E288_df['frac_error']
df['fsbar_xA']=pdfs_df['fsbar_xA']
df['fubar_xA']=pdfs_df['fubar_xA']
df['fdbar_xA']=pdfs_df['fdbar_xA']
df['fd_xA']=pdfs_df['fd_xA']
df['fu_xA']=pdfs_df['fu_xA']
df['fs_xA']=pdfs_df['fs_xA']
#######################
df['fsbar_xB']=pdfs_df['fsbar_xB']
df['fubar_xB']=pdfs_df['fubar_xB']
df['fdbar_xB']=pdfs_df['fdbar_xB']
df['fd_xB']=pdfs_df['fd_xB']
df['fu_xB']=pdfs_df['fu_xB']
df['fs_xB']=pdfs_df['fs_xB']
df.to_csv('E288_pseudo_data.csv')



# def generate_DY_PDFs_for_evaluation(PDFset,df):
#     tempdf = pd.DataFrame()
#     tempQM = df['QM']
#     distinct_QM = np.array(sorted(tempQM.unique()))
#     tempxA = df['xA']
#     minxA = np.min(tempxA)
#     maxxA = np.max(tempxA)
#     xAarray = np.linspace(minxA, maxxA, 100)
#     tempxB = df['xB']
#     minxB = np.min(tempxB)
#     maxxB = np.max(tempxB)
#     xBarray = np.linspace(minxB, maxxB, 100)
#     tempQM_array = np.linspace(distinct_QM[1],distinct_QM[1],100)
#     ####### Generating grids for xA ###################
#     tempdf['fsbar_xA']=DY_xFxQ2(PDFset,-3,xAarray,tempQM_array)
#     tempdf['fubar_xA']=DY_xFxQ2(PDFset,-2,xAarray,tempQM_array)
#     tempdf['fdbar_xA']=DY_xFxQ2(PDFset,-1,xAarray,tempQM_array)
#     tempdf['fd_xA']=DY_xFxQ2(PDFset,1,xAarray,tempQM_array)
#     tempdf['fu_xA']=DY_xFxQ2(PDFset,2,xAarray,tempQM_array)
#     tempdf['fs_xA']=DY_xFxQ2(PDFset,3,xAarray,tempQM_array)
#     ####### Generating grids for x2 ###################
#     tempdf['fsbar_xB']=DY_xFxQ2(PDFset,-3,xBarray,tempQM_array)
#     tempdf['fubar_xB']=DY_xFxQ2(PDFset,-2,xBarray,tempQM_array)
#     tempdf['fdbar_xB']=DY_xFxQ2(PDFset,-1,xBarray,tempQM_array)
#     tempdf['fd_xB']=DY_xFxQ2(PDFset,1,xBarray,tempQM_array)
#     tempdf['fu_xB']=DY_xFxQ2(PDFset,2,xBarray,tempQM_array)
#     tempdf['fs_xB']=DY_xFxQ2(PDFset,3,xBarray,tempQM_array)
#     return tempdf

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")


def generate_DY_PDFs_for_evaluation(PDFset,df):
    tempQM = df['QM']
    distinct_QM = np.array(sorted(tempQM.unique()))
    QM_val_len = len(distinct_QM)
    tempxA = df['xA']
    minxA = np.min(tempxA)
    maxxA = np.max(tempxA)
    xAarray = np.linspace(minxA, maxxA, 100)
    tempxB = df['xB']
    minxB = np.min(tempxB)
    maxxB = np.max(tempxB)
    xBarray = np.linspace(minxB, maxxB, 100)
    tempPT = df['PT']
    minPT = np.min(tempPT)
    maxPT = np.max(tempPT)
    PTarray = np.linspace(minPT, maxPT, 100)
    create_folders('Eval_PDFs')
    for i in range(0,QM_val_len):
        tempdf = pd.DataFrame()
        tempQM_array = np.linspace(distinct_QM[i],distinct_QM[i],100)
        tempdf['xA'] = xAarray
        tempdf['xB'] = xBarray
        tempdf['QM'] = tempQM_array
        tempdf['PT'] = PTarray        
        ####### Generating grids for xA ###################
        tempdf['fsbar_xA']=DY_xFxQ2(PDFset,-3,xAarray,tempQM_array)
        tempdf['fubar_xA']=DY_xFxQ2(PDFset,-2,xAarray,tempQM_array)
        tempdf['fdbar_xA']=DY_xFxQ2(PDFset,-1,xAarray,tempQM_array)
        tempdf['fd_xA']=DY_xFxQ2(PDFset,1,xAarray,tempQM_array)
        tempdf['fu_xA']=DY_xFxQ2(PDFset,2,xAarray,tempQM_array)
        tempdf['fs_xA']=DY_xFxQ2(PDFset,3,xAarray,tempQM_array)
        ####### Generating grids for x2 ###################
        tempdf['fsbar_xB']=DY_xFxQ2(PDFset,-3,xBarray,tempQM_array)
        tempdf['fubar_xB']=DY_xFxQ2(PDFset,-2,xBarray,tempQM_array)
        tempdf['fdbar_xB']=DY_xFxQ2(PDFset,-1,xBarray,tempQM_array)
        tempdf['fd_xB']=DY_xFxQ2(PDFset,1,xBarray,tempQM_array)
        tempdf['fu_xB']=DY_xFxQ2(PDFset,2,xBarray,tempQM_array)
        tempdf['fs_xB']=DY_xFxQ2(PDFset,3,xBarray,tempQM_array)
        tempdf.to_csv('Eval_PDFs/'+'Eval_pdfs_'+str(i)+'.csv')
    #return tempdf

pdfs_eval_df = generate_DY_PDFs_for_evaluation(NNPDF4_nlo,E288_df)

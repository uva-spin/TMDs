import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lhapdf
import os


NNPDF4_lo=lhapdf.mkPDF('NNPDF40_lo_as_01180')
NNPDF4_nlo=lhapdf.mkPDF('NNPDF40_nlo_as_01180')
NNPDF4_nnlo=lhapdf.mkPDF('NNPDF40_nnlo_as_01180')
CTQ61_nlo=lhapdf.mkPDF('cteq61')

def xFxQ2(dataset,flavor,x,QQ):
    temp_parton_dist_x=[(dataset.xfxQ2(flavor, x[i], QQ)) for i in range(len(x))]
    return temp_parton_dist_x
    
def pdf(pdfset, flavor, x, QQ):
    return np.array(pdfset.xfxQ2(flavor, x, QQ))
    
def PDFs(pdfset,flavor,tempX):
    tempPDFs=[pdf(pdfset,flavor,tempX[i],2.4) for i in range(0,len(tempX))]
    return tempPDFs

tempx_vals=np.array(np.linspace(0.001,1,100))


data_dictionary_pdfs={"x":[],"fu":[],"fubar":[],"fd":[],"fdbar":[],"fs":[],"fsbar":[]}


def PDF_values_X(pdfset):
    temp_u=np.array(PDFs(pdfset,2,tempx_vals))
    temp_d=np.array(PDFs(pdfset,1,tempx_vals))
    temp_s=np.array(PDFs(pdfset,3,tempx_vals))
    temp_ubar=np.array(PDFs(pdfset,-2,tempx_vals))
    temp_dbar=np.array(PDFs(pdfset,-1,tempx_vals))
    temp_sbar=np.array(PDFs(pdfset,-3,tempx_vals))
    return np.array(temp_u),np.array(temp_ubar),np.array(temp_d),np.array(temp_dbar),np.array(temp_s),np.array(temp_sbar)
        
    
def PDFs_arrays(pdfset):
    tempX = tempx_vals
    temp_pdfs = PDF_values_X(pdfset)
    ######### PDF results ###########
    tempfu = np.array(temp_pdfs[0])
    tempfd = np.array(temp_pdfs[2])
    tempfs = np.array(temp_pdfs[4])
    tempfubar = np.array(temp_pdfs[1])
    tempfdbar = np.array(temp_pdfs[3])
    tempfsbar = np.array(temp_pdfs[5])
    ##################################
    data_dictionary_pdfs["x"]=tempX
    data_dictionary_pdfs["fu"]=tempfu
    data_dictionary_pdfs["fubar"]=tempfubar
    data_dictionary_pdfs["fd"]=tempfd
    data_dictionary_pdfs["fdbar"]=tempfdbar
    data_dictionary_pdfs["fs"]=tempfs
    data_dictionary_pdfs["fsbar"]=tempfsbar
    return pd.DataFrame(data_dictionary_pdfs)

PDFs_Set1 = PDFs_arrays(NNPDF4_lo)
PDFs_Set1.to_csv('PDFsets/NNPDF4_lo.csv')
PDFs_Set2 = PDFs_arrays(NNPDF4_nlo)
PDFs_Set2.to_csv('PDFsets/NNPDF4_nlo.csv')
PDFs_Set3 = PDFs_arrays(NNPDF4_nnlo)
PDFs_Set3.to_csv('PDFsets/NNPDF4_nnlo.csv')
PDFs_Set4 = PDFs_arrays(CTQ61_nlo)
PDFs_Set4.to_csv('PDFsets/CTQ61_nlo.csv')

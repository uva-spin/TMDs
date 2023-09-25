import lhapdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PathsR import *
from Constants import *
from Input_Parameterization import *


#### Introducing LHAPDFs PDFsets & FFsets

#PDFdataset = lhapdf.mkPDF("cteq61")
#PDFdataset = lhapdf.mkPDF("CT10nnlo")
#FF_pion_dataset=["JAM19FF_pion_nlo"]
#FF_kaon_dataset=["JAM19FF_kaon_nlo"]
#FF_PiP_dataset=["NNFF10_PIp_nlo"]
#FF_PiM_dataset=["NNFF10_PIm_nlo"]
#FF_Pi0_dataset=["NNFF10_PIsum_nlo"]
#FF_KP_dataset=["NNFF10_KAp_nlo"]
#FF_KM_dataset=["NNFF10_KAm_nlo"]

###########################################################################
#####################  SIDIS PDFs #########################################
###########################################################################

PDF_HERMES13p_cosphi = pd.read_csv(SIDIS_PDFs_HERMES13p_cosphi)
PDF_HERMES13p_cos2phi = pd.read_csv(SIDIS_PDFs_HERMES13p_cos2phi)
PDF_HERMES13d_cosphi = pd.read_csv(SIDIS_PDFs_HERMES13d_cosphi)
PDF_HERMES13d_cos2phi = pd.read_csv(SIDIS_PDFs_HERMES13d_cos2phi)


PDFs_Array = (PDF_HERMES13p_cosphi,PDF_HERMES13p_cos2phi)
###########################################################################
#####################  SIDIS FFs #########################################
###########################################################################

FF_HERMES13p_cosphi_PiP = pd.read_csv(SIDIS_FFs_PiP_HERMES13p_cosphi)
FF_HERMES13p_cos2phi_PiP = pd.read_csv(SIDIS_FFs_PiP_HERMES13p_cos2phi)
FF_HERMES13d_cosphi_PiP = pd.read_csv(SIDIS_FFs_PiP_HERMES13d_cosphi)
FF_HERMES13d_cos2phi_PiP = pd.read_csv(SIDIS_FFs_PiP_HERMES13d_cos2phi)

FF_HERMES13p_cosphi_PiM = pd.read_csv(SIDIS_FFs_PiM_HERMES13p_cosphi)
FF_HERMES13p_cos2phi_PiM = pd.read_csv(SIDIS_FFs_PiM_HERMES13p_cos2phi)
FF_HERMES13d_cosphi_PiM = pd.read_csv(SIDIS_FFs_PiM_HERMES13d_cosphi)
FF_HERMES13d_cos2phi_PiM = pd.read_csv(SIDIS_FFs_PiM_HERMES13d_cos2phi)

FF_HERMES13p_cosphi_Pi0 = pd.read_csv(SIDIS_FFs_Pi0_HERMES13p_cosphi)
FF_HERMES13p_cos2phi_Pi0 = pd.read_csv(SIDIS_FFs_Pi0_HERMES13p_cos2phi)
FF_HERMES13d_cosphi_Pi0 = pd.read_csv(SIDIS_FFs_Pi0_HERMES13d_cosphi)
FF_HERMES13d_cos2phi_Pi0 = pd.read_csv(SIDIS_FFs_Pi0_HERMES13d_cos2phi)

FF_HERMES13p_cosphi_KP = pd.read_csv(SIDIS_FFs_KP_HERMES13p_cosphi)
FF_HERMES13p_cos2phi_KP = pd.read_csv(SIDIS_FFs_KP_HERMES13p_cos2phi)
FF_HERMES13d_cosphi_KP = pd.read_csv(SIDIS_FFs_KP_HERMES13d_cosphi)
FF_HERMES13d_cos2phi_KP = pd.read_csv(SIDIS_FFs_KP_HERMES13d_cos2phi)

FF_HERMES13p_cosphi_KM = pd.read_csv(SIDIS_FFs_KM_HERMES13p_cosphi)
FF_HERMES13p_cos2phi_KM = pd.read_csv(SIDIS_FFs_KM_HERMES13p_cos2phi)
FF_HERMES13d_cosphi_KM = pd.read_csv(SIDIS_FFs_KM_HERMES13d_cosphi)
FF_HERMES13d_cos2phi_KM = pd.read_csv(SIDIS_FFs_KM_HERMES13d_cos2phi)


FFs_HERMES13p_cosphi=(FF_HERMES13p_cosphi_PiP,FF_HERMES13p_cosphi_PiM,FF_HERMES13p_cosphi_Pi0,FF_HERMES13p_cosphi_KP,FF_HERMES13p_cosphi_KM)
FFs_HERMES13p_cos2phi=(FF_HERMES13p_cos2phi_PiP,FF_HERMES13p_cos2phi_PiM,FF_HERMES13p_cos2phi_Pi0,FF_HERMES13p_cos2phi_KP,FF_HERMES13p_cos2phi_KM)
FFs_HERMES13d_cosphi=(FF_HERMES13d_cosphi_PiP,FF_HERMES13d_cosphi_PiM,FF_HERMES13d_cosphi_Pi0,FF_HERMES13d_cosphi_KP,FF_HERMES13d_cosphi_KM)
FFs_HERMES13d_cos2phi=(FF_HERMES13d_cos2phi_PiP,FF_HERMES13d_cos2phi_PiM,FF_HERMES13d_cos2phi_Pi0,FF_HERMES13d_cos2phi_KP,FF_HERMES13d_cos2phi_KM)

SIDIS_FFs_Data=[None]*(len(SIDIS_DataFilesArrayR))
SIDIS_FFs_Data[0]=(FF_HERMES13p_cosphi_PiP,FF_HERMES13p_cosphi_PiM,FF_HERMES13p_cosphi_Pi0,FF_HERMES13p_cosphi_KP,FF_HERMES13p_cosphi_KM)
SIDIS_FFs_Data[1]=(FF_HERMES13p_cos2phi_PiP,FF_HERMES13p_cos2phi_PiM,FF_HERMES13p_cos2phi_Pi0,FF_HERMES13p_cos2phi_KP,FF_HERMES13p_cos2phi_KM)
#SIDIS_FFs_Data=[None]*(4)
#SIDIS_FFs_Data[0]=(FF_HERMES_PiP_2009,FF_HERMES_PiM_2009,FF_HERMES_Pi0_2009,FF_HERMES_KP_2009,FF_HERMES_KM_2009)
#SIDIS_FFs_Data[0]=(FF_HERMES_PiP_2020,FF_HERMES_PiM_2020,FF_HERMES_Pi0_2020,FF_HERMES_KP_2020,FF_HERMES_KM_2020)
#SIDIS_FFs_Data[1]=(FF_COMPASS_PiP_2009,FF_COMPASS_PiM_2009,FF_COMPASS_Pi0_2009,FF_COMPASS_KP_2009,FF_COMPASS_KM_2009)
#SIDIS_FFs_Data[1]=(FF_COMPASS_PiP_2015,FF_COMPASS_PiM_2015,FF_COMPASS_Pi0_2015,FF_COMPASS_KP_2015,FF_COMPASS_KM_2015)

######################################################
########## SIDIS Asymmetry (Theory) ################## 
######################################################

def hadarray(filename):
    tempdf=pd.read_csv(filename)
    #tempdf=filename
    temphad_data=tempdf["hadron"]
    temphad=temphad_data.dropna().unique()
    refined_had_array=[]
    for i in range(0,len(temphad)):
        if((temphad[i]=="pi+") or (temphad[i]=="pi-") or (temphad[i]=="pi0") or (temphad[i]=="k+") or (temphad[i]=="k-")):
            refined_had_array.append(temphad[i])
    return refined_had_array


def pperp2avg(a,b,z):
    return a + b*(z**2)

def kBM2Avg(m1,kperp2Avg):
    temp=((m1**2)*kperp2Avg)/((m1**2)+kperp2Avg)
    return temp

def pc2Avg(pperp2Avg,mc):
    temp = ((mc**2)*pperp2Avg)/((mc**2)+pperp2Avg)
    return temp

def phT2Avg(pperp2Avg,kperp2Avg,z):
    temp = pperp2Avg + (kperp2Avg)*z**2
    return temp

def pht2avgBM(pperp2Avg,mc,z,m1,kperp2Avg):
    temp = pc2Avg(pperp2Avg,mc) + (z**2)*kBM2Avg(m1,kperp2Avg)
    return temp


def A0_cosphi_BM(y,z,pht,m1,mc,QQ,kperp2Avg,pperp2Avg,eCharg):
    temp1 = (2*(2-y)*(np.sqrt(1-y)))/(1+(1-y)**2)
    temp2 = (2*eCharg*pht)/(m1*mc*np.sqrt(QQ))
    temp3 = (pperp2Avg)/(pht2avgBM(pperp2Avg,mc,z,m1,kperp2Avg)**4)
    temp4 = np.exp(pht**2/pperp2Avg - pht**2/pht2avgBM(pperp2Avg,mc,z,m1,kperp2Avg))
    temp5 = ((kBM2Avg(m1,kperp2Avg)**2)*(pc2Avg(pperp2Avg,mc)**2))/(kperp2Avg*pperp2Avg)
    temp6 = (z**2)*kBM2Avg(m1,kperp2Avg)*(pht**2 - pht2avgBM(pperp2Avg,mc,z,m1,kperp2Avg))+ pc2Avg(pperp2Avg,mc)*pht2avgBM(pperp2Avg,mc,z,m1,kperp2Avg)
    tempfinal = temp1*temp2*temp3*temp4*temp5*temp6
    return tempfinal


def A0_cos2phi_BM(y,z,pht,m1,mc,QQ,kperp2Avg,pperp2Avg,eCharg):
    temp1 = (2*(2-y))/(1+(1-y)**2)
    temp2 = (-eCharg*(pht**2))/(m1*mc)
    temp3 = (pperp2Avg)/(pht2avgBM(pperp2Avg,mc,z,m1,kperp2Avg)**3)
    temp4 = np.exp(pht**2/pperp2Avg - pht**2/pht2avgBM(pperp2Avg,mc,z,m1,kperp2Avg))
    temp5 = ((kBM2Avg(m1,kperp2Avg)**2)*(pc2Avg(pperp2Avg,mc)**2))/(kperp2Avg*pperp2Avg)
    temp6 = (z**2)*kBM2Avg(m1,kperp2Avg)*(pht**2 - pht2avgBM(pperp2Avg,mc,z,m1,kperp2Avg))+ pc2Avg(pperp2Avg,mc)*pht2avgBM(pperp2Avg,mc,z,m1,kperp2Avg)
    tempfinal = temp1*temp2*temp3*temp4*temp5*temp6
    return tempfinal


def A0_cosphi_Cahn(y,z,pht,QQ,kperp2Avg,pperp2Avg,eCharg):
    temp1 = (2*(2-y)*(np.sqrt(1-y)))/(1+(1-y)**2)
    temp2 = (-2*eCharg*pht)/(np.sqrt(QQ))
    temp3 = (z*kperp2Avg)/(phT2Avg(pperp2Avg,kperp2Avg,z))
    tempfinal = temp1*temp2*temp3
    return tempfinal


def A0_cos2phi_Cahn(y,z,pht,QQ,kperp2Avg,pperp2Avg,eCharg):
    temp1 = (2*(2-y))/(1+(1-y)**2)
    temp2 = (-2*eCharg*pht)/(np.sqrt(QQ))
    temp3 = (z*kperp2Avg)/(phT2Avg(pperp2Avg,kperp2Avg,z))
    tempfinal = temp1*temp2*(temp3**2)
    return tempfinal


def SIDIS_xFxQ2(datafile,flavor,hadron,dep):
    # tempvals_all=pd.read_csv(datafile)
    tempvals_all=datafile
    tempvals=tempvals_all[(tempvals_all["hadron"]==hadron)&(tempvals_all["1D_dependence"]==dep)]
    tempx=tempvals['x']
    tempQQ=tempvals['QQ']
    if(flavor==-3):
        temp_PDF=tempvals['sbar']
    elif(flavor==-2):
        temp_PDF=tempvals['ubar']   
    elif(flavor==-1):
        temp_PDF=tempvals['dbar']
    if(flavor==1):
        temp_PDF=tempvals['d']
    elif(flavor==2):
        temp_PDF=tempvals['u']
    elif(flavor==3):
        temp_PDF=tempvals['s']                 
    return np.array(temp_PDF)

def SIDIS_zFzQ(datafile,flavor,hadron,dep):
#    tempvals=pd.read_csv(datafile)
    tempvals_all=datafile
    tempvals=tempvals_all[(tempvals_all["hadron"]==hadron)&(tempvals_all["1D_dependence"]==dep)]
    tempz=tempvals['z']
    tempQQ=tempvals['QQ']
    if(flavor==-3):
        temp_FF=tempvals['sbar']
    elif(flavor==-2):
        temp_FF=tempvals['ubar']   
    elif(flavor==-1):
        temp_FF=tempvals['dbar']
    if(flavor==1):
        temp_FF=tempvals['d']
    elif(flavor==2):
        temp_FF=tempvals['u']
    elif(flavor==3):
        temp_FF=tempvals['s']                 
    return np.array(temp_FF)


def Determine_PDFs_FFs(SIDISdatafilename,hadron):
    for i in range(0,len(PDFs_Array)):
        if(SIDISdatafilename==SIDIS_DataFilesArrayR[i]):
            PDFfile=PDFs_Array[i]
            if(hadron=='pi+'):
               FFfile=SIDIS_FFs_Data[i][0]
            elif(hadron=='pi-'):
               FFfile=SIDIS_FFs_Data[i][1]
            elif(hadron=='pi0'):
               FFfile=SIDIS_FFs_Data[i][2]                      
            elif(hadron=='k+'):
               FFfile=SIDIS_FFs_Data[i][3]                      
            elif(hadron=='k-'):
               FFfile=SIDIS_FFs_Data[i][4]                      
    return PDFfile,FFfile 


def Asymmetry_cosphi_BM(SIDISdatafilename,hadron,dep,**parms):
    m1= parms["m1"]
    Nu = parms["Nu"]
    alphau= parms["au"]
    betau = parms["bu"]
    Nubar = parms["Nub"]
    alphaub= parms["aub"]
    betaub = parms["bub"]    
    Nubar = parms["Nub"]
    Nd = parms["Nd"]
    alphad= parms["ad"]
    betad = parms["bd"]
    Ndbar = parms["Ndb"]
    alphadb= parms["adb"]
    betadb = parms["bdb"]
    Ns = parms["Ns"]
    alphas= parms["aS"]
    betas = parms["bS"]
    Nsbar = parms["Nsb"]
    alphasb= parms["aSb"]
    betasb = parms["bSb"]
    kperp2Avg=Kp2A
    pperp2Avg=Pp2A
    eCharg=ee
    PDFfile=Determine_PDFs_FFs(SIDISdatafilename,hadron)[0]
    FFfile=Determine_PDFs_FFs(SIDISdatafilename,hadron)[1]
    tempvals_all=PDFfile
    #tempvals=tempvals_all[(tempvals_all["hadron"]==hadron)]
    tempvals=tempvals_all[(tempvals_all["hadron"]==hadron)&(tempvals_all["1D_dependence"]==dep)]
    QQ=tempvals['QQ']
    x=tempvals['x']
    y=tempvals['x']
    z=tempvals['z']
    phT=tempvals['phT']
    uCont1= NNq(x,Nu,alphau,betau)*(eU**2)*SIDIS_xFxQ2(PDFfile,2,hadron,dep)*SIDIS_zFzQ(FFfile,2,hadron,dep)
    ubarCont1= NNqbar(x,Nubar,alphaub,betaub)*(eUbar**2)*SIDIS_xFxQ2(PDFfile,-2,hadron,dep)*SIDIS_zFzQ(FFfile,-2,hadron,dep)
    dCont1= NNq(x,Nd,alphad,betad)*(eD**2)*SIDIS_xFxQ2(PDFfile,1,hadron,dep)*SIDIS_zFzQ(FFfile,1,hadron,dep)
    dbarCont1= NNqbar(x,Ndbar,alphadb,betadb)*(eDbar**2)*SIDIS_xFxQ2(PDFfile,-1,hadron,dep)*SIDIS_zFzQ(FFfile,-1,hadron,dep)
    sCont1= NNq(x,Ns,alphas,betas)*(eS**2)*SIDIS_xFxQ2(PDFfile,3,hadron,dep)*SIDIS_zFzQ(FFfile,3,hadron,dep)
    sbarCont1= NNqbar(x,Nsbar,alphasb,betasb)*(eSbar**2)*SIDIS_xFxQ2(PDFfile,-3,hadron,dep)*SIDIS_zFzQ(FFfile,-3,hadron,dep)
    uCont2= (eU**2)*SIDIS_xFxQ2(PDFfile,2,hadron,dep)*SIDIS_zFzQ(FFfile,2,hadron,dep)
    ubarCont2= (eUbar**2)*SIDIS_xFxQ2(PDFfile,-2,hadron,dep)*SIDIS_zFzQ(FFfile,-2,hadron,dep)
    dCont2= (eD**2)*SIDIS_xFxQ2(PDFfile,1,hadron,dep)*SIDIS_zFzQ(FFfile,1,hadron,dep)
    dbarCont2=(eDbar**2)*SIDIS_xFxQ2(PDFfile,-1,hadron,dep)*SIDIS_zFzQ(FFfile,-1,hadron,dep)
    sCont2= (eS**2)*SIDIS_xFxQ2(PDFfile,3,hadron,dep)*SIDIS_zFzQ(FFfile,3,hadron,dep)
    sbarCont2= (eSbar**2)*SIDIS_xFxQ2(PDFfile,-3,hadron,dep)*SIDIS_zFzQ(FFfile,-3,hadron,dep)
    tempNumerator = uCont1 + ubarCont1 +dCont1 + dbarCont1 + sCont1 + sbarCont1
    tempDenominator = uCont2 + ubarCont2 +dCont2 + dbarCont2 + sCont2 + sbarCont2
    tempASiv_Hadron = A0_cosphi_BM(y,z,phT,m1,mcval,QQ,kperp2Avg,pperp2Avg,eCharg)*(tempNumerator/tempDenominator)
    return tempASiv_Hadron


def Asymmetry_cosphi_Cahn(SIDISdatafilename,hadron,dep):
    kperp2Avg=Kp2A
    pperp2Avg=Pp2A
    eCharg=ee
    tempvals_all=pd.read_csv(SIDISdatafilename)
    tempvals=tempvals_all[(tempvals_all["hadron"]==hadron)&(tempvals_all["1D_dependence"]==dep)]
    QQ=tempvals['Q2']
    x=tempvals['x']
    y=tempvals['x']
    z=tempvals['z']
    phT=tempvals['phT']
    Asym_vals=tempvals['Siv']
    Asym_err =tempvals['tot_err']
    tempASiv_Hadron = A0_cosphi_Cahn(y,z,phT,QQ,kperp2Avg,pperp2Avg,eCharg)
    return tempASiv_Hadron,Asym_vals,Asym_err


def Asymmetry_cos2phi_BM(SIDISdatafilename,hadron,dep,**parms):
    m1= parms["m1"]
    Nu = parms["Nu"]
    alphau= parms["au"]
    betau = parms["bu"]
    Nubar = parms["Nub"]
    alphaub= parms["aub"]
    betaub = parms["bub"]    
    Nubar = parms["Nub"]
    Nd = parms["Nd"]
    alphad= parms["ad"]
    betad = parms["bd"]
    Ndbar = parms["Ndb"]
    alphadb= parms["adb"]
    betadb = parms["bdb"]
    Ns = parms["Ns"]
    alphas= parms["aS"]
    betas = parms["bS"]
    Nsbar = parms["Nsb"]
    alphasb= parms["aSb"]
    betasb = parms["bSb"]
    kperp2Avg=Kp2A
    pperp2Avg=Pp2A
    eCharg=ee
    PDFfile=Determine_PDFs_FFs(SIDISdatafilename,hadron)[0]
    FFfile=Determine_PDFs_FFs(SIDISdatafilename,hadron)[1]
    tempvals_all=PDFfile
    #tempvals=tempvals_all[(tempvals_all["hadron"]==hadron)]
    tempvals=tempvals_all[(tempvals_all["hadron"]==hadron)&(tempvals_all["1D_dependence"]==dep)]
    QQ=tempvals['QQ']
    x=tempvals['x']
    y=tempvals['x']
    z=tempvals['z']
    phT=tempvals['phT']
    uCont1= NNq(x,Nu,alphau,betau)*(eU**2)*SIDIS_xFxQ2(PDFfile,2,hadron,dep)*SIDIS_zFzQ(FFfile,2,hadron,dep)
    ubarCont1= NNqbar(x,Nubar,alphaub,betaub)*(eUbar**2)*SIDIS_xFxQ2(PDFfile,-2,hadron,dep)*SIDIS_zFzQ(FFfile,-2,hadron,dep)
    dCont1= NNq(x,Nd,alphad,betad)*(eD**2)*SIDIS_xFxQ2(PDFfile,1,hadron,dep)*SIDIS_zFzQ(FFfile,1,hadron,dep)
    dbarCont1= NNqbar(x,Ndbar,alphadb,betadb)*(eDbar**2)*SIDIS_xFxQ2(PDFfile,-1,hadron,dep)*SIDIS_zFzQ(FFfile,-1,hadron,dep)
    sCont1= NNq(x,Ns,alphas,betas)*(eS**2)*SIDIS_xFxQ2(PDFfile,3,hadron,dep)*SIDIS_zFzQ(FFfile,3,hadron,dep)
    sbarCont1= NNqbar(x,Nsbar,alphasb,betasb)*(eSbar**2)*SIDIS_xFxQ2(PDFfile,-3,hadron,dep)*SIDIS_zFzQ(FFfile,-3,hadron,dep)
    uCont2= (eU**2)*SIDIS_xFxQ2(PDFfile,2,hadron,dep)*SIDIS_zFzQ(FFfile,2,hadron,dep)
    ubarCont2= (eUbar**2)*SIDIS_xFxQ2(PDFfile,-2,hadron,dep)*SIDIS_zFzQ(FFfile,-2,hadron,dep)
    dCont2= (eD**2)*SIDIS_xFxQ2(PDFfile,1,hadron,dep)*SIDIS_zFzQ(FFfile,1,hadron,dep)
    dbarCont2=(eDbar**2)*SIDIS_xFxQ2(PDFfile,-1,hadron,dep)*SIDIS_zFzQ(FFfile,-1,hadron,dep)
    sCont2= (eS**2)*SIDIS_xFxQ2(PDFfile,3,hadron,dep)*SIDIS_zFzQ(FFfile,3,hadron,dep)
    sbarCont2= (eSbar**2)*SIDIS_xFxQ2(PDFfile,-3,hadron,dep)*SIDIS_zFzQ(FFfile,-3,hadron,dep)
    tempNumerator = uCont1 + ubarCont1 +dCont1 + dbarCont1 + sCont1 + sbarCont1
    tempDenominator = uCont2 + ubarCont2 +dCont2 + dbarCont2 + sCont2 + sbarCont2
    tempASiv_Hadron = A0_cos2phi_BM(y,z,phT,m1,mcval,QQ,kperp2Avg,pperp2Avg,eCharg)*(tempNumerator/tempDenominator)
    return tempASiv_Hadron
    

def Asymmetry_cos2phi_Cahn(SIDISdatafilename,hadron,dep):
    kperp2Avg=Kp2A
    pperp2Avg=Pp2A
    eCharg=ee
    tempvals_all=pd.read_csv(SIDISdatafilename)
    tempvals=tempvals_all[(tempvals_all["hadron"]==hadron)&(tempvals_all["1D_dependence"]==dep)]
    QQ=tempvals['Q2']
    x=tempvals['x']
    y=tempvals['x']
    z=tempvals['z']
    phT=tempvals['phT']
    Asym_vals=tempvals['Siv']
    Asym_err =tempvals['tot_err']
    tempASiv_Hadron = A0_cos2phi_Cahn(y,z,phT,QQ,kperp2Avg,pperp2Avg,eCharg)
    return tempASiv_Hadron,Asym_vals,Asym_err



def Asymmetry_cosphi(SIDISdatafilename,hadron,dep,**parms):
    temp_theory=Asymmetry_cosphi_BM(SIDISdatafilename,hadron,dep,**parms) + Asymmetry_cosphi_Cahn(SIDISdatafilename,hadron,dep)[0]
    temp_data=Asymmetry_cosphi_Cahn(SIDISdatafilename,hadron,dep)[1]
    temp_err=Asymmetry_cosphi_Cahn(SIDISdatafilename,hadron,dep)[2]
    return temp_theory, temp_data, temp_err


def Asymmetry_cos2phi(SIDISdatafilename,hadron,dep,**parms):
    temp_theory = Asymmetry_cos2phi_BM(SIDISdatafilename,hadron,dep,**parms) + Asymmetry_cos2phi_Cahn(SIDISdatafilename,hadron,dep)[0]
    temp_data=Asymmetry_cos2phi_Cahn(SIDISdatafilename,hadron,dep)[1]
    temp_err=Asymmetry_cos2phi_Cahn(SIDISdatafilename,hadron,dep)[2]
    return temp_theory, temp_data, temp_err


def totalfitDataSet_cosphi(datfile,**parms):
    m1= parms["m1"]
    Nu = parms["Nu"]
    au= parms["au"]
    bu = parms["bu"]
    Nub = parms["Nub"]
    aub= parms["aub"]
    bub = parms["bub"]    
    Nub = parms["Nub"]
    Nd = parms["Nd"]
    ad= parms["ad"]
    bd = parms["bd"]
    Ndb = parms["Ndb"]
    adb= parms["adb"]
    bdb = parms["bdb"]
    Ns = parms["Ns"]
    aS= parms["aS"]
    bS = parms["bS"]
    Nsb = parms["Nsb"]
    asb= parms["aSb"]
    bsb = parms["bSb"]
    #had_len=len(hadarray(datfile))
    temHads=hadarray(datfile)
    had_len=len(temHads)
    fittot=[]
    datvals=[]
    errvals=[]
    for i in range(0,had_len):
        if temHads[i]=="pi+":
            tempfitx=Asymmetry_cosphi(datfile,"pi+","x",**parms)[0]
            tempfitz=Asymmetry_cosphi(datfile,"pi+","z",**parms)[0]
            tempfitphT=Asymmetry_cosphi(datfile,"pi+","phT",**parms)[0]
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT))
            fittot.append(tempfit)
            tempdatx=Asymmetry_cosphi(datfile,"pi+","x",**parms)[1]
            tempdatz=Asymmetry_cosphi(datfile,"pi+","z",**parms)[1]
            tempdatphT=Asymmetry_cosphi(datfile,"pi+","phT",**parms)[1]
            tempdat=np.concatenate((tempdatx,tempdatz,tempdatphT))
            datvals.append(tempdat)
            temperrx=Asymmetry_cosphi(datfile,"pi+","x",**parms)[2]
            temperrz=Asymmetry_cosphi(datfile,"pi+","z",**parms)[2]
            temperrphT=Asymmetry_cosphi(datfile,"pi+","phT",**parms)[2]
            temperr=np.concatenate((temperrx,temperrz,temperrphT))
            errvals.append(temperr)
        elif temHads[i]=="pi-":
            tempfitx=Asymmetry_cosphi(datfile,"pi-","x",**parms)[0]
            tempfitz=Asymmetry_cosphi(datfile,"pi-","z",**parms)[0]
            tempfitphT=Asymmetry_cosphi(datfile,"pi-","phT",**parms)[0]
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT))
            fittot.append(tempfit)
            tempdatx=Asymmetry_cosphi(datfile,"pi-","x",**parms)[1]
            tempdatz=Asymmetry_cosphi(datfile,"pi-","z",**parms)[1]
            tempdatphT=Asymmetry_cosphi(datfile,"pi-","phT",**parms)[1]
            tempdat=np.concatenate((tempdatx,tempdatz,tempdatphT))
            datvals.append(tempdat)
            temperrx=Asymmetry_cosphi(datfile,"pi-","x",**parms)[2]
            temperrz=Asymmetry_cosphi(datfile,"pi-","z",**parms)[2]
            temperrphT=Asymmetry_cosphi(datfile,"pi-","phT",**parms)[2]
            temperr=np.concatenate((temperrx,temperrz,temperrphT))
            errvals.append(temperr)
        elif temHads[i]=="pi0":
            tempfitx=Asymmetry_cosphi(datfile,"pi0","x",**parms)[0]
            tempfitz=Asymmetry_cosphi(datfile,"pi0","z",**parms)[0]
            tempfitphT=Asymmetry_cosphi(datfile,"pi0","phT",**parms)[0]
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT))
            fittot.append(tempfit)
            tempdatx=Asymmetry_cosphi(datfile,"pi0","x",**parms)[1]
            tempdatz=Asymmetry_cosphi(datfile,"pi0","z",**parms)[1]
            tempdatphT=Asymmetry_cosphi(datfile,"pi0","phT",**parms)[1]
            tempdat=np.concatenate((tempdatx,tempdatz,tempdatphT))
            datvals.append(tempdat)
            temperrx=Asymmetry_cosphi(datfile,"pi0","x",**parms)[2]
            temperrz=Asymmetry_cosphi(datfile,"pi0","z",**parms)[2]
            temperrphT=Asymmetry_cosphi(datfile,"pi0","phT",**parms)[2]
            temperr=np.concatenate((temperrx,temperrz,temperrphT))
            errvals.append(temperr)
        elif temHads[i]=="k+":
            tempfitx=Asymmetry_cosphi(datfile,"k+","x",**parms)[0]
            tempfitz=Asymmetry_cosphi(datfile,"k+","z",**parms)[0]
            tempfitphT=Asymmetry_cosphi(datfile,"k+","phT",**parms)[0]
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT))
            fittot.append(tempfit)
            tempdatx=Asymmetry_cosphi(datfile,"k+","x",**parms)[1]
            tempdatz=Asymmetry_cosphi(datfile,"k+","z",**parms)[1]
            tempdatphT=Asymmetry_cosphi(datfile,"k+","phT",**parms)[1]
            tempdat=np.concatenate((tempdatx,tempdatz,tempdatphT))
            datvals.append(tempdat)
            temperrx=Asymmetry_cosphi(datfile,"k+","x",**parms)[2]
            temperrz=Asymmetry_cosphi(datfile,"k+","z",**parms)[2]
            temperrphT=Asymmetry_cosphi(datfile,"k+","phT",**parms)[2]
            temperr=np.concatenate((temperrx,temperrz,temperrphT))
            errvals.append(temperr)
        elif temHads[i]=="k-":
            tempfitx=Asymmetry_cosphi(datfile,"k-","x",**parms)[0]
            tempfitz=Asymmetry_cosphi(datfile,"k-","z",**parms)[0]
            tempfitphT=Asymmetry_cosphi(datfile,"k-","phT",**parms)[0]
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT))
            fittot.append(tempfit)
            tempdatx=Asymmetry_cosphi(datfile,"k-","x",**parms)[1]
            tempdatz=Asymmetry_cosphi(datfile,"k-","z",**parms)[1]
            tempdatphT=Asymmetry_cosphi(datfile,"k-","phT",**parms)[1]
            tempdat=np.concatenate((tempdatx,tempdatz,tempdatphT))
            datvals.append(tempdat)
            temperrx=Asymmetry_cosphi(datfile,"k-","x",**parms)[2]
            temperrz=Asymmetry_cosphi(datfile,"k-","z",**parms)[2]
            temperrphT=Asymmetry_cosphi(datfile,"k-","phT",**parms)[2]
            temperr=np.concatenate((temperrx,temperrz,temperrphT))
            errvals.append(temperr)
    return np.concatenate((fittot), axis=None),np.concatenate((datvals), axis=None),np.concatenate((errvals), axis=None)


def totalfitDataSet_cos2phi(datfile,**parms):
    m1= parms["m1"]
    Nu = parms["Nu"]
    au= parms["au"]
    bu = parms["bu"]
    Nub = parms["Nub"]
    aub= parms["aub"]
    bub = parms["bub"]    
    Nub = parms["Nub"]
    Nd = parms["Nd"]
    ad= parms["ad"]
    bd = parms["bd"]
    Ndb = parms["Ndb"]
    adb= parms["adb"]
    bdb = parms["bdb"]
    Ns = parms["Ns"]
    aS= parms["aS"]
    bS = parms["bS"]
    Nsb = parms["Nsb"]
    asb= parms["aSb"]
    bsb = parms["bSb"]
    #had_len=len(hadarray(datfile))
    temHads=hadarray(datfile)
    had_len=len(temHads)
    fittot=[]
    datvals=[]
    errvals=[]
    for i in range(0,had_len):
        if temHads[i]=="pi+":
            tempfitx=Asymmetry_cos2phi(datfile,"pi+","x",**parms)[0]
            tempfitz=Asymmetry_cos2phi(datfile,"pi+","z",**parms)[0]
            tempfitphT=Asymmetry_cos2phi(datfile,"pi+","phT",**parms)[0]
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT))
            fittot.append(tempfit)
            tempdatx=Asymmetry_cos2phi(datfile,"pi+","x",**parms)[1]
            tempdatz=Asymmetry_cos2phi(datfile,"pi+","z",**parms)[1]
            tempdatphT=Asymmetry_cos2phi(datfile,"pi+","phT",**parms)[1]
            tempdat=np.concatenate((tempdatx,tempdatz,tempdatphT))
            datvals.append(tempdat)
            temperrx=Asymmetry_cos2phi(datfile,"pi+","x",**parms)[2]
            temperrz=Asymmetry_cos2phi(datfile,"pi+","z",**parms)[2]
            temperrphT=Asymmetry_cos2phi(datfile,"pi+","phT",**parms)[2]
            temperr=np.concatenate((temperrx,temperrz,temperrphT))
            errvals.append(temperr)
        elif temHads[i]=="pi-":
            tempfitx=Asymmetry_cos2phi(datfile,"pi-","x",**parms)[0]
            tempfitz=Asymmetry_cos2phi(datfile,"pi-","z",**parms)[0]
            tempfitphT=Asymmetry_cos2phi(datfile,"pi-","phT",**parms)[0]
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT))
            fittot.append(tempfit)
            tempdatx=Asymmetry_cos2phi(datfile,"pi-","x",**parms)[1]
            tempdatz=Asymmetry_cos2phi(datfile,"pi-","z",**parms)[1]
            tempdatphT=Asymmetry_cos2phi(datfile,"pi-","phT",**parms)[1]
            tempdat=np.concatenate((tempdatx,tempdatz,tempdatphT))
            datvals.append(tempdat)
            temperrx=Asymmetry_cos2phi(datfile,"pi-","x",**parms)[2]
            temperrz=Asymmetry_cos2phi(datfile,"pi-","z",**parms)[2]
            temperrphT=Asymmetry_cos2phi(datfile,"pi-","phT",**parms)[2]
            temperr=np.concatenate((temperrx,temperrz,temperrphT))
            errvals.append(temperr)
        elif temHads[i]=="pi0":
            tempfitx=Asymmetry_cos2phi(datfile,"pi0","x",**parms)[0]
            tempfitz=Asymmetry_cos2phi(datfile,"pi0","z",**parms)[0]
            tempfitphT=Asymmetry_cos2phi(datfile,"pi0","phT",**parms)[0]
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT))
            fittot.append(tempfit)
            tempdatx=Asymmetry_cos2phi(datfile,"pi0","x",**parms)[1]
            tempdatz=Asymmetry_cos2phi(datfile,"pi0","z",**parms)[1]
            tempdatphT=Asymmetry_cos2phi(datfile,"pi0","phT",**parms)[1]
            tempdat=np.concatenate((tempdatx,tempdatz,tempdatphT))
            datvals.append(tempdat)
            temperrx=Asymmetry_cos2phi(datfile,"pi0","x",**parms)[2]
            temperrz=Asymmetry_cos2phi(datfile,"pi0","z",**parms)[2]
            temperrphT=Asymmetry_cos2phi(datfile,"pi0","phT",**parms)[2]
            temperr=np.concatenate((temperrx,temperrz,temperrphT))
            errvals.append(temperr)
        elif temHads[i]=="k+":
            tempfitx=Asymmetry_cos2phi(datfile,"k+","x",**parms)[0]
            tempfitz=Asymmetry_cos2phi(datfile,"k+","z",**parms)[0]
            tempfitphT=Asymmetry_cos2phi(datfile,"k+","phT",**parms)[0]
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT))
            fittot.append(tempfit)
            tempdatx=Asymmetry_cos2phi(datfile,"k+","x",**parms)[1]
            tempdatz=Asymmetry_cos2phi(datfile,"k+","z",**parms)[1]
            tempdatphT=Asymmetry_cos2phi(datfile,"k+","phT",**parms)[1]
            tempdat=np.concatenate((tempdatx,tempdatz,tempdatphT))
            datvals.append(tempdat)
            temperrx=Asymmetry_cos2phi(datfile,"k+","x",**parms)[2]
            temperrz=Asymmetry_cos2phi(datfile,"k+","z",**parms)[2]
            temperrphT=Asymmetry_cos2phi(datfile,"k+","phT",**parms)[2]
            temperr=np.concatenate((temperrx,temperrz,temperrphT))
            errvals.append(temperr)
        elif temHads[i]=="k-":
            tempfitx=Asymmetry_cos2phi(datfile,"k-","x",**parms)[0]
            tempfitz=Asymmetry_cos2phi(datfile,"k-","z",**parms)[0]
            tempfitphT=Asymmetry_cos2phi(datfile,"k-","phT",**parms)[0]
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT))
            fittot.append(tempfit)
            tempdatx=Asymmetry_cos2phi(datfile,"k-","x",**parms)[1]
            tempdatz=Asymmetry_cos2phi(datfile,"k-","z",**parms)[1]
            tempdatphT=Asymmetry_cos2phi(datfile,"k-","phT",**parms)[1]
            tempdat=np.concatenate((tempdatx,tempdatz,tempdatphT))
            datvals.append(tempdat)
            temperrx=Asymmetry_cos2phi(datfile,"k-","x",**parms)[2]
            temperrz=Asymmetry_cos2phi(datfile,"k-","z",**parms)[2]
            temperrphT=Asymmetry_cos2phi(datfile,"k-","phT",**parms)[2]
            temperr=np.concatenate((temperrx,temperrz,temperrphT))
            errvals.append(temperr)
    return np.concatenate((fittot), axis=None),np.concatenate((datvals), axis=None),np.concatenate((errvals), axis=None)

###########################################################
########## SIDIS Asymmetry (Data)  ########################
###########################################################


def dataslice(filename,Had,Var):
    #tempdf=pd.read_csv(filename)
    tempdf=filename
    temp_slice=tempdf[(tempdf["hadron"]==Had)&(tempdf["1D_dependence"]==Var)]
    tempQ2=np.array(temp_slice["Q2"],dtype=object)
    tempX=np.array(temp_slice["x"],dtype=object)
    tempZ=np.array(temp_slice["z"],dtype=object)
    tempPHT=np.array(temp_slice["phT"],dtype=object)
    tempSiv=np.array(temp_slice["Siv"],dtype=object)
    temperrSiv=np.array(temp_slice["tot_err"],dtype=object)
    return tempQ2,tempX,tempZ,tempPHT,tempSiv,temperrSiv


# def Kin_hadron(datfile,hadron):
#     tempXfile=dataslice(datfile,hadron,"x")
#     tempZfile=dataslice(datfile,hadron,"z")
#     tempPhTfile=dataslice(datfile,hadron,"phT")
#     ##### Q2 ################
#     tempQ2_x=np.array(tempXfile[0],dtype=object)
#     tempQ2_z=np.array(tempZfile[0],dtype=object)
#     tempQ2_phT=np.array(tempPhTfile[0],dtype=object)
#     tempQ2=np.concatenate((tempQ2_x,tempQ2_z,tempQ2_phT))
#     ##### X ################
#     tempX_x=np.array(tempXfile[1],dtype=object)
#     tempX_z=np.array(tempZfile[1],dtype=object)
#     tempX_phT=np.array(tempPhTfile[1],dtype=object)
#     tempX=np.concatenate((tempX_x,tempX_z,tempX_phT))
#     ##### Z ################
#     tempZ_x=np.array(tempXfile[2],dtype=object)
#     tempZ_z=np.array(tempZfile[2],dtype=object)
#     tempZ_phT=np.array(tempPhTfile[2],dtype=object)
#     tempZ=np.concatenate((tempZ_x,tempZ_z,tempZ_phT))
#     ##### phT ################
#     tempphT_x=np.array(tempXfile[3],dtype=object)
#     tempphT_z=np.array(tempZfile[3],dtype=object)
#     tempphT_phT=np.array(tempPhTfile[3],dtype=object)
#     tempphT=np.concatenate((tempphT_x,tempphT_z,tempphT_phT))
#     return tempQ2,tempX,tempZ,tempphT


def ASiv_data(datfile,hadron):
    #tempdf=pd.read_csv(datfile)
    tempdf=datfile
    tempXfile=dataslice(tempdf,hadron,"x")
    tempZfile=dataslice(tempdf,hadron,"z")
    tempPhTfile=dataslice(tempdf,hadron,"phT")    
    ##### Asy ################
    tempAsy_x=np.array(tempXfile[4],dtype=object)
    tempAsy_z=np.array(tempZfile[4],dtype=object)
    tempAsy_phT=np.array(tempPhTfile[4],dtype=object)
    tempAsy=np.concatenate((tempAsy_x,tempAsy_z,tempAsy_phT))
    ##### err ################
    tempAsyErr_x=np.array(tempXfile[5],dtype=object)
    tempAsyErr_z=np.array(tempZfile[5],dtype=object)
    tempAsyErr_phT=np.array(tempPhTfile[5],dtype=object)
    tempAsyErr=np.concatenate((tempAsyErr_x,tempAsyErr_z,tempAsyErr_phT))
    return tempAsy,tempAsyErr

def ASiv_Val(datfile):
    #had_len=len(hadarray(datfile))
    temHads=hadarray(datfile)
    had_len=len(temHads)
    temp_SivData=[]
    temp_SivErr=[]
    tempdf=pd.read_csv(datfile)
    for i in range(0,had_len):
        temp_SivData.append(ASiv_data(tempdf,temHads[i])[0])
        temp_SivErr.append(ASiv_data(tempdf,temHads[i])[1])        
    return temp_SivData, temp_SivErr

# def ASiv_Err(datfile):
#     #had_len=len(hadarray(datfile))
#     temHads=hadarray(datfile)
#     had_len=len(temHads)
#     temp_SivData=[]
#     tempdf=pd.read_csv(datfile)
#     for i in range(0,had_len):
#         temp_SivData.append(ASiv_data(tempdf,temHads[i])[1])        
#     return temp_SivData









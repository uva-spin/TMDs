import os
import tensorflow as tf
import pandas as pd
import numpy as np
import lhapdf
import matplotlib.pyplot as plt


from iminuit import Minuit
import numpy as np


Mp=0.938272
alpha_s=1/(137.0359998)

Kp2A=0.03
#Pp2A=0.12
#p2unp=0.25

ee=1
eU=2/3
eUbar=-2/3
eD=-1/3
eDbar=1/3
eS=-1/3
eSbar=1/3

qCharge=np.array([eSbar,eUbar,eDbar,eU,eD,eS])
qFlavor=np.array([-3,-2,-1,1,2,3])


def NNq(x,Nq,aq,bq):
    tempNNq = Nq*(x**aq)*((1-x)**(bq))*((aq+bq)**(aq+bq))/((aq**aq)*(bq**bq))
    return tempNNq

def NNqbar(x,Nq,aq,bq):
    tempNNq = Nq*(x**aq)*((1-x)**(bq))*((aq+bq)**(aq+bq))/((aq**aq)*(bq**bq))
    return tempNNq


## PATHS


## SIDIS ###
Dat1='../../Data/HERMES13p.csv'
SIDIS_DataFilesArrayR=[Dat1]

SIDIS_PDFs_HERMES13p='../Calc_Grids_DSS/SIDIS_PDFs/PDFs_HERMES13p.csv'

SIDIS_PDFs_Array=(SIDIS_PDFs_HERMES13p)

###########################################################################
#####################  SIDIS FFs #########################################
###########################################################################


SIDIS_FFs_PiP_HERMES13p='../Calc_Grids_DSS/SIDIS_FFs/FF_PiP_HERMES13p.csv'

SIDIS_FFs_PiM_HERMES13p='../Calc_Grids_DSS/SIDIS_FFs/FF_PiM_HERMES13p.csv'

SIDIS_FFs_Pi0_HERMES13p='../Calc_Grids_DSS/SIDIS_FFs/FF_Pi0_HERMES13p.csv'

SIDIS_FFs_KP_HERMES13p='../Calc_Grids_DSS/SIDIS_FFs/FF_KP_HERMES13p.csv'

SIDIS_FFs_KM_HERMES13p='../Calc_Grids_DSS/SIDIS_FFs/FF_KM_HERMES13p.csv'

SIDIS_FFs_HERMES13p=(SIDIS_FFs_PiP_HERMES13p)



###########################################################################
#####################  SIDIS PDFs #########################################
###########################################################################

PDF_HERMES13p = pd.read_csv(SIDIS_PDFs_HERMES13p)


PDFs_Array = (PDF_HERMES13p)
###########################################################################
#####################  SIDIS FFs #########################################
###########################################################################

FF_HERMES13p_PiP = pd.read_csv(SIDIS_FFs_PiP_HERMES13p)

FF_HERMES13p_PiM = pd.read_csv(SIDIS_FFs_PiM_HERMES13p)

FF_HERMES13p_Pi0 = pd.read_csv(SIDIS_FFs_Pi0_HERMES13p)

FF_HERMES13p_KP = pd.read_csv(SIDIS_FFs_KP_HERMES13p)

FF_HERMES13p_KM = pd.read_csv(SIDIS_FFs_KM_HERMES13p)


FFs_HERMES13p=(FF_HERMES13p_PiP,FF_HERMES13p_PiM,FF_HERMES13p_Pi0,FF_HERMES13p_KP,FF_HERMES13p_KM)

SIDIS_FFs_Data=[None]*(len(SIDIS_DataFilesArrayR))
SIDIS_FFs_Data[0]=(FF_HERMES13p_PiP,FF_HERMES13p_PiM,FF_HERMES13p_Pi0,FF_HERMES13p_KP,FF_HERMES13p_KM)
#SIDIS_FFs_Data[1]=(FF_HERMES13p_cos2phi_PiP,FF_HERMES13p_cos2phi_PiM,FF_HERMES13p_cos2phi_Pi0,FF_HERMES13p_cos2phi_KP,FF_HERMES13p_cos2phi_KM)



Org_Data_path = '../../Data/'
HERMES13p= 'HERMES13p.csv'
herm13p = pd.read_csv(Org_Data_path + HERMES13p).dropna(axis=0, how='all').dropna(axis=1, how='all')
df = pd.concat([herm13p])



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

# def Determine_PDFs_FFs(SIDISdatafilename,hadron):
#     for i in range(0,len(PDFs_Array)):
#         if(SIDISdatafilename==SIDIS_DataFilesArrayR[i]):
#             PDFfile=PDFs_Array[i]
#             if(hadron=='pi+'):
#                FFfile=SIDIS_FFs_Data[i][0]
#             elif(hadron=='pi-'):
#                FFfile=SIDIS_FFs_Data[i][1]
#             elif(hadron=='pi0'):
#                FFfile=SIDIS_FFs_Data[i][2]                      
#             elif(hadron=='k+'):
#                FFfile=SIDIS_FFs_Data[i][3]                      
#             elif(hadron=='k-'):
#                FFfile=SIDIS_FFs_Data[i][4]                      
#     return PDFfile,FFfile 


PDFfile=PDFs_Array
# FFfile=FF_HERMES13p_PiP
# FFfile=FF_HERMES13p_PiM
# FFfile=FF_HERMES13p_Pi0                     
# FFfile=FF_HERMES13p_KP                      
# FFfile=FF_HERMES13p_KM                   

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


def chisquare(y, yhat, err):
    return np.sum(((y - yhat)/err)**2)

def pperp2avgVal(a,b,z):
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


def NCq(had,flavor,z):
    gamma=1.06
    delta=0.07
    temp_zfactor=(z**gamma)*((1-z)**(delta))*((gamma+delta)**(gamma+delta))/((gamma**gamma)*(delta**delta))
    if((str(had)=="pi+")&(flavor==2)):
        MCv = 0.49
    elif((str(had)=="pi+")&(flavor==-1)):
        MCv = 0.49
    elif((str(had)=="pi-")&(flavor==1)):
        MCv = 0.49
    elif((str(had)=="pi-")&(flavor==-2)):
        MCv = 0.49
    else:
        MCv = -1
    tempNCq=MCv*temp_zfactor
    return tempNCq


mcval = 1.224

def pp2avg(z):
    return 0.2 + 0.5*(z**2)



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
    temp2 = (2*eCharg*pht**pht)/(QQ)
    temp3 = (z*z*kperp2Avg*kperp2Avg)/(phT2Avg(pperp2Avg,kperp2Avg,z)**2)
    tempfinal = temp1*temp2*(temp3**2)
    return tempfinal

def A0_BM(y,z,QQ,pht,m1,mc,kperp2Avg,pperp2Avg,eCharg):
    temp = A0_cosphi_BM(y,z,pht,m1,mc,QQ,kperp2Avg,pperp2Avg,eCharg) - np.sqrt(QQ)*A0_cos2phi_BM(y,z,pht,m1,mc,QQ,kperp2Avg,pperp2Avg,eCharg)
    return temp


def A0_Cahn(y,z,QQ,pht,m1,mc,kperp2Avg,pperp2Avg,eCharg):
    temp = A0_cosphi_Cahn(y,z,pht,QQ,kperp2Avg,pperp2Avg,eCharg) - np.sqrt(QQ)*A0_cos2phi_Cahn(y,z,pht,QQ,kperp2Avg,pperp2Avg,eCharg)
    return temp


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
    #pperp2Avg=Pp2A
    eCharg=ee
    #PDFfile=Determine_PDFs_FFs(SIDISdatafilename,hadron)[0]
    if(hadron=='pi+'):
        FFfile=FF_HERMES13p_PiP
    elif(hadron=='pi-'):
        FFfile=FF_HERMES13p_PiM
    elif(hadron=='pi0'):
        FFfile=FF_HERMES13p_Pi0                      
    elif(hadron=='k+'):
        FFfile=FF_HERMES13p_KP                      
    elif(hadron=='k-'):
        FFfile=FF_HERMES13p_KM   
    #FFfile=Determine_PDFs_FFs(SIDISdatafilename,hadron)[1]
    tempvals_all=PDFfile
    #tempvals=tempvals_all[(tempvals_all["hadron"]==hadron)]
    tempvals=tempvals_all[(tempvals_all["hadron"]==hadron)&(tempvals_all["1D_dependence"]==dep)]
    QQ=tempvals['QQ']
    x=tempvals['x']
    y=tempvals['y']
    z=tempvals['z']
    phT=tempvals['phT']
    uCont1= NNq(x,Nu,alphau,betau)*(eU**2)*SIDIS_xFxQ2(PDFfile,2,hadron,dep)*NCq(hadron,2,z)*SIDIS_zFzQ(FFfile,2,hadron,dep)
    ubarCont1= NNqbar(x,Nubar,alphaub,betaub)*(eUbar**2)*SIDIS_xFxQ2(PDFfile,-2,hadron,dep)*NCq(hadron,-2,z)*SIDIS_zFzQ(FFfile,-2,hadron,dep)
    dCont1= NNq(x,Nd,alphad,betad)*(eD**2)*SIDIS_xFxQ2(PDFfile,1,hadron,dep)*NCq(hadron,1,z)*SIDIS_zFzQ(FFfile,1,hadron,dep)
    dbarCont1= NNqbar(x,Ndbar,alphadb,betadb)*(eDbar**2)*SIDIS_xFxQ2(PDFfile,-1,hadron,dep)*NCq(hadron,-1,z)*SIDIS_zFzQ(FFfile,-1,hadron,dep)
    sCont1= NNq(x,Ns,alphas,betas)*(eS**2)*SIDIS_xFxQ2(PDFfile,3,hadron,dep)*NCq(hadron,3,z)*SIDIS_zFzQ(FFfile,3,hadron,dep)
    sbarCont1= NNqbar(x,Nsbar,alphasb,betasb)*(eSbar**2)*SIDIS_xFxQ2(PDFfile,-3,hadron,dep)*NCq(hadron,-3,z)*SIDIS_zFzQ(FFfile,-3,hadron,dep)
    uCont2= (eU**2)*SIDIS_xFxQ2(PDFfile,2,hadron,dep)*NCq(hadron,2,z)*SIDIS_zFzQ(FFfile,2,hadron,dep)
    ubarCont2= (eUbar**2)*SIDIS_xFxQ2(PDFfile,-2,hadron,dep)*NCq(hadron,-2,z)*SIDIS_zFzQ(FFfile,-2,hadron,dep)
    dCont2= (eD**2)*SIDIS_xFxQ2(PDFfile,1,hadron,dep)*NCq(hadron,1,z)*SIDIS_zFzQ(FFfile,1,hadron,dep)
    dbarCont2=(eDbar**2)*SIDIS_xFxQ2(PDFfile,-1,hadron,dep)*NCq(hadron,-1,z)*SIDIS_zFzQ(FFfile,-1,hadron,dep)
    sCont2= (eS**2)*SIDIS_xFxQ2(PDFfile,3,hadron,dep)*NCq(hadron,3,z)*SIDIS_zFzQ(FFfile,3,hadron,dep)
    sbarCont2= (eSbar**2)*SIDIS_xFxQ2(PDFfile,-3,hadron,dep)*NCq(hadron,-3,z)*SIDIS_zFzQ(FFfile,-3,hadron,dep)
    tempNumerator = uCont1 + ubarCont1 +dCont1 + dbarCont1 + sCont1 + sbarCont1
    tempDenominator = uCont2 + ubarCont2 +dCont2 + dbarCont2 + sCont2 + sbarCont2
    ppavgval=pp2avg(z)
    tempASiv_Hadron = A0_cosphi_BM(y,z,phT,m1,mcval,QQ,kperp2Avg,ppavgval,eCharg)*(tempNumerator/tempDenominator)
    return tempASiv_Hadron


def Asymmetry_cosphi_Cahn(SIDISdatafilename,hadron,dep):
    kperp2Avg=Kp2A
    eCharg=ee
    tempvals_all=pd.read_csv(SIDISdatafilename)
    tempvals=tempvals_all[(tempvals_all["hadron"]==hadron)&(tempvals_all["1D_dependence"]==dep)]
    QQ=tempvals['Q2']
    x=tempvals['x']
    y=tempvals['y']
    z=tempvals['z']
    phT=tempvals['phT']
    Asym_vals=tempvals['Asym']
    Asym_err =tempvals['dAsym']
    ppavgval=pp2avg(z)
    tempASiv_Hadron = A0_cosphi_Cahn(y,z,phT,QQ,kperp2Avg,ppavgval,eCharg)
    return tempASiv_Hadron,Asym_vals,Asym_err

### Here we multiply the cos2phi term output with Q  
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
    #pperp2Avg=Pp2A
    eCharg=ee
    #PDFfile=Determine_PDFs_FFs(SIDISdatafilename,hadron)[0]
    #FFfile=Determine_PDFs_FFs(SIDISdatafilename,hadron)[1]
    if(hadron=='pi+'):
        FFfile=FF_HERMES13p_PiP
    elif(hadron=='pi-'):
        FFfile=FF_HERMES13p_PiM
    elif(hadron=='pi0'):
        FFfile=FF_HERMES13p_Pi0                      
    elif(hadron=='k+'):
        FFfile=FF_HERMES13p_KP                      
    elif(hadron=='k-'):
        FFfile=FF_HERMES13p_KM   
    tempvals_all=PDFfile
    #tempvals=tempvals_all[(tempvals_all["hadron"]==hadron)]
    tempvals=tempvals_all[(tempvals_all["hadron"]==hadron)&(tempvals_all["1D_dependence"]==dep)]
    QQ=tempvals['QQ']
    x=tempvals['x']
    y=tempvals['y']
    z=tempvals['z']
    phT=tempvals['phT']
    uCont1= NNq(x,Nu,alphau,betau)*(eU**2)*SIDIS_xFxQ2(PDFfile,2,hadron,dep)*NCq(hadron,2,z)*SIDIS_zFzQ(FFfile,2,hadron,dep)
    ubarCont1= NNqbar(x,Nubar,alphaub,betaub)*(eUbar**2)*SIDIS_xFxQ2(PDFfile,-2,hadron,dep)*NCq(hadron,-2,z)*SIDIS_zFzQ(FFfile,-2,hadron,dep)
    dCont1= NNq(x,Nd,alphad,betad)*(eD**2)*SIDIS_xFxQ2(PDFfile,1,hadron,dep)*NCq(hadron,1,z)*SIDIS_zFzQ(FFfile,1,hadron,dep)
    dbarCont1= NNqbar(x,Ndbar,alphadb,betadb)*(eDbar**2)*SIDIS_xFxQ2(PDFfile,-1,hadron,dep)*NCq(hadron,-1,z)*SIDIS_zFzQ(FFfile,-1,hadron,dep)
    sCont1= NNq(x,Ns,alphas,betas)*(eS**2)*SIDIS_xFxQ2(PDFfile,3,hadron,dep)*NCq(hadron,3,z)*SIDIS_zFzQ(FFfile,3,hadron,dep)
    sbarCont1= NNqbar(x,Nsbar,alphasb,betasb)*(eSbar**2)*SIDIS_xFxQ2(PDFfile,-3,hadron,dep)*NCq(hadron,-3,z)*SIDIS_zFzQ(FFfile,-3,hadron,dep)
    uCont2= (eU**2)*SIDIS_xFxQ2(PDFfile,2,hadron,dep)*NCq(hadron,2,z)*SIDIS_zFzQ(FFfile,2,hadron,dep)
    ubarCont2= (eUbar**2)*SIDIS_xFxQ2(PDFfile,-2,hadron,dep)*NCq(hadron,-2,z)*SIDIS_zFzQ(FFfile,-2,hadron,dep)
    dCont2= (eD**2)*SIDIS_xFxQ2(PDFfile,1,hadron,dep)*NCq(hadron,1,z)*SIDIS_zFzQ(FFfile,1,hadron,dep)
    dbarCont2=(eDbar**2)*SIDIS_xFxQ2(PDFfile,-1,hadron,dep)*NCq(hadron,-1,z)*SIDIS_zFzQ(FFfile,-1,hadron,dep)
    sCont2= (eS**2)*SIDIS_xFxQ2(PDFfile,3,hadron,dep)*NCq(hadron,3,z)*SIDIS_zFzQ(FFfile,3,hadron,dep)
    sbarCont2= (eSbar**2)*SIDIS_xFxQ2(PDFfile,-3,hadron,dep)*NCq(hadron,-3,z)*SIDIS_zFzQ(FFfile,-3,hadron,dep)
    tempNumerator = uCont1 + ubarCont1 +dCont1 + dbarCont1 + sCont1 + sbarCont1
    tempDenominator = uCont2 + ubarCont2 +dCont2 + dbarCont2 + sCont2 + sbarCont2
    ppavgval=pp2avg(z)
    tempASiv_Hadron = np.sqrt(QQ)*A0_cos2phi_BM(y,z,phT,m1,mcval,QQ,kperp2Avg,ppavgval,eCharg)*(tempNumerator/tempDenominator)
    return tempASiv_Hadron
    

### Here we multiply the cos2phi term output with Q    
def Asymmetry_cos2phi_Cahn(SIDISdatafilename,hadron,dep):
    kperp2Avg=Kp2A
    #pperp2Avg=Pp2A
    eCharg=ee
    tempvals_all=pd.read_csv(SIDISdatafilename)
    tempvals=tempvals_all[(tempvals_all["hadron"]==hadron)&(tempvals_all["1D_dependence"]==dep)]
    QQ=tempvals['Q2']
    x=tempvals['x']
    y=tempvals['y']
    z=tempvals['z']
    phT=tempvals['phT']
    Asym_vals=np.sqrt(QQ)*tempvals['Asym']
    Asym_err =tempvals['dAsym']
    ppavgval=pp2avg(z)
    tempASiv_Hadron = np.sqrt(QQ)*A0_cos2phi_Cahn(y,z,phT,QQ,kperp2Avg,ppavgval,eCharg)
    return tempASiv_Hadron,Asym_vals,Asym_err


def Asymmetry_cosphi(SIDISdatafilename,hadron,dep,**parms):
    temp_theory= np.array(Asymmetry_cosphi_BM(SIDISdatafilename,hadron,dep,**parms)) + np.array(Asymmetry_cosphi_Cahn(SIDISdatafilename,hadron,dep)[0])
    temp_data=Asymmetry_cosphi_Cahn(SIDISdatafilename,hadron,dep)[1]
    temp_err=Asymmetry_cosphi_Cahn(SIDISdatafilename,hadron,dep)[2]
    return temp_theory, temp_data, temp_err


def Asymmetry_cos2phi(SIDISdatafilename,hadron,dep,**parms):
    temp_theory = np.array(Asymmetry_cos2phi_BM(SIDISdatafilename,hadron,dep,**parms)) + np.array(Asymmetry_cos2phi_Cahn(SIDISdatafilename,hadron,dep)[0])
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
            tempfity=Asymmetry_cosphi(datfile,"pi+","y",**parms)[0]
            tempfitz=Asymmetry_cosphi(datfile,"pi+","z",**parms)[0]
            tempfitphT=Asymmetry_cosphi(datfile,"pi+","phT",**parms)[0]
            tempfit=np.concatenate((tempfitx,tempfity,tempfitz,tempfitphT))
            fittot.append(tempfit)
            tempdatx=Asymmetry_cosphi(datfile,"pi+","x",**parms)[1]
            tempdaty=Asymmetry_cosphi(datfile,"pi+","y",**parms)[1]
            tempdatz=Asymmetry_cosphi(datfile,"pi+","z",**parms)[1]
            tempdatphT=Asymmetry_cosphi(datfile,"pi+","phT",**parms)[1]
            tempdat=np.concatenate((tempdatx,tempdaty,tempdatz,tempdatphT))
            datvals.append(tempdat)
            temperrx=Asymmetry_cosphi(datfile,"pi+","x",**parms)[2]
            temperry=Asymmetry_cosphi(datfile,"pi+","y",**parms)[2]
            temperrz=Asymmetry_cosphi(datfile,"pi+","z",**parms)[2]
            temperrphT=Asymmetry_cosphi(datfile,"pi+","phT",**parms)[2]
            temperr=np.concatenate((temperrx,temperry,temperrz,temperrphT))
            errvals.append(temperr)
        elif temHads[i]=="pi-":
            tempfitx=Asymmetry_cosphi(datfile,"pi-","x",**parms)[0]
            tempfity=Asymmetry_cosphi(datfile,"pi-","y",**parms)[0]
            tempfitz=Asymmetry_cosphi(datfile,"pi-","z",**parms)[0]
            tempfitphT=Asymmetry_cosphi(datfile,"pi-","phT",**parms)[0]
            tempfit=np.concatenate((tempfitx,tempfity,tempfitz,tempfitphT))
            fittot.append(tempfit)
            tempdatx=Asymmetry_cosphi(datfile,"pi-","x",**parms)[1]
            tempdaty=Asymmetry_cosphi(datfile,"pi-","y",**parms)[1]
            tempdatz=Asymmetry_cosphi(datfile,"pi-","z",**parms)[1]
            tempdatphT=Asymmetry_cosphi(datfile,"pi-","phT",**parms)[1]
            tempdat=np.concatenate((tempdatx,tempdaty,tempdatz,tempdatphT))
            datvals.append(tempdat)
            temperrx=Asymmetry_cosphi(datfile,"pi-","x",**parms)[2]
            temperry=Asymmetry_cosphi(datfile,"pi-","y",**parms)[2]
            temperrz=Asymmetry_cosphi(datfile,"pi-","z",**parms)[2]
            temperrphT=Asymmetry_cosphi(datfile,"pi-","phT",**parms)[2]
            temperr=np.concatenate((temperrx,temperry,temperrz,temperrphT))
            errvals.append(temperr)
        elif temHads[i]=="pi0":
            tempfitx=Asymmetry_cosphi(datfile,"pi0","x",**parms)[0]
            tempfity=Asymmetry_cosphi(datfile,"pi0","y",**parms)[0]
            tempfitz=Asymmetry_cosphi(datfile,"pi0","z",**parms)[0]
            tempfitphT=Asymmetry_cosphi(datfile,"pi0","phT",**parms)[0]
            tempfit=np.concatenate((tempfitx,tempfity,tempfitz,tempfitphT))
            fittot.append(tempfit)
            tempdatx=Asymmetry_cosphi(datfile,"pi0","x",**parms)[1]
            tempdaty=Asymmetry_cosphi(datfile,"pi0","y",**parms)[1]
            tempdatz=Asymmetry_cosphi(datfile,"pi0","z",**parms)[1]
            tempdatphT=Asymmetry_cosphi(datfile,"pi0","phT",**parms)[1]
            tempdat=np.concatenate((tempdatx,tempdaty,tempdatz,tempdatphT))
            datvals.append(tempdat)
            temperrx=Asymmetry_cosphi(datfile,"pi0","x",**parms)[2]
            temperry=Asymmetry_cosphi(datfile,"pi0","y",**parms)[2]
            temperrz=Asymmetry_cosphi(datfile,"pi0","z",**parms)[2]
            temperrphT=Asymmetry_cosphi(datfile,"pi0","phT",**parms)[2]
            temperr=np.concatenate((temperrx,temperry,temperrz,temperrphT))
            errvals.append(temperr)
        elif temHads[i]=="k+":
            tempfitx=Asymmetry_cosphi(datfile,"k+","x",**parms)[0]
            tempfity=Asymmetry_cosphi(datfile,"k+","y",**parms)[0]
            tempfitz=Asymmetry_cosphi(datfile,"k+","z",**parms)[0]
            tempfitphT=Asymmetry_cosphi(datfile,"k+","phT",**parms)[0]
            tempfit=np.concatenate((tempfitx,tempfity,tempfitz,tempfitphT))
            fittot.append(tempfit)
            tempdatx=Asymmetry_cosphi(datfile,"k+","x",**parms)[1]
            tempdaty=Asymmetry_cosphi(datfile,"k+","y",**parms)[1]
            tempdatz=Asymmetry_cosphi(datfile,"k+","z",**parms)[1]
            tempdatphT=Asymmetry_cosphi(datfile,"k+","phT",**parms)[1]
            tempdat=np.concatenate((tempdatx,tempdaty,tempdatz,tempdatphT))
            datvals.append(tempdat)
            temperrx=Asymmetry_cosphi(datfile,"k+","x",**parms)[2]
            temperry=Asymmetry_cosphi(datfile,"k+","y",**parms)[2]
            temperrz=Asymmetry_cosphi(datfile,"k+","z",**parms)[2]
            temperrphT=Asymmetry_cosphi(datfile,"k+","phT",**parms)[2]
            temperr=np.concatenate((temperrx,temperry,temperrz,temperrphT))
            errvals.append(temperr)
        elif temHads[i]=="k-":
            tempfitx=Asymmetry_cosphi(datfile,"k-","x",**parms)[0]
            tempfity=Asymmetry_cosphi(datfile,"k-","y",**parms)[0]
            tempfitz=Asymmetry_cosphi(datfile,"k-","z",**parms)[0]
            tempfitphT=Asymmetry_cosphi(datfile,"k-","phT",**parms)[0]
            tempfit=np.concatenate((tempfitx,tempfity,tempfitz,tempfitphT))
            fittot.append(tempfit)
            tempdatx=Asymmetry_cosphi(datfile,"k-","x",**parms)[1]
            tempdaty=Asymmetry_cosphi(datfile,"k-","y",**parms)[1]
            tempdatz=Asymmetry_cosphi(datfile,"k-","z",**parms)[1]
            tempdatphT=Asymmetry_cosphi(datfile,"k-","phT",**parms)[1]
            tempdat=np.concatenate((tempdatx,tempdaty,tempdatz,tempdatphT))
            datvals.append(tempdat)
            temperrx=Asymmetry_cosphi(datfile,"k-","x",**parms)[2]
            temperry=Asymmetry_cosphi(datfile,"k-","y",**parms)[2]
            temperrz=Asymmetry_cosphi(datfile,"k-","z",**parms)[2]
            temperrphT=Asymmetry_cosphi(datfile,"k-","phT",**parms)[2]
            temperr=np.concatenate((temperrx,temperry,temperrz,temperrphT))
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
            tempfity=Asymmetry_cos2phi(datfile,"pi+","y",**parms)[0]
            tempfitz=Asymmetry_cos2phi(datfile,"pi+","z",**parms)[0]
            tempfitphT=Asymmetry_cos2phi(datfile,"pi+","phT",**parms)[0]
            tempfit=np.concatenate((tempfitx,tempfity,tempfitz,tempfitphT))
            fittot.append(tempfit)
            tempdatx=Asymmetry_cos2phi(datfile,"pi+","x",**parms)[1]
            tempdaty=Asymmetry_cos2phi(datfile,"pi+","y",**parms)[1]
            tempdatz=Asymmetry_cos2phi(datfile,"pi+","z",**parms)[1]
            tempdatphT=Asymmetry_cos2phi(datfile,"pi+","phT",**parms)[1]
            tempdat=np.concatenate((tempdatx,tempdaty,tempdatz,tempdatphT))
            datvals.append(tempdat)
            temperrx=Asymmetry_cos2phi(datfile,"pi+","x",**parms)[2]
            temperry=Asymmetry_cos2phi(datfile,"pi+","y",**parms)[2]
            temperrz=Asymmetry_cos2phi(datfile,"pi+","z",**parms)[2]
            temperrphT=Asymmetry_cos2phi(datfile,"pi+","phT",**parms)[2]
            temperr=np.concatenate((temperrx,temperry,temperrz,temperrphT))
            errvals.append(temperr)
        elif temHads[i]=="pi-":
            tempfitx=Asymmetry_cos2phi(datfile,"pi-","x",**parms)[0]
            tempfity=Asymmetry_cos2phi(datfile,"pi-","y",**parms)[0]
            tempfitz=Asymmetry_cos2phi(datfile,"pi-","z",**parms)[0]
            tempfitphT=Asymmetry_cos2phi(datfile,"pi-","phT",**parms)[0]
            tempfit=np.concatenate((tempfitx,tempfity,tempfitz,tempfitphT))
            fittot.append(tempfit)
            tempdatx=Asymmetry_cos2phi(datfile,"pi-","x",**parms)[1]
            tempdaty=Asymmetry_cos2phi(datfile,"pi-","y",**parms)[1]
            tempdatz=Asymmetry_cos2phi(datfile,"pi-","z",**parms)[1]
            tempdatphT=Asymmetry_cos2phi(datfile,"pi-","phT",**parms)[1]
            tempdat=np.concatenate((tempdatx,tempdaty,tempdatz,tempdatphT))
            datvals.append(tempdat)
            temperrx=Asymmetry_cos2phi(datfile,"pi-","x",**parms)[2]
            temperry=Asymmetry_cos2phi(datfile,"pi-","y",**parms)[2]
            temperrz=Asymmetry_cos2phi(datfile,"pi-","z",**parms)[2]
            temperrphT=Asymmetry_cos2phi(datfile,"pi-","phT",**parms)[2]
            temperr=np.concatenate((temperrx,temperry,temperrz,temperrphT))
            errvals.append(temperr)
        elif temHads[i]=="pi0":
            tempfitx=Asymmetry_cos2phi(datfile,"pi0","x",**parms)[0]
            tempfity=Asymmetry_cos2phi(datfile,"pi0","y",**parms)[0]
            tempfitz=Asymmetry_cos2phi(datfile,"pi0","z",**parms)[0]
            tempfitphT=Asymmetry_cos2phi(datfile,"pi0","phT",**parms)[0]
            tempfit=np.concatenate((tempfitx,tempfity,tempfitz,tempfitphT))
            fittot.append(tempfit)
            tempdatx=Asymmetry_cos2phi(datfile,"pi0","x",**parms)[1]
            tempdaty=Asymmetry_cos2phi(datfile,"pi0","y",**parms)[1]
            tempdatz=Asymmetry_cos2phi(datfile,"pi0","z",**parms)[1]
            tempdatphT=Asymmetry_cos2phi(datfile,"pi0","phT",**parms)[1]
            tempdat=np.concatenate((tempdatx,tempdaty,tempdatz,tempdatphT))
            datvals.append(tempdat)
            temperrx=Asymmetry_cos2phi(datfile,"pi0","x",**parms)[2]
            temperry=Asymmetry_cos2phi(datfile,"pi0","y",**parms)[2]
            temperrz=Asymmetry_cos2phi(datfile,"pi0","z",**parms)[2]
            temperrphT=Asymmetry_cos2phi(datfile,"pi0","phT",**parms)[2]
            temperr=np.concatenate((temperrx,temperry,temperrz,temperrphT))
            errvals.append(temperr)
        elif temHads[i]=="k+":
            tempfitx=Asymmetry_cos2phi(datfile,"k+","x",**parms)[0]
            tempfity=Asymmetry_cos2phi(datfile,"k+","y",**parms)[0]
            tempfitz=Asymmetry_cos2phi(datfile,"k+","z",**parms)[0]
            tempfitphT=Asymmetry_cos2phi(datfile,"k+","phT",**parms)[0]
            tempfit=np.concatenate((tempfitx,tempfity,tempfitz,tempfitphT))
            fittot.append(tempfit)
            tempdatx=Asymmetry_cos2phi(datfile,"k+","x",**parms)[1]
            tempdaty=Asymmetry_cos2phi(datfile,"k+","y",**parms)[1]
            tempdatz=Asymmetry_cos2phi(datfile,"k+","z",**parms)[1]
            tempdatphT=Asymmetry_cos2phi(datfile,"k+","phT",**parms)[1]
            tempdat=np.concatenate((tempdatx,tempdaty,tempdatz,tempdatphT))
            datvals.append(tempdat)
            temperrx=Asymmetry_cos2phi(datfile,"k+","x",**parms)[2]
            temperry=Asymmetry_cos2phi(datfile,"k+","y",**parms)[2]
            temperrz=Asymmetry_cos2phi(datfile,"k+","z",**parms)[2]
            temperrphT=Asymmetry_cos2phi(datfile,"k+","phT",**parms)[2]
            temperr=np.concatenate((temperrx,temperry,temperrz,temperrphT))
            errvals.append(temperr)
        elif temHads[i]=="k-":
            tempfitx=Asymmetry_cos2phi(datfile,"k-","x",**parms)[0]
            tempfity=Asymmetry_cos2phi(datfile,"k-","y",**parms)[0]
            tempfitz=Asymmetry_cos2phi(datfile,"k-","z",**parms)[0]
            tempfitphT=Asymmetry_cos2phi(datfile,"k-","phT",**parms)[0]
            tempfit=np.concatenate((tempfitx,tempfity,tempfitz,tempfitphT))
            fittot.append(tempfit)
            tempdatx=Asymmetry_cos2phi(datfile,"k-","x",**parms)[1]
            tempdaty=Asymmetry_cos2phi(datfile,"k-","y",**parms)[1]
            tempdatz=Asymmetry_cos2phi(datfile,"k-","z",**parms)[1]
            tempdatphT=Asymmetry_cos2phi(datfile,"k-","phT",**parms)[1]
            tempdat=np.concatenate((tempdatx,tempdaty,tempdatz,tempdatphT))
            datvals.append(tempdat)
            temperrx=Asymmetry_cos2phi(datfile,"k-","x",**parms)[2]
            temperry=Asymmetry_cos2phi(datfile,"k-","y",**parms)[2]
            temperrz=Asymmetry_cos2phi(datfile,"k-","z",**parms)[2]
            temperrphT=Asymmetry_cos2phi(datfile,"k-","phT",**parms)[2]
            temperr=np.concatenate((temperrx,temperry,temperrz,temperrphT))
            errvals.append(temperr)
    return np.concatenate((fittot), axis=None),np.concatenate((datvals), axis=None),np.concatenate((errvals), axis=None)


def totalfitDataSet(datfile,**parms):
    temp_theory = totalfitDataSet_cosphi(datfile,m1=m1v,Nu=Nuv,au=auv,bu=buv,Nub=Nubv,aub=aubv,bub=bubv,
    Nd=Ndv,ad=adv,bd=bdv,Ndb=Ndbv,adb=adbv,bdb=bdbv,
    Ns=Nsv,aS=asv,bS=bsv,Nsb=Nsbv,aSb=asbv,bSb=bsbv)[0] - totalfitDataSet_cos2phi(datfile,m1=m1v,Nu=Nuv,au=auv,bu=buv,Nub=Nubv,aub=aubv,bub=bubv,
    Nd=Ndv,ad=adv,bd=bdv,Ndb=Ndbv,adb=adbv,bdb=bdbv,
    Ns=Nsv,aS=asv,bS=bsv,Nsb=Nsbv,aSb=asbv,bSb=bsbv)[0]
    temp_data = totalfitDataSet_cosphi(datfile,m1=m1v,Nu=Nuv,au=auv,bu=buv,Nub=Nubv,aub=aubv,bub=bubv,
    Nd=Ndv,ad=adv,bd=bdv,Ndb=Ndbv,adb=adbv,bdb=bdbv,
    Ns=Nsv,aS=asv,bS=bsv,Nsb=Nsbv,aSb=asbv,bSb=bsbv)[1] - totalfitDataSet_cos2phi(datfile,m1=m1v,Nu=Nuv,au=auv,bu=buv,Nub=Nubv,aub=aubv,bub=bubv,
    Nd=Ndv,ad=adv,bd=bdv,Ndb=Ndbv,adb=adbv,bdb=bdbv,
    Ns=Nsv,aS=asv,bS=bsv,Nsb=Nsbv,aSb=asbv,bSb=bsbv)[1]
    temp_err = totalfitDataSet_cosphi(datfile,m1=m1v,Nu=Nuv,au=auv,bu=buv,Nub=Nubv,aub=aubv,bub=bubv,
    Nd=Ndv,ad=adv,bd=bdv,Ndb=Ndbv,adb=adbv,bdb=bdbv,
    Ns=Nsv,aS=asv,bS=bsv,Nsb=Nsbv,aSb=asbv,bSb=bsbv)[2]
    return temp_theory, temp_data, temp_err


par_array = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

m1v=par_array[0]
Nuv=par_array[1]
auv=par_array[2]
buv=par_array[3]
Nubv=par_array[4]
aubv=par_array[5]
bubv=par_array[6]
Ndv=par_array[7]
adv=par_array[8]
bdv=par_array[9]
Ndbv=par_array[10]
adbv=par_array[11]
bdbv=par_array[12]
Nsv=par_array[13]
asv=par_array[14]
bsv=par_array[15]
Nsbv=par_array[16]
asbv=par_array[17]
bsbv=par_array[18]


def totalchi2Minuit(m1,Nu,au,bu,Nub,aub,bub,Nd,ad,bd,Ndb,adb,bdb,Ns,aS,bS,Nsb,aSb,bSb):
    th_vals=np.array(totalfitDataSet(SIDIS_DataFilesArrayR[0],m1=m1,Nu=Nu,au=au,bu=bu,Nub=Nub,aub=aub,bub=bub,
        Nd=Nd,ad=ad,bd=bd,Ndb=Ndb,adb=adb,bdb=bdb,Ns=Ns,aS=aS,bS=bS,Nsb=Nsb,aSb=aSb,bSb=bSb)[0])
    dat_vals=np.array(totalfitDataSet(SIDIS_DataFilesArrayR[0],m1=m1,Nu=Nu,au=au,bu=bu,Nub=Nub,aub=aub,bub=bub,
        Nd=Nd,ad=ad,bd=bd,Ndb=Ndb,adb=adb,bdb=bdb,Ns=Ns,aS=aS,bS=bS,Nsb=Nsb,aSb=aSb,bSb=bSb)[1])
    err_vals=np.array(totalfitDataSet(SIDIS_DataFilesArrayR[0],m1=m1,Nu=Nu,au=au,bu=bu,Nub=Nub,aub=aub,bub=bub,
        Nd=Nd,ad=ad,bd=bd,Ndb=Ndb,adb=adb,bdb=bdb,Ns=Ns,aS=aS,bS=bS,Nsb=Nsb,aSb=aSb,bSb=bSb)[2])
    Chi2=np.sum(((dat_vals-th_vals)/err_vals)**2)
    return Chi2


par_name_array=('m1','Nu','alphau','betau','Nubar','alphaub','betaub','Nd','alphad','betad','Ndbar','alphadb','betadb','Ns','alphas','betas','Nsbar','alphasb','betasb')




from datetime import datetime
start = datetime.now()
print("start =", start)

def generate_file(n_array):
    ms = Minuit(totalchi2Minuit,m1=m1v,Nu=Nuv,au=auv,bu=buv,Nub=Nubv,aub=aubv,bub=bubv,
    Nd=Ndv,ad=adv,bd=bdv,Ndb=Ndbv,adb=adbv,bdb=bdbv,
    Ns=Nsv,aS=asv,bS=bsv,Nsb=Nsbv,aSb=asbv,bSb=bsbv,
    limit_m1=(0.3,0.4),limit_Nu=(-50,50),limit_Nub=(-50,50),limit_Nd = (-50,50),limit_Ndb = (-50,50),limit_Ns=(-50,50),limit_Nsb = (-50,50),
    limit_au=(0,50),limit_bu=(0,50),limit_aub=(0, 50),limit_bub=(0,50),
    limit_ad=(0,50),limit_bd=(0,50),limit_adb=(0, 50),limit_bdb=(0,50),
    limit_aS=(0,50),limit_bS=(0,50),limit_aSb=(0, 50),limit_bSb=(0,50),
    errordef=1)
    ms.migrad()
    temp_df=pd.DataFrame({'parameter':[],'value':[],'error':[],'chi2':[]})
    temp_val=[]
    temp_err=[]
    for i in range(0,len(n_array)):
        temp_val.append(ms.values[i])
        temp_err.append(ms.errors[i])
    temp_df['parameter'] = n_array
    temp_df['value'] = temp_val
    temp_df['error'] = temp_err
    temp_df['chi2'] = ms.fval
    #temp_df['N_data'] = Total_data_points
    #return temp_df
    finish = datetime.now()
    print("finish =", finish)
    return temp_df.to_csv('Fit_Results.csv')

print(generate_file(par_name_array))
done = datetime.now()
print("done =", done)



# from datetime import datetime
# start = datetime.now()
# print("start =", start)

# def generate_file(n_array):
#     ms = Minuit(totalchi2Minuit,m1=m1v,Nu=Nuv,au=auv,bu=buv,Nub=Nubv,aub=aubv,bub=bubv,
#     Nd=Ndv,ad=adv,bd=bdv,Ndb=Ndbv,adb=adbv,bdb=bdbv,
#     Ns=Nsv,aS=asv,bS=bsv,Nsb=Nsbv,aSb=asbv,bSb=bsbv,
#     limit_m1=(3,7),limit_Ns=(-20,20), limit_Nsb = (-20,20),
#     limit_au=(0,20),limit_bu=(0,20),limit_aub=(0, 20),limit_bub=(0,20),
#     limit_ad=(0,20),limit_bd=(0,20),limit_adb=(0, 20),limit_bdb=(0,20),
#     limit_aS=(0,20),limit_bS=(0,20),limit_aSb=(0, 20),limit_bSb=(0,20),
#     errordef=1)
#     ms.migrad()
#     temp_df=pd.DataFrame({'parameter':[],'value':[],'error':[],'chi2':[],'N_data':[]})
#     temp_val=[]
#     temp_err=[]
#     for i in range(0,len(n_array)):
#         temp_val.append(ms.values[i])
#         temp_err.append(ms.errors[i])
#     temp_df['parameter'] = n_array
#     temp_df['value'] = temp_val
#     temp_df['error'] = temp_err
#     temp_df['chi2'] = ms.fval
#     temp_df['N_data'] = Total_data_points
#     #return temp_df
#     finish = datetime.now()
#     print("finish =", finish)
#     return temp_df.to_csv('Fit_Results.csv')

# print(generate_file(par_name_array))
# done = datetime.now()
# print("done =", done)



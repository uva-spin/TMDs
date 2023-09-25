import lhapdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Input_Parameterization import *
#from Sivers_SIDIS_Chi2_3D import *
#from Sivers_SIDIS_Chi2_R import *
#from PathsX import *
from Sivers_SIDIS_Definitions_R import *
from PathsR import *
from Constants import *

from iminuit import Minuit
import numpy as np

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
    temp2 = (-2*eCharg*pht)/(np.sqrt(QQ))
    temp3 = (z*kperp2Avg)/(phT2Avg(pperp2Avg,kperp2Avg,z))
    tempfinal = temp1*temp2*(temp3**2)
    return tempfinal




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
    PDFfile=Determine_PDFs_FFs(SIDISdatafilename,hadron)[0]
    FFfile=Determine_PDFs_FFs(SIDISdatafilename,hadron)[1]
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
    PDFfile=Determine_PDFs_FFs(SIDISdatafilename,hadron)[0]
    FFfile=Determine_PDFs_FFs(SIDISdatafilename,hadron)[1]
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
    tempASiv_Hadron = A0_cos2phi_BM(y,z,phT,m1,mcval,QQ,kperp2Avg,ppavgval,eCharg)*(tempNumerator/tempDenominator)
    return tempASiv_Hadron
    

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
    Asym_vals=tempvals['Asym']
    Asym_err =tempvals['dAsym']
    ppavgval=pp2avg(z)
    tempASiv_Hadron = A0_cos2phi_Cahn(y,z,phT,QQ,kperp2Avg,ppavgval,eCharg)
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



def Create_Asym_Data(datafile,m1,Nu,au,bu,Nub,aub,bub,Nd,ad,bd,Ndb,adb,bdb,Ns,aS,bS,Nsb,aSb,bSb):
    tempdf=pd.read_csv(datafile)
    temphad=np.array(tempdf['hadron'],dtype=object)
    tempQ2=np.array(tempdf['Q2'],dtype=object)
    tempX=np.array(tempdf['x'],dtype=object)
    tempY=np.array(tempdf['y'],dtype=object)
    tempZ=np.array(tempdf['z'],dtype=object)
    tempPHT=np.array(tempdf['phT'],dtype=object)
    tempSivData=np.array(tempdf['Asym'],dtype=object)
    tempSivErr=np.array(tempdf['dAsym'],dtype=object)
    tempDEP=np.array(tempdf['1D_dependence'],dtype=object)
    data_dictionary={"hadron":[],"Q2":[],"x":[],"y":[],"z":[],"phT":[],"Asym":[],"dAsym":[],"1D_dependence":[]}
    data_dictionary["hadron"]=temphad
    data_dictionary["Q2"]=tempQ2
    data_dictionary["x"]=tempX
    data_dictionary["y"]=tempY
    data_dictionary["z"]=tempZ
    data_dictionary["phT"]=tempPHT
    data_dictionary["dAsym"]=tempSivErr
    data_dictionary["1D_dependence"]=tempDEP
    ### Change this function accordingly ####
    temp_BM = totalfitDataSet_cosphi(datafile,m1=m1,Nu=Nu,au=au,bu=bu,Nub=Nub,aub=aub,bub=bub,
        Nd=Nd,ad=ad,bd=bd,Ndb=Ndb,adb=adb,bdb=bdb,Ns=Ns,aS=aS,bS=bS,Nsb=Nsb,aSb=aSb,bSb=bSb)[0]
    ############################################
    data_dictionary["Asym"]=temp_BM
    return pd.DataFrame(data_dictionary)


def Create_Asym_Rearranged_Data(datafile,m1,Nu,au,bu,Nub,aub,bub,Nd,ad,bd,Ndb,adb,bdb,Ns,aS,bS,Nsb,aSb,bSb):
    tempdf=pd.read_csv(datafile)
    temphad=np.array(tempdf['hadron'],dtype=object)
    tempQ2=np.array(tempdf['Q2'],dtype=object)
    tempQ = np.array(np.sqrt(tempdf['Q2']),dtype=object)
    tempX=np.array(tempdf['x'],dtype=object)
    tempY=np.array(tempdf['y'],dtype=object)
    tempZ=np.array(tempdf['z'],dtype=object)
    tempPHT=np.array(tempdf['phT'],dtype=object)
    tempSivData=np.array(tempdf['Asym'],dtype=object)
    tempSivErr=np.array(tempdf['dAsym'],dtype=object)
    tempDEP=np.array(tempdf['1D_dependence'],dtype=object)
    data_dictionary={"x":[],"y":[],"z":[],"phT":[],"Q2":[],"Asym":[],"dAsym":[],"hadron":[],"1D_dependence":[]}
    data_dictionary["hadron"]=temphad
    data_dictionary["Q2"]=tempQ2
    data_dictionary["x"]=tempX
    data_dictionary["y"]=tempY
    data_dictionary["z"]=tempZ
    data_dictionary["phT"]=tempPHT
    data_dictionary["dAsym"]=tempSivErr
    data_dictionary["1D_dependence"]=tempDEP
    temp_BM = totalfitDataSet_cosphi(datafile,m1=m1,Nu=Nu,au=au,bu=bu,Nub=Nub,aub=aub,bub=bub,
        Nd=Nd,ad=ad,bd=bd,Ndb=Ndb,adb=adb,bdb=bdb,Ns=Ns,aS=aS,bS=bS,Nsb=Nsb,aSb=aSb,bSb=bSb)[0]
    - tempQ*totalfitDataSet_cos2phi(datafile,m1=m1,Nu=Nu,au=au,bu=bu,Nub=Nub,aub=aub,bub=bub,
        Nd=Nd,ad=ad,bd=bd,Ndb=Ndb,adb=adb,bdb=bdb,Ns=Ns,aS=aS,bS=bS,Nsb=Nsb,aSb=aSb,bSb=bSb)[0]
    ############################################
    data_dictionary["Asym"]=temp_BM
    return pd.DataFrame(data_dictionary)


H13p=Create_Asym_Rearranged_Data(SIDIS_DataFilesArrayR[0],0.3,19.7,0.64,5,20,2.2,20,-15,5.4,18,-17,1.3,11,19.9,0,0,20,0,0)
H13p.to_csv('HERMES13p_Pseudo.csv')
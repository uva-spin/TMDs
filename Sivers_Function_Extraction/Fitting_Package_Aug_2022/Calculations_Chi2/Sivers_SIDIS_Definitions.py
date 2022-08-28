import lhapdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from Global_Constants import *


#### Introducing LHAPDFs PDFsets & FFsets

PDFdataset = lhapdf.mkPDF("cteq61")
#PDFdataset = lhapdf.mkPDF("CT10nnlo")
#FF_pion_dataset=["JAM19FF_pion_nlo"]
#FF_kaon_dataset=["JAM19FF_kaon_nlo"]

# FF_PiP_dataset=["NNFF10_PIp_nlo"]
# FF_PiM_dataset=["NNFF10_PIm_nlo"]
# FF_Pi0_dataset=["NNFF10_PIsum_nlo"]
# FF_KP_dataset=["NNFF10_KAp_nlo"]
# FF_KM_dataset=["NNFF10_KAm_nlo"]

FF_PiP_dataset=["DSS14_NLO_Pip"]
FF_PiM_dataset=["DSS14_NLO_Pim"]
FF_Pi0_dataset=["DSS14_NLO_PiSum"]
FF_KP_dataset=["DSS17_NLO_KaonPlus"]
FF_KM_dataset=["DSS17_NLO_KaonMinus"]

######################################################
########## SIDIS Asymmetry (Theory) ##################
######################################################

def hadarray(filename):
    tempdf=pd.read_csv(filename)
    temphad_data=tempdf["hadron"]
    temphad=temphad_data.dropna().unique()
    refined_had_array=[]
    for i in range(0,len(temphad)):
        if((temphad[i]=="pi+") or (temphad[i]=="pi-") or (temphad[i]=="pi0") or (temphad[i]=="k+") or (temphad[i]=="k-")):
            refined_had_array.append(temphad[i])
    return refined_had_array



def ks2Avg(m1,kperp2Avg):
    test_ks2Avg=((m1**2)*kperp2Avg)/((m1**2)+kperp2Avg)
    return test_ks2Avg

def A0(z,pht,m1,kperp2Avg,pperp2Avg,eCharg):
    tempA0part1=(((z**2)*kperp2Avg+pperp2Avg)*((ks2Avg(m1,kperp2Avg))**2))/((((z**2)*(ks2Avg(m1,kperp2Avg))+pperp2Avg)**2)*kperp2Avg)
    tempA0part21=(pht**2)*(z**2)*(ks2Avg(m1,kperp2Avg) - kperp2Avg)
    tempA0part22=((z**2)*(ks2Avg(m1,kperp2Avg))+pperp2Avg)*((z**2)*kperp2Avg+pperp2Avg)
    tempA0part2=np.exp(-tempA0part21/tempA0part22)
    tempA0part3=(np.sqrt(2*eCharg))*z*pht/m1
    tempA0=tempA0part1*tempA0part2*tempA0part3
    return tempA0

def NNq(x,Nq,aq,bq):
    tempNNq = Nq*(x**aq)*((1-x)**(bq))*((aq+bq)**(aq+bq))/((aq**aq)*(bq**bq))
    return tempNNq

def NNqbar(x,Nqbar):
    tempNNqbar = Nqbar
    return tempNNqbar

def xFxQ2(dataset,flavor,x,QQ):
    temp_parton_dist_x=np.array(dataset.xfxQ2(flavor, x, QQ),dtype=object)
    return temp_parton_dist_x

def zFzQ(dataset,flavor,zz,QQ):
    # Here "0" represents the central values from the girds
    temp_zD1=lhapdf.mkPDF(dataset[0], 0)
    zD1_vec=np.array(temp_zD1.xfxQ2(flavor,zz,QQ),dtype=object)
    return zD1_vec


def Asymmetry(QQ,x,z,pht,m1,Nu,au,bu,Nubar,Nd,ad,bd,Ndbar,Ns,aS,bs,Nsbar,lhaqID,lhaqbarID):
    kperp2Avg=Kp2A
    pperpAvg=Pp2A
    eCharg=ee
    if((lhaqID==2)and(lhaqbarID==-1)):
        ### This is pi+
        uCont1= NNq(x,Nu,au,bu)*(eU**2)*xFxQ2(PDFdataset,2,x,QQ)*zFzQ(FF_PiP_dataset,2,z,QQ)
        ubarCont1= NNqbar(x,Nubar)*(eUbar**2)*xFxQ2(PDFdataset,-2,x,QQ)*zFzQ(FF_PiP_dataset,-2,z,QQ)
        dCont1= NNq(x,Nd,ad,bd)*(eD**2)*xFxQ2(PDFdataset,1,x,QQ)*zFzQ(FF_PiP_dataset,1,z,QQ)
        dbarCont1= NNqbar(x,Ndbar)*(eDbar**2)*xFxQ2(PDFdataset,-1,x,QQ)*zFzQ(FF_PiP_dataset,-1,z,QQ)
        sCont1= NNq(x,Ns,aS,bs)*(eS**2)*xFxQ2(PDFdataset,3,x,QQ)*zFzQ(FF_PiP_dataset,3,z,QQ)
        sbarCont1= NNqbar(x,Nsbar)*(eSbar**2)*xFxQ2(PDFdataset,-3,x,QQ)*zFzQ(FF_PiP_dataset,-3,z,QQ)
        uCont2= (eU**2)*xFxQ2(PDFdataset,2,x,QQ)*zFzQ(FF_PiP_dataset,2,z,QQ)
        ubarCont2= (eUbar**2)*xFxQ2(PDFdataset,-2,x,QQ)*zFzQ(FF_PiP_dataset,-2,z,QQ)
        dCont2= (eD**2)*xFxQ2(PDFdataset,1,x,QQ)*zFzQ(FF_PiP_dataset,1,z,QQ)
        dbarCont2=(eDbar**2)*xFxQ2(PDFdataset,-1,x,QQ)*zFzQ(FF_PiP_dataset,-1,z,QQ)
        sCont2= (eS**2)*xFxQ2(PDFdataset,3,x,QQ)*zFzQ(FF_PiP_dataset,3,z,QQ)
        sbarCont2= (eSbar**2)*xFxQ2(PDFdataset,-3,x,QQ)*zFzQ(FF_PiP_dataset,-3,z,QQ)
        tempNumerator = uCont1 + ubarCont1 +dCont1 + dbarCont1 + sCont1 + sbarCont1
        tempDenominator = uCont2 + ubarCont2 +dCont2 + dbarCont2 + sCont2 + sbarCont2
        tempASiv = A0(z,pht,m1,kperp2Avg,pperpAvg,eCharg)*(tempNumerator/tempDenominator)
    elif((lhaqID==1)and(lhaqbarID==-2)):
        ### This is pi-
        uCont1= NNq(x,Nu,au,bu)*(eU**2)*xFxQ2(PDFdataset,2,x,QQ)*zFzQ(FF_PiM_dataset,2,z,QQ)
        ubarCont1= NNqbar(x,Nubar)*(eUbar**2)*xFxQ2(PDFdataset,-2,x,QQ)*zFzQ(FF_PiM_dataset,-2,z,QQ)
        dCont1= NNq(x,Nd,ad,bd)*(eD**2)*xFxQ2(PDFdataset,1,x,QQ)*zFzQ(FF_PiM_dataset,1,z,QQ)
        dbarCont1= NNqbar(x,Ndbar)*(eDbar**2)*xFxQ2(PDFdataset,-1,x,QQ)*zFzQ(FF_PiM_dataset,-1,z,QQ)
        sCont1= NNq(x,Ns,aS,bs)*(eS**2)*xFxQ2(PDFdataset,3,x,QQ)*zFzQ(FF_PiM_dataset,3,z,QQ)
        sbarCont1= NNqbar(x,Nsbar)*(eSbar**2)*xFxQ2(PDFdataset,-3,x,QQ)*zFzQ(FF_PiM_dataset,-3,z,QQ)
        uCont2= (eU**2)*xFxQ2(PDFdataset,2,x,QQ)*zFzQ(FF_PiM_dataset,2,z,QQ)
        ubarCont2= (eUbar**2)*xFxQ2(PDFdataset,-2,x,QQ)*zFzQ(FF_PiM_dataset,-2,z,QQ)
        dCont2= (eD**2)*xFxQ2(PDFdataset,1,x,QQ)*zFzQ(FF_PiM_dataset,1,z,QQ)
        dbarCont2= (eDbar**2)*xFxQ2(PDFdataset,-1,x,QQ)*zFzQ(FF_PiM_dataset,-1,z,QQ)
        sCont2= (eS**2)*xFxQ2(PDFdataset,3,x,QQ)*zFzQ(FF_PiM_dataset,3,z,QQ)
        sbarCont2= (eSbar**2)*xFxQ2(PDFdataset,-3,x,QQ)*zFzQ(FF_PiM_dataset,-3,z,QQ)
        tempNumerator = uCont1 + ubarCont1 +dCont1 + dbarCont1 + sCont1 + sbarCont1
        tempDenominator = uCont2 + ubarCont2 +dCont2 + dbarCont2 + sCont2 + sbarCont2
        tempASiv = A0(z,pht,m1,kperp2Avg,pperpAvg,eCharg)*(tempNumerator/tempDenominator)
    elif((lhaqID==1)and(lhaqbarID==-1)):
        ### This is pi0
        uCont1= NNq(x,Nu,au,bu)*(eU**2)*xFxQ2(PDFdataset,2,x,QQ)*zFzQ(FF_Pi0_dataset,2,z,QQ)
        ubarCont1= NNqbar(x,Nubar)*(eUbar**2)*xFxQ2(PDFdataset,-2,x,QQ)*zFzQ(FF_Pi0_dataset,-2,z,QQ)
        dCont1= NNq(x,Nd,ad,bd)*(eD**2)*xFxQ2(PDFdataset,1,x,QQ)*zFzQ(FF_Pi0_dataset,1,z,QQ)
        dbarCont1= NNqbar(x,Ndbar)*(eDbar**2)*xFxQ2(PDFdataset,-1,x,QQ)*zFzQ(FF_Pi0_dataset,-1,z,QQ)
        sCont1= NNq(x,Ns,aS,bs)*(eS**2)*xFxQ2(PDFdataset,3,x,QQ)*zFzQ(FF_Pi0_dataset,3,z,QQ)
        sbarCont1= NNqbar(x,Nsbar)*(eSbar**2)*xFxQ2(PDFdataset,-3,x,QQ)*zFzQ(FF_Pi0_dataset,-3,z,QQ)
        uCont2= (eU**2)*xFxQ2(PDFdataset,2,x,QQ)*zFzQ(FF_Pi0_dataset,2,z,QQ)
        ubarCont2= (eUbar**2)*xFxQ2(PDFdataset,-2,x,QQ)*zFzQ(FF_Pi0_dataset,-2,z,QQ)
        dCont2= (eD**2)*xFxQ2(PDFdataset,1,x,QQ)*zFzQ(FF_Pi0_dataset,1,z,QQ)
        dbarCont2= (eDbar**2)*xFxQ2(PDFdataset,-1,x,QQ)*zFzQ(FF_Pi0_dataset,-1,z,QQ)
        sCont2= (eS**2)*xFxQ2(PDFdataset,3,x,QQ)*zFzQ(FF_Pi0_dataset,3,z,QQ)
        sbarCont2= (eSbar**2)*xFxQ2(PDFdataset,-3,x,QQ)*zFzQ(FF_Pi0_dataset,-3,z,QQ)
        tempNumerator = uCont1 + ubarCont1 +dCont1 + dbarCont1 + sCont1 + sbarCont1
        tempDenominator = uCont2 + ubarCont2 +dCont2 + dbarCont2 + sCont2 + sbarCont2
        tempASiv = A0(z,pht,m1,kperp2Avg,pperpAvg,eCharg)*(tempNumerator/tempDenominator)
    elif((lhaqID==2)and(lhaqbarID==-3)):
        ### This is k+
        uCont1= NNq(x,Nu,au,bu)*(eU**2)*xFxQ2(PDFdataset,2,x,QQ)*zFzQ(FF_KP_dataset,2,z,QQ)
        ubarCont1= NNqbar(x,Nubar)*(eUbar**2)*xFxQ2(PDFdataset,-2,x,QQ)*zFzQ(FF_KP_dataset,-2,z,QQ)
        dCont1= NNq(x,Nd,ad,bd)*(eD**2)*xFxQ2(PDFdataset,1,x,QQ)*zFzQ(FF_KP_dataset,1,z,QQ)
        dbarCont1= NNqbar(x,Ndbar)*(eDbar**2)*xFxQ2(PDFdataset,-1,x,QQ)*zFzQ(FF_KP_dataset,-1,z,QQ)
        sCont1= NNq(x,Ns,aS,bs)*(eS**2)*xFxQ2(PDFdataset,3,x,QQ)*zFzQ(FF_KP_dataset,3,z,QQ)
        sbarCont1= NNqbar(x,Nsbar)*(eSbar**2)*xFxQ2(PDFdataset,-3,x,QQ)*zFzQ(FF_KP_dataset,-3,z,QQ)
        uCont2= (eU**2)*xFxQ2(PDFdataset,2,x,QQ)*zFzQ(FF_KP_dataset,2,z,QQ)
        ubarCont2= (eUbar**2)*xFxQ2(PDFdataset,-2,x,QQ)*zFzQ(FF_KP_dataset,-2,z,QQ)
        dCont2= (eD**2)*xFxQ2(PDFdataset,1,x,QQ)*zFzQ(FF_KP_dataset,1,z,QQ)
        dbarCont2= (eDbar**2)*xFxQ2(PDFdataset,-1,x,QQ)*zFzQ(FF_KP_dataset,-1,z,QQ)
        sCont2= (eS**2)*xFxQ2(PDFdataset,3,x,QQ)*zFzQ(FF_KP_dataset,3,z,QQ)
        sbarCont2= (eSbar**2)*xFxQ2(PDFdataset,-3,x,QQ)*zFzQ(FF_KP_dataset,-3,z,QQ)
        tempNumerator = uCont1 + ubarCont1 +dCont1 + dbarCont1 + sCont1 + sbarCont1
        tempDenominator = uCont2 + ubarCont2 +dCont2 + dbarCont2 + sCont2 + sbarCont2
        tempASiv = A0(z,pht,m1,kperp2Avg,pperpAvg,eCharg)*(tempNumerator/tempDenominator)
    elif((lhaqID==3)and(lhaqbarID==-2)):
        ### This is k+
        uCont1= NNq(x,Nu,au,bu)*(eU**2)*xFxQ2(PDFdataset,2,x,QQ)*zFzQ(FF_KM_dataset,2,z,QQ)
        ubarCont1= NNqbar(x,Nubar)*(eUbar**2)*xFxQ2(PDFdataset,-2,x,QQ)*zFzQ(FF_KM_dataset,-2,z,QQ)
        dCont1= NNq(x,Nd,ad,bd)*(eD**2)*xFxQ2(PDFdataset,1,x,QQ)*zFzQ(FF_KM_dataset,1,z,QQ)
        dbarCont1= NNqbar(x,Ndbar)*(eDbar**2)*xFxQ2(PDFdataset,-1,x,QQ)*zFzQ(FF_KM_dataset,-1,z,QQ)
        sCont1= NNq(x,Ns,aS,bs)*(eS**2)*xFxQ2(PDFdataset,3,x,QQ)*zFzQ(FF_KM_dataset,3,z,QQ)
        sbarCont1= NNqbar(x,Nsbar)*(eSbar**2)*xFxQ2(PDFdataset,-3,x,QQ)*zFzQ(FF_KM_dataset,-3,z,QQ)
        uCont2= (eU**2)*xFxQ2(PDFdataset,2,x,QQ)*zFzQ(FF_KM_dataset,2,z,QQ)
        ubarCont2= (eUbar**2)*xFxQ2(PDFdataset,-2,x,QQ)*zFzQ(FF_KM_dataset,-2,z,QQ)
        dCont2= (eD**2)*xFxQ2(PDFdataset,1,x,QQ)*zFzQ(FF_KM_dataset,1,z,QQ)
        dbarCont2= (eDbar**2)*xFxQ2(PDFdataset,-1,x,QQ)*zFzQ(FF_KM_dataset,-1,z,QQ)
        sCont2= (eS**2)*xFxQ2(PDFdataset,3,x,QQ)*zFzQ(FF_KM_dataset,3,z,QQ)
        sbarCont2= (eSbar**2)*xFxQ2(PDFdataset,-3,x,QQ)*zFzQ(FF_KM_dataset,-3,z,QQ)
        tempNumerator = uCont1 + ubarCont1 +dCont1 + dbarCont1 + sCont1 + sbarCont1
        tempDenominator = uCont2 + ubarCont2 +dCont2 + dbarCont2 + sCont2 + sbarCont2
        tempASiv = A0(z,pht,m1,kperp2Avg,pperpAvg,eCharg)*(tempNumerator/tempDenominator)
    return tempASiv


def ASivFitHadron(hadron,KV,**parms):
    m1= parms["m1"]
    Nu = parms["Nu"]
    alphau= parms["alphau"]
    betau = parms["betau"]
    Nubar = parms["Nubar"]
    Nd = parms["Nd"]
    alphad= parms["alphad"]
    betad = parms["betad"]
    Ndbar = parms["Ndbar"]
    Ns = parms["Ns"]
    alphas= parms["alphas"]
    betas = parms["betas"]
    Nsbar = parms["Nsbar"]
    if(hadron=="pi+"):
        Qflag=2
        AniQflag=-1
    elif(hadron=="pi-"):
        Qflag=1
        AniQflag=-2
    elif(hadron=="pi0"):
        Qflag=1
        AniQflag=-1
    elif(hadron=="k+"):
        Qflag=2
        AniQflag=-3
    elif(hadron=="k-"):
        Qflag=3
        AniQflag=-2
    ################
    QQ,x,z,pht=KV
    array_size=len(x)
    tempASivHad_val=[]
    for i in range(0,array_size):
        tempASivHad=Asymmetry(QQ[i],x[i],z[i],pht[i],m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar,Qflag,AniQflag)       
        tempASivHad_val.append(tempASivHad)
    return tempASivHad_val



def totalfitDataSet(datfile,m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar):
    had_len=len(hadarray(datfile))
    temHads=hadarray(datfile)
    fittot=[]
    for i in range(0,had_len):
        if temHads[i]=="pi+":
            tempfit=ASivFitHadron("pi+",Kin_hadron(datfile,"pi+"),m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            fittot.append(tempfit)
        elif temHads[i]=="pi-":
            tempfit=ASivFitHadron("pi-",Kin_hadron(datfile,"pi-"),m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            fittot.append(tempfit)
        elif temHads[i]=="pi0":
            tempfit=ASivFitHadron("pi0",Kin_hadron(datfile,"pi0"),m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            fittot.append(tempfit)
        elif temHads[i]=="k+":
            tempfit=ASivFitHadron("k+",Kin_hadron(datfile,"k+"),m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            fittot.append(tempfit)
        elif temHads[i]=="k-":
            tempfit=ASivFitHadron("k-",Kin_hadron(datfile,"k-"),m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            fittot.append(tempfit)
    return np.concatenate((fittot), axis=None)



###########################################################
########## SIDIS Asymmetry (Data)  ########################
###########################################################


def dataslice(filename,Had,Var):
    tempdf=pd.read_csv(filename)
    temp_slice=tempdf[(tempdf["hadron"]==Had)&(tempdf["1D_dependence"]==Var)]
    tempQ2=np.array(temp_slice["Q2"],dtype=object)
    tempX=np.array(temp_slice["x"],dtype=object)
    tempZ=np.array(temp_slice["z"],dtype=object)
    tempPHT=np.array(temp_slice["phT"],dtype=object)
    tempSiv=np.array(temp_slice["Siv"],dtype=object)
    temperrSiv=np.array(temp_slice["tot_err"],dtype=object)
    return tempQ2,tempX,tempZ,tempPHT,tempSiv,temperrSiv


def Kin_hadron(datfile,hadron):
    tempXfile=dataslice(datfile,hadron,"x")
    tempZfile=dataslice(datfile,hadron,"z")
    tempPhTfile=dataslice(datfile,hadron,"phT")
    ##### Q2 ################
    tempQ2_x=np.array(tempXfile[0],dtype=object)
    tempQ2_z=np.array(tempZfile[0],dtype=object)
    tempQ2_phT=np.array(tempPhTfile[0],dtype=object)
    tempQ2=np.concatenate((tempQ2_x,tempQ2_z,tempQ2_phT))
    ##### X ################
    tempX_x=np.array(tempXfile[1],dtype=object)
    tempX_z=np.array(tempZfile[1],dtype=object)
    tempX_phT=np.array(tempPhTfile[1],dtype=object)
    tempX=np.concatenate((tempX_x,tempX_z,tempX_phT))
    ##### Z ################
    tempZ_x=np.array(tempXfile[2],dtype=object)
    tempZ_z=np.array(tempZfile[2],dtype=object)
    tempZ_phT=np.array(tempPhTfile[2],dtype=object)
    tempZ=np.concatenate((tempZ_x,tempZ_z,tempZ_phT))
    ##### phT ################
    tempphT_x=np.array(tempXfile[3],dtype=object)
    tempphT_z=np.array(tempZfile[3],dtype=object)
    tempphT_phT=np.array(tempPhTfile[3],dtype=object)
    tempphT=np.concatenate((tempphT_x,tempphT_z,tempphT_phT))
    return tempQ2,tempX,tempZ,tempphT


def ASiv_data(datfile,hadron):
    tempXfile=dataslice(datfile,hadron,"x")
    tempZfile=dataslice(datfile,hadron,"z")
    tempPhTfile=dataslice(datfile,hadron,"phT")    
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
    had_len=len(hadarray(datfile))
    temHads=hadarray(datfile)
    temp_SivData=[]
    for i in range(0,had_len):
        temp_SivData.append(ASiv_data(datfile,temHads[i])[0])        
    return temp_SivData

def ASiv_Err(datfile):
    had_len=len(hadarray(datfile))
    temHads=hadarray(datfile)
    temp_SivData=[]
    for i in range(0,had_len):
        temp_SivData.append(ASiv_data(datfile,temHads[i])[1])        
    return temp_SivData




############################################################
############ Chi2 Function(s) ##############################
############################################################

def totalfitfunc(datfilesarray,m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar):
    datfilesnum=len(datfilesarray)
    temptotal=[]
    for i in range(0,datfilesnum):
        temptotal.append(totalfitDataSet(datfilesarray[i],m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar))
    return np.concatenate((temptotal), axis=None)

# def totalchi2Minuit(m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar):
#     datfilesarray=DataFilesArray
#     datfilesnum=len(datfilesarray)
#     temptotal=[]
#     for i in range(0,datfilesnum):
#         temptotal.append(totalfitDataSet(datfilesarray[i],m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar))
#     tempTheory=np.concatenate((temptotal), axis=None)
#     tempY=SiversVals(datfilesarray)
#     tempYErr=SiversErrVals(datfilesarray)
#     tempChi2=np.sum(((tempY-tempTheory)/tempYErr)**2)
#     return tempChi2

############################################################
############ Calculating Chi2 values #######################
############################################################

def Asymmetry_for_Hadron(datfile,hadron,dep,m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar):
    kperp2Avg=Kp2A
    pperpAvg=Pp2A
    eCharg=ee                            
    PDFfile=PDFdataset
    #had_len=len(hadarray(datfile))
    if hadron=="pi+":
        tempfit=ASivFitHadron("pi+",Kin_hadron(datfile,"pi+"),m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
    elif hadron=="pi-":
        tempfit=ASivFitHadron("pi-",Kin_hadron(datfile,"pi-"),m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
    elif hadron=="pi0":
        tempfit=ASivFitHadron("pi0",Kin_hadron(datfile,"pi0"),m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
    elif hadron=="k+":
        tempfit=ASivFitHadron("k+",Kin_hadron(datfile,"k+"),m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
    elif hadron=="k-":
        tempfit=ASivFitHadron("k-",Kin_hadron(datfile,"k-"),m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
    return tempfit

def ASiv_data_comp(datfile,hadron,dependence):
    tempDepfile=dataslice(datfile,hadron,dependence)
    ##### Asy ################
    tempAsy_Dep=np.array(tempDepfile[4],dtype=object)
    ##### err ################
    tempAsyErr_Dep=np.array(tempDepfile[5],dtype=object)+ tempAsy_Dep*0.20
    return tempAsy_Dep,tempAsyErr_Dep


def Chi2_list_prep1(datfile,hadron,dependence,m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar):
    #set_string_function
    tempfit=Asymmetry_for_Hadron(datfile,hadron,dependence,m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
    tempdata=ASiv_data_comp(datfile,hadron,dependence)[0]
    tempdataerr=ASiv_data_comp(datfile,hadron,dependence)[1]
    tempchi2=chisquare(tempfit, tempdata, tempdataerr)
    return tempchi2,len(tempdata)

def chisquare(y, yhat, err):
    return np.sum(((y - yhat)/err)**2)


def Chi2CompSingleDataSetModified(datfile,m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar):
    had_len=len(hadarray(datfile))
    temHads=hadarray(datfile)
    Chi2Array=[]
    HadronArray=[]
    DependenceArray=[]
    DataPoints=[]
    tempdf_dat=pd.DataFrame({'Hadron':[],'Dependence':[],'Chi2':[],'N_data':[]})
    for i in range(0,had_len):
        if temHads[i]=="pi+":
            HadronArray.append("pi+")
            tempChi2valX=Chi2_list_prep1(datfile,"pi+","x",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsX=Chi2_list_prep1(datfile,"pi+","x",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsX)
            DependenceArray.append("x")
            Chi2Array.append(tempChi2valX)
            HadronArray.append("pi+")
            tempChi2valZ=Chi2_list_prep1(datfile,"pi+","z",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsZ=Chi2_list_prep1(datfile,"pi+","z",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsZ)            
            DependenceArray.append("z")
            Chi2Array.append(tempChi2valZ)
            HadronArray.append("pi+")
            tempChi2valPHT=Chi2_list_prep1(datfile,"pi+","phT",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsPHT=Chi2_list_prep1(datfile,"pi+","phT",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsPHT)            
            DependenceArray.append("phT")
            Chi2Array.append(tempChi2valPHT)
        elif temHads[i]=="pi-":
            HadronArray.append("pi-")
            tempChi2valX=Chi2_list_prep1(datfile,"pi-","x",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsX=Chi2_list_prep1(datfile,"pi-","x",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsX)
            DependenceArray.append("x")
            Chi2Array.append(tempChi2valX)
            HadronArray.append("pi-")
            tempChi2valZ=Chi2_list_prep1(datfile,"pi-","z",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsZ=Chi2_list_prep1(datfile,"pi-","z",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsZ)            
            DependenceArray.append("z")
            Chi2Array.append(tempChi2valZ)
            HadronArray.append("pi-")
            tempChi2valPHT=Chi2_list_prep1(datfile,"pi-","phT",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsPHT=Chi2_list_prep1(datfile,"pi-","phT",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsPHT)            
            DependenceArray.append("phT")
            Chi2Array.append(tempChi2valPHT)
        elif temHads[i]=="pi0":
            HadronArray.append("pi0")
            tempChi2valX=Chi2_list_prep1(datfile,"pi0","x",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsX=Chi2_list_prep1(datfile,"pi0","x",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsX)
            DependenceArray.append("x")
            Chi2Array.append(tempChi2valX)
            HadronArray.append("pi0")
            tempChi2valZ=Chi2_list_prep1(datfile,"pi0","z",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsZ=Chi2_list_prep1(datfile,"pi0","z",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsZ)            
            DependenceArray.append("z")
            Chi2Array.append(tempChi2valZ)
            HadronArray.append("pi0")
            tempChi2valPHT=Chi2_list_prep1(datfile,"pi0","phT",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsPHT=Chi2_list_prep1(datfile,"pi0","phT",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsPHT)            
            DependenceArray.append("phT")
            Chi2Array.append(tempChi2valPHT)
        elif temHads[i]=="k+":
            HadronArray.append("k+")
            tempChi2valX=Chi2_list_prep1(datfile,"k+","x",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsX=Chi2_list_prep1(datfile,"k+","x",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsX)
            DependenceArray.append("x")
            Chi2Array.append(tempChi2valX)
            HadronArray.append("k+")
            tempChi2valZ=Chi2_list_prep1(datfile,"k+","z",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsZ=Chi2_list_prep1(datfile,"k+","z",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsZ)            
            DependenceArray.append("z")
            Chi2Array.append(tempChi2valZ)
            HadronArray.append("k+")
            tempChi2valPHT=Chi2_list_prep1(datfile,"k+","phT",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsPHT=Chi2_list_prep1(datfile,"k+","phT",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsPHT)            
            DependenceArray.append("phT")
            Chi2Array.append(tempChi2valPHT)
        elif temHads[i]=="k-":
            HadronArray.append("k-")
            tempChi2valX=Chi2_list_prep1(datfile,"k-","x",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsX=Chi2_list_prep1(datfile,"k-","x",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsX)
            DependenceArray.append("x")
            Chi2Array.append(tempChi2valX)
            HadronArray.append("k-")
            tempChi2valZ=Chi2_list_prep1(datfile,"k-","z",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsZ=Chi2_list_prep1(datfile,"k-","z",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsZ)            
            DependenceArray.append("z")
            Chi2Array.append(tempChi2valZ)
            HadronArray.append("k-")
            tempChi2valPHT=Chi2_list_prep1(datfile,"k-","phT",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsPHT=Chi2_list_prep1(datfile,"k-","phT",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsPHT)            
            DependenceArray.append("phT")
            Chi2Array.append(tempChi2valPHT)
    tempdf_dat['Hadron']=HadronArray
    tempdf_dat['Dependence']=DependenceArray
    tempdf_dat['Chi2']=Chi2Array
    tempdf_dat['N_data']=DataPoints
    return tempdf_dat
    #return np.array(Chi2Array)    

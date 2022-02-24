#######################################################################
############          SIDIS Definitions               #################
############        Written by Ishara Fernando        #################
############ Last upgrade: February-01-2022 ###########################
#######################################################################

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

FF_PiP_dataset=["NNFF10_PIp_nlo"]
FF_PiM_dataset=["NNFF10_PIm_nlo"]
FF_Pi0_dataset=["NNFF10_PIsum_nlo"]
FF_KP_dataset=["NNFF10_KAp_nlo"]
FF_KM_dataset=["NNFF10_KAm_nlo"]

# FF_PiP_dataset=["DSS14_NLO_Pip"]
# FF_PiM_dataset=["DSS14_NLO_Pim"]
# FF_Pi0_dataset=["DSS14_NLO_PiSum"]
# FF_KP_dataset=["DSS17_NLO_KaonPlus"]
# FF_KM_dataset=["DSS17_NLO_KaonMinus"]

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


#####################################################################
############         Asymmetry Calculations            ##############
#####################################################################

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

#### Asymmetry for given hadron in SIDIS ######

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


#### Asymmetry for a given hadron and given dependence ####

def Asymmetry_for_Hadron(datfile,hadron,dep,m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar):
    #had_len=len(hadarray(datfile))
    if hadron=="pi+":
        tempfit=ASivFitHadron("pi+",Kin_hadron_Th(datfile,"pi+",dep),m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
    elif hadron=="pi-":
        tempfit=ASivFitHadron("pi-",Kin_hadron_Th(datfile,"pi-",dep),m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
    elif hadron=="pi0":
        tempfit=ASivFitHadron("pi0",Kin_hadron_Th(datfile,"pi0",dep),m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
    elif hadron=="k+":
        tempfit=ASivFitHadron("k+",Kin_hadron_Th(datfile,"k+",dep),m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
    elif hadron=="k-":
        tempfit=ASivFitHadron("k-",Kin_hadron_Th(datfile,"k-",dep),m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
    return tempfit

#############################################################
############# Preparation of Kinematics #####################
#############################################################

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

def Kin_hadron_Th(datfile,hadron,dep):
    if dep == "x":
        tempXfile=dataslice(datfile,hadron,"x")
        tempQ2=np.array(tempXfile[0],dtype=object)
        tempX=np.array(tempXfile[1],dtype=object)
        tempZ=np.array(tempXfile[2],dtype=object)
        tempphT=np.array(tempXfile[3],dtype=object)
    elif dep == "z":
        tempZfile=dataslice(datfile,hadron,"z")
        tempQ2=np.array(tempZfile[0],dtype=object)
        tempX=np.array(tempZfile[1],dtype=object)
        tempZ=np.array(tempZfile[2],dtype=object)
        tempphT=np.array(tempZfile[3],dtype=object)
    elif dep == "phT":
        tempPhTfile=dataslice(datfile,hadron,"phT")
        tempQ2=np.array(tempPhTfile[0],dtype=object)
        tempX=np.array(tempPhTfile[1],dtype=object)
        tempZ=np.array(tempPhTfile[2],dtype=object)
        tempphT=np.array(tempPhTfile[3],dtype=object)
    return tempQ2,tempX,tempZ,tempphT


##################################################################
############# Asymmetry values for a data file ###################
##################################################################

def totalfitDataSet(datfile,m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar):
    had_len=len(hadarray(datfile))
    temHads=hadarray(datfile)
    fittot=[]
    for i in range(0,had_len):
        if temHads[i]=="pi+":
            tempfitx=Asymmetry_for_Hadron(datfile,"pi+","x",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            tempfitz=Asymmetry_for_Hadron(datfile,"pi+","z",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            tempfitphT=Asymmetry_for_Hadron(datfile,"pi+","phT",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT), axis=None)
            fittot.append(tempfit)
        elif temHads[i]=="pi-":
            tempfitx=Asymmetry_for_Hadron(datfile,"pi-","x",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            tempfitz=Asymmetry_for_Hadron(datfile,"pi-","z",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            tempfitphT=Asymmetry_for_Hadron(datfile,"pi-","phT",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT), axis=None)
            fittot.append(tempfit)
        elif temHads[i]=="pi0":
            tempfitx=Asymmetry_for_Hadron(datfile,"pi0","x",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            tempfitz=Asymmetry_for_Hadron(datfile,"pi0","z",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            tempfitphT=Asymmetry_for_Hadron(datfile,"pi0","phT",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT), axis=None)
            fittot.append(tempfit)
        elif temHads[i]=="k+":
            tempfitx=Asymmetry_for_Hadron(datfile,"k+","x",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            tempfitz=Asymmetry_for_Hadron(datfile,"k+","z",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            tempfitphT=Asymmetry_for_Hadron(datfile,"k+","phT",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT), axis=None)
            fittot.append(tempfit)
        elif temHads[i]=="k-":
            tempfitx=Asymmetry_for_Hadron(datfile,"k-","x",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            tempfitz=Asymmetry_for_Hadron(datfile,"k-","z",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            tempfitphT=Asymmetry_for_Hadron(datfile,"k-","phT",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT), axis=None)
            fittot.append(tempfit)
    return np.concatenate((fittot), axis=None)


##################################################################
############# Kinematics for Asymmetry values for a data file ####
##################################################################

def totalfit_Theory_Kin(datfile):
    had_len=len(hadarray(datfile))
    temHads=hadarray(datfile)
    fittot=[]
    for i in range(0,had_len):
        if temHads[i]=="pi+":
            tempfitx=Kin_hadron_Th(datfile,"pi+","x")
            tempfitz=Kin_hadron_Th(datfile,"pi+","z")
            tempfitphT=Kin_hadron_Th(datfile,"pi+","phT")
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT), axis=None)
            fittot.append(tempfit)
        elif temHads[i]=="pi-":
            tempfitx=Kin_hadron_Th(datfile,"pi-","x")
            tempfitz=Kin_hadron_Th(datfile,"pi-","z")
            tempfitphT=Kin_hadron_Th(datfile,"pi-","phT")
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT), axis=None)
            fittot.append(tempfit)
        elif temHads[i]=="pi0":
            tempfitx=Kin_hadron_Th(datfile,"pi0","x")
            tempfitz=Kin_hadron_Th(datfile,"pi0","z")
            tempfitphT=Kin_hadron_Th(datfile,"pi0","phT")
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT), axis=None)
            fittot.append(tempfit)
        elif temHads[i]=="k+":
            tempfitx=Kin_hadron_Th(datfile,"k+","x")
            tempfitz=Kin_hadron_Th(datfile,"k+","z")
            tempfitphT=Kin_hadron_Th(datfile,"k+","phT")
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT), axis=None)
            fittot.append(tempfit)
        elif temHads[i]=="k-":
            tempfitx=Kin_hadron_Th(datfile,"k-","x")
            tempfitz=Kin_hadron_Th(datfile,"k-","z")
            tempfitphT=Kin_hadron_Th(datfile,"k-","phT")
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT), axis=None)
            fittot.append(tempfit)
    return np.concatenate((fittot))

###########################################################
########## SIDIS Asymmetry (Data)  ########################
###########################################################


def ASiv_data_comp(datfile,hadron,dependence):
    tempDepfile=dataslice(datfile,hadron,dependence)
    ##### Asy ################
    tempAsy_Dep=np.array(tempDepfile[4],dtype=object)
    ##### err ################
    ##### Here we can index the systematic uncertainty to be considered ##
    tempAsyErr_Dep=np.array(tempDepfile[5],dtype=object)+ tempAsy_Dep*0
    return tempAsy_Dep,tempAsyErr_Dep

def ASiv_data_Kins(datfile,hadron,dependence):
    tempDepfile=dataslice(datfile,hadron,dependence)
    ##########################
    tempQ2 = np.array(tempDepfile[0],dtype=object)
    tempX = np.array(tempDepfile[1],dtype=object)
    tempZ = np.array(tempDepfile[2],dtype=object)
    tempphT = np.array(tempDepfile[3],dtype=object)
    return tempQ2,tempX,tempZ,tempphT

###### Asymmetry Data after processing with the kinematics ########

def totalData(datfile,indx):
    had_len=len(hadarray(datfile))
    temHads=hadarray(datfile)
    fittot=[]
    for i in range(0,had_len):
        if temHads[i]=="pi+":
            tempfitx=ASiv_data_comp(datfile,"pi+","x")[indx]
            tempfitz=ASiv_data_comp(datfile,"pi+","z")[indx]
            tempfitphT=ASiv_data_comp(datfile,"pi+","phT")[indx]
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT), axis=None)
            fittot.append(tempfit)
        elif temHads[i]=="pi-":
            tempfitx=ASiv_data_comp(datfile,"pi-","x")[indx]
            tempfitz=ASiv_data_comp(datfile,"pi-","z")[indx]
            tempfitphT=ASiv_data_comp(datfile,"pi-","phT")[indx]
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT), axis=None)
            fittot.append(tempfit)
        elif temHads[i]=="pi0":
            tempfitx=ASiv_data_comp(datfile,"pi0","x")[indx]
            tempfitz=ASiv_data_comp(datfile,"pi0","z")[indx]
            tempfitphT=ASiv_data_comp(datfile,"pi0","phT")[indx]
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT), axis=None)
            fittot.append(tempfit)
        elif temHads[i]=="k+":
            tempfitx=ASiv_data_comp(datfile,"k+","x")[indx]
            tempfitz=ASiv_data_comp(datfile,"k+","z")[indx]
            tempfitphT=ASiv_data_comp(datfile,"k+","phT")[indx]
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT), axis=None)
            fittot.append(tempfit)
        elif temHads[i]=="k-":
            tempfitx=ASiv_data_comp(datfile,"k-","x")[indx]
            tempfitz=ASiv_data_comp(datfile,"k-","z")[indx]
            tempfitphT=ASiv_data_comp(datfile,"k-","phT")[indx]
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT), axis=None)
            fittot.append(tempfit)
    return np.concatenate((fittot), axis=None)

######## Kinematics for Asymmetry data ######################

def totalData_Kins(datfile):
    had_len=len(hadarray(datfile))
    temHads=hadarray(datfile)
    fittot=[]
    for i in range(0,had_len):
        if temHads[i]=="pi+":
            tempfitx=ASiv_data_Kins(datfile,"pi+","x")
            tempfitz=ASiv_data_Kins(datfile,"pi+","z")
            tempfitphT=ASiv_data_Kins(datfile,"pi+","phT")
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT), axis=None)
            fittot.append(tempfit)
        elif temHads[i]=="pi-":
            tempfitx=ASiv_data_Kins(datfile,"pi-","x")
            tempfitz=ASiv_data_Kins(datfile,"pi-","z")
            tempfitphT=ASiv_data_Kins(datfile,"pi-","phT")
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT), axis=None)
            fittot.append(tempfit)
        elif temHads[i]=="pi0":
            tempfitx=ASiv_data_Kins(datfile,"pi0","x")
            tempfitz=ASiv_data_Kins(datfile,"pi0","z")
            tempfitphT=ASiv_data_Kins(datfile,"pi0","phT")
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT), axis=None)
            fittot.append(tempfit)
        elif temHads[i]=="k+":
            tempfitx=ASiv_data_Kins(datfile,"k+","x")
            tempfitz=ASiv_data_Kins(datfile,"k+","z")
            tempfitphT=ASiv_data_Kins(datfile,"k+","phT")
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT), axis=None)
            fittot.append(tempfit)
        elif temHads[i]=="k-":
            tempfitx=ASiv_data_Kins(datfile,"k-","x")
            tempfitz=ASiv_data_Kins(datfile,"k-","z")
            tempfitphT=ASiv_data_Kins(datfile,"k-","phT")
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT), axis=None)
            fittot.append(tempfit)
    return np.concatenate((fittot), axis=None)

#### This will generate chi2 values for given hadron & dependence ####

def chisquare(y, yhat, err):
    return np.sum(((y - yhat)/err)**2)

def Chi2_list_prep(datfile,hadron,dependence,m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar):
    #set_string_function
    tempfit=Asymmetry_for_Hadron(datfile,hadron,dependence,m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
    tempdata=ASiv_data_comp(datfile,hadron,dependence)[0]
    #### Add any systematic uncertainty for data file ##########
    tempdataerr=ASiv_data_comp(datfile,hadron,dependence)[1] + tempdata*0
    tempchi2=chisquare(tempfit, tempdata, tempdataerr)
    return tempchi2,len(tempdata)

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
            tempChi2valX=Chi2_list_prep(datfile,"pi+","x",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsX=Chi2_list_prep(datfile,"pi+","x",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsX)
            DependenceArray.append("x")
            Chi2Array.append(tempChi2valX)
            HadronArray.append("pi+")
            tempChi2valZ=Chi2_list_prep(datfile,"pi+","z",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsZ=Chi2_list_prep(datfile,"pi+","z",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsZ)            
            DependenceArray.append("z")
            Chi2Array.append(tempChi2valZ)
            HadronArray.append("pi+")
            tempChi2valPHT=Chi2_list_prep(datfile,"pi+","phT",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsPHT=Chi2_list_prep(datfile,"pi+","phT",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsPHT)            
            DependenceArray.append("phT")
            Chi2Array.append(tempChi2valPHT)
        elif temHads[i]=="pi-":
            HadronArray.append("pi-")
            tempChi2valX=Chi2_list_prep(datfile,"pi-","x",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsX=Chi2_list_prep(datfile,"pi-","x",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsX)
            DependenceArray.append("x")
            Chi2Array.append(tempChi2valX)
            HadronArray.append("pi-")
            tempChi2valZ=Chi2_list_prep(datfile,"pi-","z",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsZ=Chi2_list_prep(datfile,"pi-","z",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsZ)            
            DependenceArray.append("z")
            Chi2Array.append(tempChi2valZ)
            HadronArray.append("pi-")
            tempChi2valPHT=Chi2_list_prep(datfile,"pi-","phT",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsPHT=Chi2_list_prep(datfile,"pi-","phT",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsPHT)            
            DependenceArray.append("phT")
            Chi2Array.append(tempChi2valPHT)
        elif temHads[i]=="pi0":
            HadronArray.append("pi0")
            tempChi2valX=Chi2_list_prep(datfile,"pi0","x",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsX=Chi2_list_prep(datfile,"pi0","x",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsX)
            DependenceArray.append("x")
            Chi2Array.append(tempChi2valX)
            HadronArray.append("pi0")
            tempChi2valZ=Chi2_list_prep(datfile,"pi0","z",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsZ=Chi2_list_prep(datfile,"pi0","z",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsZ)            
            DependenceArray.append("z")
            Chi2Array.append(tempChi2valZ)
            HadronArray.append("pi0")
            tempChi2valPHT=Chi2_list_prep(datfile,"pi0","phT",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsPHT=Chi2_list_prep(datfile,"pi0","phT",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsPHT)            
            DependenceArray.append("phT")
            Chi2Array.append(tempChi2valPHT)
        elif temHads[i]=="k+":
            HadronArray.append("k+")
            tempChi2valX=Chi2_list_prep(datfile,"k+","x",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsX=Chi2_list_prep(datfile,"k+","x",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsX)
            DependenceArray.append("x")
            Chi2Array.append(tempChi2valX)
            HadronArray.append("k+")
            tempChi2valZ=Chi2_list_prep(datfile,"k+","z",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsZ=Chi2_list_prep(datfile,"k+","z",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsZ)            
            DependenceArray.append("z")
            Chi2Array.append(tempChi2valZ)
            HadronArray.append("k+")
            tempChi2valPHT=Chi2_list_prep(datfile,"k+","phT",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsPHT=Chi2_list_prep(datfile,"k+","phT",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsPHT)            
            DependenceArray.append("phT")
            Chi2Array.append(tempChi2valPHT)
        elif temHads[i]=="k-":
            HadronArray.append("k-")
            tempChi2valX=Chi2_list_prep(datfile,"k-","x",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsX=Chi2_list_prep(datfile,"k-","x",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsX)
            DependenceArray.append("x")
            Chi2Array.append(tempChi2valX)
            HadronArray.append("k-")
            tempChi2valZ=Chi2_list_prep(datfile,"k-","z",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsZ=Chi2_list_prep(datfile,"k-","z",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsZ)            
            DependenceArray.append("z")
            Chi2Array.append(tempChi2valZ)
            HadronArray.append("k-")
            tempChi2valPHT=Chi2_list_prep(datfile,"k-","phT",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[0]
            tempPointsPHT=Chi2_list_prep(datfile,"k-","phT",m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)[1]
            DataPoints.append(tempPointsPHT)            
            DependenceArray.append("phT")
            Chi2Array.append(tempChi2valPHT)
    tempdf_dat['Hadron']=HadronArray
    tempdf_dat['Dependence']=DependenceArray
    tempdf_dat['Chi2']=Chi2Array
    tempdf_dat['N_data']=DataPoints
    return tempdf_dat
    #return np.array(Chi2Array)    


############################################################
############ Chi2 Function(s) ##############################
############################################################

def totalfitfunc(datfilesarray,m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar):
    datfilesnum=len(datfilesarray)
    temptotal=[]
    for i in range(0,datfilesnum):
        temptotal.append(totalfitDataSet(datfilesarray[i],m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar))
    return np.concatenate((temptotal), axis=None)

############################################################
############ Usefule for calculating #N data points ########
############################################################


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

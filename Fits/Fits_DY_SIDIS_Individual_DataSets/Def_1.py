import lhapdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


# Global Constants
M1_test=np.sqrt(0.8)
Kp2A=0.57
Pp2A=0.12

ee=1
eU=2/3
eUbar=-2/3
eD=-1/3
eDbar=1/3
eS=-1/3
eSbar=1/3

AlphaU_test=1.0
BetaU_test=6.6
AlphaD_test=1.9
BetaD_test=10
AlphaS_test=0
BetaS_test=0

NU_test=0.18
NUbar_test=-0.01
ND_test=-0.52
NDbar_test=-0.06
NS_test=0
NSbar_test=0


# M1_test=1.303

# AlphaU_test=0.6455
# BetaU_test=3.122
# AlphaD_test=1.777
# BetaD_test=7.788
# AlphaS_test=0.00006887
# BetaS_test=0.0000000005537

# NU_test=0.1695
# NUbar_test=0.007605
# ND_test=-0.4345
# NDbar_test=-0.1420
# NS_test=0.5626
# NSbar_test=-0.1221

M1_t2 = 1.303

AlphaU_t2=0.645
BetaU_t2=3.122
AlphaD_t2=1.777
BetaD_t2=7.788
AlphaS_t2=6.84*10**(-5)
BetaS_t2=5.987*10**(-10)

NU_t2=0.169
NUbar_t2=0.007
ND_t2=-0.434
NDbar_t2=-0.142
NS_t2=0.563
NSbar_t2=-0.122

PDFdataset = lhapdf.mkPDF("cteq61")
#PDFdataset = lhapdf.mkPDF("CT10nnlo")
#FF_pion_dataset=["JAM19FF_pion_nlo"]
#FF_kaon_dataset=["JAM19FF_kaon_nlo"]
FF_PiP_dataset=["NNFF10_PIp_nlo"]
FF_PiM_dataset=["NNFF10_PIm_nlo"]
FF_Pi0_dataset=["NNFF10_PIsum_nlo"]
FF_KP_dataset=["NNFF10_KAp_nlo"]
FF_KM_dataset=["NNFF10_KAm_nlo"]

def hadarray(filename):
    tempdf=pd.read_csv(filename)
    temphad_data=tempdf["hadron"]
    temphad=temphad_data.dropna().unique()
    refined_had_array=[]
    for i in range(0,len(temphad)):
        if((temphad[i]=="pi+") or (temphad[i]=="pi-") or (temphad[i]=="pi0") or (temphad[i]=="k+") or (temphad[i]=="k-")):
            refined_had_array.append(temphad[i])
    return refined_had_array

#print(hadarray(Datafile))

def dataslice(filename,Had,Var):
    tempdf=pd.read_csv(filename)
    temp_slice=tempdf[(tempdf["hadron"]==Had)&(tempdf["1D_dependence"]==Var)]
    tempQ2=np.array(temp_slice["Q2"])
    tempX=np.array(temp_slice["x"])
    tempZ=np.array(temp_slice["z"])
    tempPHT=np.array(temp_slice["phT"])
    tempSiv=np.array(temp_slice["Siv"])
    temperrSiv=np.array(temp_slice["tot_err"])
    return tempQ2,tempX,tempZ,tempPHT,tempSiv,temperrSiv

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

#def NNq(x,Nq,aq,bq):
 #   aaq=abs(aq)
  #  bbq=abs(bq)
   # tempNNq = Nq*(x**aaq)*((1-x)**(bbq))*((aaq+bbq)**(aaq+bbq))/((aaq**aaq)*(bbq**bbq))
    #return tempNNq

def NNqbar(x,Nqbar):
    tempNNqbar = Nqbar
    return tempNNqbar

def xFxQ2(dataset,flavor,x,QQ):
    temp_parton_dist_x=np.array(dataset.xfxQ2(flavor, x, QQ))
    return temp_parton_dist_x

def zFzQ(dataset,flavor,zz,QQ):
    # Here "0" represents the central values from the girds
    temp_zD1=lhapdf.mkPDF(dataset[0], 0)
    zD1_vec=np.array(temp_zD1.xfxQ2(flavor,zz,QQ))
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

## ** two stars represents a dictionary
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


#ASivFitHadron("pi+",Kin_hadron(DataFilesArray[0],"pi+"),m1=1,Nu=1,alphau=1,betau=1,Nubar=1,Nd=1,alphad=1,betad=1,Ndbar=1,Ns=1,alphas=1,betas=1,Nsbar=1)

def Kin_hadron(datfile,hadron):
    tempXfile=dataslice(datfile,hadron,"x")
    tempZfile=dataslice(datfile,hadron,"z")
    tempPhTfile=dataslice(datfile,hadron,"phT")
    ##### Q2 ################
    tempQ2_x=np.array(tempXfile[0])
    tempQ2_z=np.array(tempZfile[0])
    tempQ2_phT=np.array(tempPhTfile[0])
    tempQ2=np.concatenate((tempQ2_x,tempQ2_z,tempQ2_phT))
    ##### X ################
    tempX_x=np.array(tempXfile[1])
    tempX_z=np.array(tempZfile[1])
    tempX_phT=np.array(tempPhTfile[1])
    tempX=np.concatenate((tempX_x,tempX_z,tempX_phT))
    ##### Z ################
    tempZ_x=np.array(tempXfile[2])
    tempZ_z=np.array(tempZfile[2])
    tempZ_phT=np.array(tempPhTfile[2])
    tempZ=np.concatenate((tempZ_x,tempZ_z,tempZ_phT))
    ##### phT ################
    tempphT_x=np.array(tempXfile[3])
    tempphT_z=np.array(tempZfile[3])
    tempphT_phT=np.array(tempPhTfile[3])
    tempphT=np.concatenate((tempphT_x,tempphT_z,tempphT_phT))
    return tempQ2,tempX,tempZ,tempphT


def Kin_Had(datfile):
    had_len=len(hadarray(datfile))
    temHads=hadarray(datfile)
    temp_kin=[]
    for i in range(0,had_len):
        temp_kin.append(Kin_hadron(datfile,temHads[i]))        
    return temp_kin

#print(len(Kin_Had(Datafile)))
#print(Kin_Had(Datafile)[3])


#### Sivers values

def ASiv_data(datfile,hadron):
    tempXfile=dataslice(datfile,hadron,"x")
    tempZfile=dataslice(datfile,hadron,"z")
    tempPhTfile=dataslice(datfile,hadron,"phT")    
    ##### Asy ################
    tempAsy_x=np.array(tempXfile[4])
    tempAsy_z=np.array(tempZfile[4])
    tempAsy_phT=np.array(tempPhTfile[4])
    tempAsy=np.concatenate((tempAsy_x,tempAsy_z,tempAsy_phT))
    ##### err ################
    tempAsyErr_x=np.array(tempXfile[5])
    tempAsyErr_z=np.array(tempZfile[5])
    tempAsyErr_phT=np.array(tempPhTfile[5])
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

#print(np.concatenate(ASiv_Val(Datafile), axis=None))
#print(np.concatenate(ASiv_Err(Datafile), axis=None))

##### This function will calculate the theory values for each data set

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


### This function will combine all the data sets into a single array

def totalfitfunc(datfilesarray,m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar):
    datfilesnum=len(datfilesarray)
    temptotal=[]
    for i in range(0,datfilesnum):
        temptotal.append(totalfitDataSet(datfilesarray[i],m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar))
    return np.concatenate((temptotal), axis=None)

def totalchi2Minuit(m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar):
    datfilesarray=DataFilesArray
    datfilesnum=len(datfilesarray)
    temptotal=[]
    for i in range(0,datfilesnum):
        temptotal.append(totalfitDataSet(datfilesarray[i],m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar))
    tempTheory=np.concatenate((temptotal), axis=None)
    tempY=SiversVals(datfilesarray)
    tempYErr=SiversErrVals(datfilesarray)
    tempChi2=np.sum(((tempY-tempTheory)/tempYErr)**2)
    return tempChi2


### This function collects all data and combine together into one array
    
def SiversVals(datafilesarray):
    datfilesnum=len(datafilesarray)
    tempSiv=[]
    for i in range(0,datfilesnum):
        tempSiv.append(ASiv_Val(datafilesarray[i]))
    tempflatSiv=np.concatenate((tempSiv), axis=None)
    return np.concatenate((tempflatSiv), axis=None)

def SiversErrVals(datafilesarray):
    datfilesnum=len(datafilesarray)
    tempSivErr=[]
    for i in range(0,datfilesnum):
        tempSivErr.append(ASiv_Err(datafilesarray[i]))
    tempflatSivErr=np.concatenate((tempSivErr), axis=None)
    return np.concatenate((tempflatSivErr), axis=None)

def totchi2(y,yhat,err):
    tempval=np.sum(((y-yhat)/err)**2)
    return tempval


def param_samples(pars, cov):
    par_sample = []
    pars_sigma=np.diag(cov)
    org_theory=totalfitfunc(DataFilesArray,*pars)
    org_chi2=totchi2(SiversVals(DataFilesArray),org_theory,SiversErrVals(DataFilesArray))
    #print(org_chi2)
    for i in range(100):
        #temp_pars = np.random.normal(pars, 0.1*np.diag(cov))
        temp_pars = np.random.normal(pars, 0.03*np.diag(cov))
        par_sample.append(temp_pars)
        temp_theory=totalfitfunc(DataFilesArray,*temp_pars)
        temp_chi2=totchi2(SiversVals(DataFilesArray),temp_theory,SiversErrVals(DataFilesArray))
        #print(temp_chi2)
        if(np.abs(org_chi2-temp_chi2)<105):
            par_sample.append(temp_pars)
        #temp_theory=ASivFitHadron(had,(tempQ,tempX,tempZ,tempphT),m1=result[0],Nu=result[1],alphau=result[2],betau=result[3],Nubar=result[4],Nd=result[5],alphad=result[6],betad=result[7],Ndbar=result[8],Ns=result[9],alphas=result[10],betas=result[11],Nsbar=result[12])
        #d = func(X, *temp_pars).real
        #res.append(list(d))
    return np.array(par_sample)

#print(totchi2(SiversVals(DataFilesArray),theoryH09,SiversErrVals(DataFilesArray))/len(SiversVals(DataFilesArray)))
#print(totchi2(SiversVals(DataFilesArray),theoryH09,SiversErrVals(DataFilesArray)))

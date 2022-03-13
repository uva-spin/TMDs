import lhapdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from Global_Constants import *


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


# SIDIS_PDFs_HERMES_p_2009='../Calc_Grids/SIDIS_PDFs/PDFs_HERMES_p_2009.csv'
SIDIS_PDFs_HERMES_p_2020='../Calc_Grids/SIDIS_PDFs/PDFs_HERMES_p_2020.csv'
SIDIS_PDFs_COMPASS_d_2009='../Calc_Grids/SIDIS_PDFs/PDFs_COMPASS_d_2009.csv'
SIDIS_PDFs_COMPASS_p_2009='../Calc_Grids/SIDIS_PDFs/PDFs_COMPASS_p_2015.csv'

SIDIS_PDFs_Array=(SIDIS_PDFs_HERMES_p_2020,SIDIS_PDFs_COMPASS_d_2009,SIDIS_PDFs_COMPASS_p_2009)
###########################################################################
#####################  SIDIS FFs #########################################
###########################################################################


# SIDIS_FFs_PiP_HERMES_p_2009='../Calc_Grids/SIDIS_FFs/FF_PiP_HERMES_p_2009.csv'
SIDIS_FFs_PiP_HERMES_p_2020='../Calc_Grids/SIDIS_FFs/FF_PiP_HERMES_p_2020.csv'
SIDIS_FFs_PiP_COMPASS_d_2009='../Calc_Grids/SIDIS_FFs/FF_PiP_COMPASS_d_2009.csv'
SIDIS_FFs_PiP_COMPASS_p_2015='../Calc_Grids/SIDIS_FFs/FF_PiP_COMPASS_p_2015.csv'

# SIDIS_FFs_PiM_HERMES_p_2009='../Calc_Grids/SIDIS_FFs/FF_PiM_HERMES_p_2009.csv'
SIDIS_FFs_PiM_HERMES_p_2020='../Calc_Grids/SIDIS_FFs/FF_PiM_HERMES_p_2020.csv'
SIDIS_FFs_PiM_COMPASS_d_2009='../Calc_Grids/SIDIS_FFs/FF_PiM_COMPASS_d_2009.csv'
SIDIS_FFs_PiM_COMPASS_p_2015='../Calc_Grids/SIDIS_FFs/FF_PiM_COMPASS_p_2015.csv'

# SIDIS_FFs_Pi0_HERMES_p_2009='../Calc_Grids/SIDIS_FFs/FF_Pi0_HERMES_p_2009.csv'
SIDIS_FFs_Pi0_HERMES_p_2020='../Calc_Grids/SIDIS_FFs/FF_Pi0_HERMES_p_2020.csv'
SIDIS_FFs_Pi0_COMPASS_d_2009='../Calc_Grids/SIDIS_FFs/FF_Pi0_COMPASS_d_2009.csv'
SIDIS_FFs_Pi0_COMPASS_p_2015='../Calc_Grids/SIDIS_FFs/FF_Pi0_COMPASS_p_2015.csv'

# SIDIS_FFs_KP_HERMES_p_2009='../Calc_Grids/SIDIS_FFs/FF_KP_HERMES_p_2009.csv'
SIDIS_FFs_KP_HERMES_p_2020='../Calc_Grids/SIDIS_FFs/FF_KP_HERMES_p_2020.csv'
SIDIS_FFs_KP_COMPASS_d_2009='../Calc_Grids/SIDIS_FFs/FF_KP_COMPASS_d_2009.csv'
SIDIS_FFs_KP_COMPASS_p_2015='../Calc_Grids/SIDIS_FFs/FF_KP_COMPASS_p_2015.csv'

# SIDIS_FFs_KM_HERMES_p_2009='../Calc_Grids/SIDIS_FFs/FF_KM_HERMES_p_2009.csv'
SIDIS_FFs_KM_HERMES_p_2020='../Calc_Grids/SIDIS_FFs/FF_KM_HERMES_p_2020.csv'
SIDIS_FFs_KM_COMPASS_d_2009='../Calc_Grids/SIDIS_FFs/FF_KM_COMPASS_d_2009.csv'
SIDIS_FFs_KM_COMPASS_p_2015='../Calc_Grids/SIDIS_FFs/FF_KM_COMPASS_p_2015.csv'


# SIDIS_FFs_HERMES_p_2009=(SIDIS_FFs_PiP_HERMES_p_2009,SIDIS_FFs_PiM_HERMES_p_2009,SIDIS_FFs_Pi0_HERMES_p_2009,SIDIS_FFs_KP_HERMES_p_2009,SIDIS_FFs_KM_HERMES_p_2009)
SIDIS_FFs_HERMES_p_2020=(SIDIS_FFs_PiP_HERMES_p_2020,SIDIS_FFs_PiM_HERMES_p_2020,SIDIS_FFs_Pi0_HERMES_p_2020,SIDIS_FFs_KP_HERMES_p_2020,SIDIS_FFs_KM_HERMES_p_2020)
SIDIS_FFs_COMPASS_d_2009=(SIDIS_FFs_PiP_COMPASS_d_2009,SIDIS_FFs_PiM_COMPASS_d_2009,SIDIS_FFs_Pi0_COMPASS_d_2009,SIDIS_FFs_KP_COMPASS_d_2009,SIDIS_FFs_KM_COMPASS_d_2009)
SIDIS_FFs_COMPASS_p_2015=(SIDIS_FFs_PiP_COMPASS_p_2015,SIDIS_FFs_PiM_COMPASS_p_2015,SIDIS_FFs_Pi0_COMPASS_p_2015,SIDIS_FFs_KP_COMPASS_p_2015,SIDIS_FFs_KM_COMPASS_p_2015)


###########################################################################
#####################  DY PDFs #########################################
###########################################################################

DY_PDFs_COMPASS_p_2017_x1='../Calc_Grids/DY_PDFs/PDFs_x1_COMPASS_p_DY_2017.csv'
DY_PDFs_COMPASS_p_2017_x2='../Calc_Grids/DY_PDFs/PDFs_x2_COMPASS_p_DY_2017.csv'


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

def SIDIS_xFxQ2(datafile,flavor,hadron,dep):
    tempvals_all=pd.read_csv(datafile)
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
    tempvals_all=pd.read_csv(datafile)
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



def Asymmetry_for_Hadron(SIDISdatafilename,hadron,dep,m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar):
    kperp2Avg=Kp2A
    pperpAvg=Pp2A
    eCharg=ee
    # if(SIDISdatafilename==SIDIS_DataFilesArray[0]):
    #     PDFfile=SIDIS_PDFs_Array[0]
    #     if(hadron=='pi+'):
    #         FFfile=SIDIS_FFs_HERMES_p_2009[0]
    #     elif(hadron=='pi-'):
    #         FFfile=SIDIS_FFs_HERMES_p_2009[1]
    #     elif(hadron=='pi0'):
    #         FFfile=SIDIS_FFs_HERMES_p_2009[2]                      
    #     elif(hadron=='k+'):
    #         FFfile=SIDIS_FFs_HERMES_p_2009[3]                      
    #     elif(hadron=='k-'):
    #         FFfile=SIDIS_FFs_HERMES_p_2009[4]                      
    if(SIDISdatafilename==SIDIS_DataFilesArray[0]):
        PDFfile=SIDIS_PDFs_Array[0]
        if(hadron=='pi+'):
            FFfile=SIDIS_FFs_HERMES_p_2020[0]
        elif(hadron=='pi-'):
            FFfile=SIDIS_FFs_HERMES_p_2020[1]
        elif(hadron=='pi0'):
            FFfile=SIDIS_FFs_HERMES_p_2020[2]                      
        elif(hadron=='k+'):
            FFfile=SIDIS_FFs_HERMES_p_2020[3]                      
        elif(hadron=='k-'):
            FFfile=SIDIS_FFs_HERMES_p_2020[4]                      
    elif(SIDISdatafilename==SIDIS_DataFilesArray[1]):
        PDFfile=SIDIS_PDFs_Array[1]
        if(hadron=='pi+'):
            FFfile=SIDIS_FFs_COMPASS_d_2009[0]
        elif(hadron=='pi-'):
            FFfile=SIDIS_FFs_COMPASS_d_2009[1]
        elif(hadron=='pi0'):
            FFfile=SIDIS_FFs_COMPASS_d_2009[2]                      
        elif(hadron=='k+'):
            FFfile=SIDIS_FFs_COMPASS_d_2009[3]                      
        elif(hadron=='k-'):
            FFfile=SIDIS_FFs_COMPASS_d_2009[4]                      
    elif(SIDISdatafilename==SIDIS_DataFilesArray[2]):
        PDFfile=SIDIS_PDFs_Array[2]
        if(hadron=='pi+'):
            FFfile=SIDIS_FFs_COMPASS_p_2015[0]
        elif(hadron=='pi-'):
            FFfile=SIDIS_FFs_COMPASS_p_2015[1]
        elif(hadron=='pi0'):
            FFfile=SIDIS_FFs_COMPASS_p_2015[2]                      
        elif(hadron=='k+'):
            FFfile=SIDIS_FFs_COMPASS_p_2015[3]                      
        elif(hadron=='k-'):
            FFfile=SIDIS_FFs_COMPASS_p_2015[4]                             
    tempvals_all=pd.read_csv(PDFfile)
    #tempvals=tempvals_all[(tempvals_all["hadron"]==hadron)]
    tempvals=tempvals_all[(tempvals_all["hadron"]==hadron)&(tempvals_all["1D_dependence"]==dep)]
    # QQ=tempvals['QQ']
    x=tempvals['x']
    z=tempvals['z']
    phT=tempvals['phT']
    uCont1= NNq(x,Nu,alphau,betau)*(eU**2)*SIDIS_xFxQ2(PDFfile,2,hadron,dep)*SIDIS_zFzQ(FFfile,2,hadron,dep)
    ubarCont1= NNqbar(x,Nubar)*(eUbar**2)*SIDIS_xFxQ2(PDFfile,-2,hadron,dep)*SIDIS_zFzQ(FFfile,-2,hadron,dep)
    dCont1= NNq(x,Nd,alphad,betad)*(eD**2)*SIDIS_xFxQ2(PDFfile,1,hadron,dep)*SIDIS_zFzQ(FFfile,1,hadron,dep)
    dbarCont1= NNqbar(x,Ndbar)*(eDbar**2)*SIDIS_xFxQ2(PDFfile,-1,hadron,dep)*SIDIS_zFzQ(FFfile,-1,hadron,dep)
    sCont1= NNq(x,Ns,alphas,betas)*(eS**2)*SIDIS_xFxQ2(PDFfile,3,hadron,dep)*SIDIS_zFzQ(FFfile,3,hadron,dep)
    sbarCont1= NNqbar(x,Nsbar)*(eSbar**2)*SIDIS_xFxQ2(PDFfile,-3,hadron,dep)*SIDIS_zFzQ(FFfile,-3,hadron,dep)
    uCont2= (eU**2)*SIDIS_xFxQ2(PDFfile,2,hadron,dep)*SIDIS_zFzQ(FFfile,2,hadron,dep)
    ubarCont2= (eUbar**2)*SIDIS_xFxQ2(PDFfile,-2,hadron,dep)*SIDIS_zFzQ(FFfile,-2,hadron,dep)
    dCont2= (eD**2)*SIDIS_xFxQ2(PDFfile,1,hadron,dep)*SIDIS_zFzQ(FFfile,1,hadron,dep)
    dbarCont2=(eDbar**2)*SIDIS_xFxQ2(PDFfile,-1,hadron,dep)*SIDIS_zFzQ(FFfile,-1,hadron,dep)
    sCont2= (eS**2)*SIDIS_xFxQ2(PDFfile,3,hadron,dep)*SIDIS_zFzQ(FFfile,3,hadron,dep)
    sbarCont2= (eSbar**2)*SIDIS_xFxQ2(PDFfile,-3,hadron,dep)*SIDIS_zFzQ(FFfile,-3,hadron,dep)
    tempNumerator = uCont1 + ubarCont1 +dCont1 + dbarCont1 + sCont1 + sbarCont1
    tempDenominator = uCont2 + ubarCont2 +dCont2 + dbarCont2 + sCont2 + sbarCont2
    tempASiv_Hadron = A0(z,phT,m1,kperp2Avg,pperpAvg,eCharg)*(tempNumerator/tempDenominator)
    return tempASiv_Hadron
    #return tempASiv_Hadron
     

# def Asymmetry(PDFfile,FFfile,QQ,x,z,pht,m1,Nu,au,bu,Nubar,Nd,ad,bd,Ndbar,Ns,aS,bs,Nsbar,lhaqID,lhaqbarID):
#     kperp2Avg=Kp2A
#     pperpAvg=Pp2A
#     eCharg=ee
#     if((lhaqID==2)and(lhaqbarID==-1)):
#         ### This is pi+
#         tempASiv = Asymmetry_for_Hadron(PDFfile,FFfile,QQ,x,z,pht,m1,Nu,au,bu,Nubar,Nd,ad,bd,Ndbar,Ns,aS,bs,Nsbar)
#     elif((lhaqID==1)and(lhaqbarID==-2)):
#         ### This is pi-
#         tempASiv = A0(z,pht,m1,kperp2Avg,pperpAvg,eCharg)*(tempNumerator/tempDenominator)
#     elif((lhaqID==2)and(lhaqbarID==-3)):
#         ### This is k+
#         tempASiv = A0(z,pht,m1,kperp2Avg,pperpAvg,eCharg)*(tempNumerator/tempDenominator)
#     elif((lhaqID==3)and(lhaqbarID==-2)):
#         ### This is k+
#         tempASiv = A0(z,pht,m1,kperp2Avg,pperpAvg,eCharg)*(tempNumerator/tempDenominator)
#     return tempASiv


# def ASivFitHadron(filename,hadron,KV,**parms):
#     m1= parms["m1"]
#     Nu = parms["Nu"]
#     alphau= parms["alphau"]
#     betau = parms["betau"]
#     Nubar = parms["Nubar"]
#     Nd = parms["Nd"]
#     alphad= parms["alphad"]
#     betad = parms["betad"]
#     Ndbar = parms["Ndbar"]
#     Ns = parms["Ns"]
#     alphas= parms["alphas"]
#     betas = parms["betas"]
#     Nsbar = parms["Nsbar"]
#     ################
#     QQ,x,z,pht=KV
#     array_size=len(x)
#     tempASivHad_val=[]
#     for i in range(0,array_size):
#         tempASivHad=Asymmetry_for_Hadron(filename,hadron,m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)
#         tempASivHad_val.append(tempASivHad)
#     return tempASivHad_val

#Asymmetry_for_Hadron(SIDISdatafilename,hadron,dep,m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar)

def totalfitDataSet(datfile,m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar):
    had_len=len(hadarray(datfile))
    temHads=hadarray(datfile)
    fittot=[]
    array = []
    df = pd.DataFrame(array, columns = ['hadron','y','pht','x','z','Siv'], index = ['pi+','pi-','p0','k+','k-'])
    for i in range(0,had_len):
        if temHads[i]=="pi+":
            tempfitx=Asymmetry_for_Hadron(datfile,"pi+","x",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            df.append({'x':tempfitx}, ignore_index=True)
            tempfitz=Asymmetry_for_Hadron(datfile,"pi+","z",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            df.append({'z':tempfitz}, ignore_index=True)
            tempfitphT=Asymmetry_for_Hadron(datfile,"pi+","phT",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            df.append({'pht':tempfitphT}, ignore_index=True)
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT))
            fittot.append(tempfit)
        elif temHads[i]=="pi-":
            tempfitx=Asymmetry_for_Hadron(datfile,"pi-","x",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            tempfitz=Asymmetry_for_Hadron(datfile,"pi-","z",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            tempfitphT=Asymmetry_for_Hadron(datfile,"pi-","phT",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT))
            fittot.append(tempfit)
        elif temHads[i]=="pi0":
            tempfitx=Asymmetry_for_Hadron(datfile,"pi0","x",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            tempfitz=Asymmetry_for_Hadron(datfile,"pi0","z",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            tempfitphT=Asymmetry_for_Hadron(datfile,"pi0","phT",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT))
            fittot.append(tempfit)
        elif temHads[i]=="k+":
            tempfitx=Asymmetry_for_Hadron(datfile,"k+","x",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            tempfitz=Asymmetry_for_Hadron(datfile,"k+","z",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            tempfitphT=Asymmetry_for_Hadron(datfile,"k+","phT",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT))
            fittot.append(tempfit)
        elif temHads[i]=="k-":
            tempfitx=Asymmetry_for_Hadron(datfile,"k-","x",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            tempfitz=Asymmetry_for_Hadron(datfile,"k-","z",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            tempfitphT=Asymmetry_for_Hadron(datfile,"k-","phT",m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT))
            fittot.append(tempfit)
        df = pd.DataFrame(fittot, columns = ['hadron','y','pht','x','z','Siv'] ,index = ['pi+','pi-','p0','k+','k-'])
        df.to_csv('Test.csv')
    return np.concatenate((fittot), axis=None)
    
    
totalfitDataSet(Dat2,m1=M1_t2,Nu=NU_t2,alphau=AlphaU_t2,betau=BetaU_t2,Nubar=NUbar_t2,Nd=ND_t2,alphad=AlphaD_t2,betad=BetaD_t2,Ndbar=NDbar_t2,Ns=NS_t2,alphas=AlphaS_t2,betas=BetaS_t2,Nsbar=NSbar_t2)
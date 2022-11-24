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

PDF_HERMES_2009 = pd.read_csv(SIDIS_PDFs_HERMES_p_2009)
PDF_HERMES_2020 = pd.read_csv(SIDIS_PDFs_HERMES_p_2020)
PDF_COMPASS_2009 = pd.read_csv(SIDIS_PDFs_COMPASS_d_2009)
PDF_COMPASS_2015 = pd.read_csv(SIDIS_PDFs_COMPASS_p_2015)
#PDF_HERMES_2020
PDFs_Array = (PDF_HERMES_2020,PDF_COMPASS_2015)
###########################################################################
#####################  SIDIS FFs #########################################
###########################################################################

FF_HERMES_PiP_2009 = pd.read_csv(SIDIS_FFs_PiP_HERMES_p_2009)
FF_HERMES_PiP_2020 = pd.read_csv(SIDIS_FFs_PiP_HERMES_p_2020)
FF_COMPASS_PiP_2009 = pd.read_csv(SIDIS_FFs_PiP_COMPASS_d_2009)
FF_COMPASS_PiP_2015 = pd.read_csv(SIDIS_FFs_PiP_COMPASS_p_2015)

FF_HERMES_PiM_2009 = pd.read_csv(SIDIS_FFs_PiM_HERMES_p_2009)
FF_HERMES_PiM_2020 = pd.read_csv(SIDIS_FFs_PiM_HERMES_p_2020)
FF_COMPASS_PiM_2009 = pd.read_csv(SIDIS_FFs_PiM_COMPASS_d_2009)
FF_COMPASS_PiM_2015 = pd.read_csv(SIDIS_FFs_PiM_COMPASS_p_2015)

FF_HERMES_Pi0_2009 = pd.read_csv(SIDIS_FFs_Pi0_HERMES_p_2009)
FF_HERMES_Pi0_2020 = pd.read_csv(SIDIS_FFs_Pi0_HERMES_p_2020)
FF_COMPASS_Pi0_2009 = pd.read_csv(SIDIS_FFs_Pi0_COMPASS_d_2009)
FF_COMPASS_Pi0_2015 = pd.read_csv(SIDIS_FFs_Pi0_COMPASS_p_2015)

FF_HERMES_KP_2009 = pd.read_csv(SIDIS_FFs_KP_HERMES_p_2009)
FF_HERMES_KP_2020 = pd.read_csv(SIDIS_FFs_KP_HERMES_p_2020)
FF_COMPASS_KP_2009 = pd.read_csv(SIDIS_FFs_KP_COMPASS_d_2009)
FF_COMPASS_KP_2015 = pd.read_csv(SIDIS_FFs_KP_COMPASS_p_2015)

FF_HERMES_KM_2009 = pd.read_csv(SIDIS_FFs_KM_HERMES_p_2009)
FF_HERMES_KM_2020 = pd.read_csv(SIDIS_FFs_KM_HERMES_p_2020)
FF_COMPASS_KM_2009 = pd.read_csv(SIDIS_FFs_KM_COMPASS_d_2009)
FF_COMPASS_KM_2015 = pd.read_csv(SIDIS_FFs_KM_COMPASS_p_2015)

FFs_HERMES_2009=(FF_HERMES_PiP_2009,FF_HERMES_PiM_2009,FF_HERMES_Pi0_2009,FF_HERMES_KP_2009,FF_HERMES_KM_2009)
FFs_HERMES_2020=(FF_HERMES_PiP_2020,FF_HERMES_PiM_2020,FF_HERMES_Pi0_2020,FF_HERMES_KP_2020,FF_HERMES_KM_2020)
FFs_COMPASS_2009=(FF_COMPASS_PiP_2009,FF_COMPASS_PiM_2009,FF_COMPASS_Pi0_2009,FF_COMPASS_KP_2009,FF_COMPASS_KM_2009)
FFs_COMPASS_2015=(FF_COMPASS_PiP_2015,FF_COMPASS_PiM_2015,FF_COMPASS_Pi0_2015,FF_COMPASS_KP_2015,FF_COMPASS_KM_2015)

SIDIS_FFs_Data=[None]*(len(SIDIS_DataFilesArrayR))
#SIDIS_FFs_Data=[None]*(4)
#SIDIS_FFs_Data[0]=(FF_HERMES_PiP_2009,FF_HERMES_PiM_2009,FF_HERMES_Pi0_2009,FF_HERMES_KP_2009,FF_HERMES_KM_2009)
SIDIS_FFs_Data[0]=(FF_HERMES_PiP_2020,FF_HERMES_PiM_2020,FF_HERMES_Pi0_2020,FF_HERMES_KP_2020,FF_HERMES_KM_2020)
#SIDIS_FFs_Data[1]=(FF_COMPASS_PiP_2009,FF_COMPASS_PiM_2009,FF_COMPASS_Pi0_2009,FF_COMPASS_KP_2009,FF_COMPASS_KM_2009)
SIDIS_FFs_Data[1]=(FF_COMPASS_PiP_2015,FF_COMPASS_PiM_2015,FF_COMPASS_Pi0_2015,FF_COMPASS_KP_2015,FF_COMPASS_KM_2015)

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


def Asymmetry_for_Hadron(SIDISdatafilename,hadron,dep,**parms):
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
    alphasb= parms["asb"]
    betasb = parms["bsb"]
    kperp2Avg=Kp2A
    pperpAvg=Pp2A
    eCharg=ee
    PDFfile=Determine_PDFs_FFs(SIDISdatafilename,hadron)[0]
    FFfile=Determine_PDFs_FFs(SIDISdatafilename,hadron)[1]
    tempvals_all=PDFfile
    #tempvals=tempvals_all[(tempvals_all["hadron"]==hadron)]
    tempvals=tempvals_all[(tempvals_all["hadron"]==hadron)&(tempvals_all["1D_dependence"]==dep)]
    # QQ=tempvals['QQ']
    x=tempvals['x']
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
    tempASiv_Hadron = A0(z,phT,m1,kperp2Avg,pperpAvg,eCharg)*(tempNumerator/tempDenominator)
    return tempASiv_Hadron
    #return tempASiv_Hadron
     

def totalfitDataSet(datfile,**parms):
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
    asb= parms["asb"]
    bsb = parms["bsb"]
    #had_len=len(hadarray(datfile))
    temHads=hadarray(datfile)
    had_len=len(temHads)
    fittot=[]
    for i in range(0,had_len):
        if temHads[i]=="pi+":
            tempfitx=Asymmetry_for_Hadron(datfile,"pi+","x",**parms)
            tempfitz=Asymmetry_for_Hadron(datfile,"pi+","z",**parms)
            tempfitphT=Asymmetry_for_Hadron(datfile,"pi+","phT",**parms)
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT))
            fittot.append(tempfit)
        elif temHads[i]=="pi-":
            tempfitx=Asymmetry_for_Hadron(datfile,"pi-","x",**parms)
            tempfitz=Asymmetry_for_Hadron(datfile,"pi-","z",**parms)
            tempfitphT=Asymmetry_for_Hadron(datfile,"pi-","phT",**parms)
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT))
            fittot.append(tempfit)
        elif temHads[i]=="pi0":
            tempfitx=Asymmetry_for_Hadron(datfile,"pi0","x",**parms)
            tempfitz=Asymmetry_for_Hadron(datfile,"pi0","z",**parms)
            tempfitphT=Asymmetry_for_Hadron(datfile,"pi0","phT",**parms)
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT))
            fittot.append(tempfit)
        elif temHads[i]=="k+":
            tempfitx=Asymmetry_for_Hadron(datfile,"k+","x",**parms)
            tempfitz=Asymmetry_for_Hadron(datfile,"k+","z",**parms)
            tempfitphT=Asymmetry_for_Hadron(datfile,"k+","phT",**parms)
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT))
            fittot.append(tempfit)
        elif temHads[i]=="k-":
            tempfitx=Asymmetry_for_Hadron(datfile,"k-","x",**parms)
            tempfitz=Asymmetry_for_Hadron(datfile,"k-","z",**parms)
            tempfitphT=Asymmetry_for_Hadron(datfile,"k-","phT",**parms)
            tempfit=np.concatenate((tempfitx,tempfitz,tempfitphT))
            fittot.append(tempfit)
    return np.concatenate((fittot), axis=None)


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









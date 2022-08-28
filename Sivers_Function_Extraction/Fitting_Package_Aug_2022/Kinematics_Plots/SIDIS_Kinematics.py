#######################################################################
############          SIDIS Definitions               #################
############        Written by Ishara Fernando        #################
############ Last upgrade: February-01-2022 ###########################
#######################################################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




######################################################
########## SIDIS Asymmetry (Theory) ##################
######################################################



#### Asymmetry for given hadron in SIDIS ######


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

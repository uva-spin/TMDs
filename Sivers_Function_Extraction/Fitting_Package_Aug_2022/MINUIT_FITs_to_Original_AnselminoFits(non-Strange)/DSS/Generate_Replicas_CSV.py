import os
import functions_develop
import pandas as pd
from Input_Parameterization import *


import copy


def chi2(y, yhat, err):
    return np.sum(((y-yhat)/err)**2)

def Create_SIDIS_P_Data(datafile, m1, Nu, au, bu, Nub, Nd, ad, bd, Ndb):
    tempdf=pd.read_csv(datafile)
    temphad=np.array(tempdf['hadron'],dtype=object)
    tempQ2=np.array(tempdf['Q2'],dtype=object)
    tempX=np.array(tempdf['x'],dtype=object)
    tempY=np.array(tempdf['y'],dtype=object)
    tempZ=np.array(tempdf['z'],dtype=object)
    tempPHT=np.array(tempdf['phT'],dtype=object)
    tempSivData=np.array(tempdf['Siv'],dtype=object)
    tempSivErr=np.array(tempdf['tot_err'],dtype=object)
    tempDEP=np.array(tempdf['1D_dependence'],dtype=object)
    data_dictionary={"hadron":[],"Q2":[],"x":[],"y":[],"z":[],"phT":[],"Siv":[],"tot_err":[],"1D_dependence":[]}
    data_dictionary["hadron"]=temphad
    data_dictionary["Q2"]=tempQ2
    data_dictionary["x"]=tempX
    data_dictionary["y"]=tempY
    data_dictionary["z"]=tempZ
    data_dictionary["phT"]=tempPHT
    data_dictionary["tot_err"]=tempSivErr
    data_dictionary["1D_dependence"]=tempDEP
    PiP=copy.deepcopy(data_dictionary)
    PiM=copy.deepcopy(data_dictionary)
    Pi0=copy.deepcopy(data_dictionary)
    KP=copy.deepcopy(data_dictionary)
    KM=copy.deepcopy(data_dictionary)
    SivHad=functions_develop.Sivers_Hadron()
    ############################################
    temp_Siv=[]
    temp_Siv_Org_Data=[]
    for i in range(len(temphad)):
        temp=np.array([[data_dictionary["x"][i],data_dictionary["z"][i],
                        data_dictionary["phT"][i],data_dictionary["Q2"][i]]])
        temp_had=data_dictionary["hadron"][i]
        #print(temp_had)
        temp_Siv.append(SivHad.sivers(temp_had,temp, m1, Nu, au, bu, Nub, Nd, ad, bd, Ndb)[0])
        temp_Siv_Org_Data.append(tempSivData)
    ############################################
    data_dictionary["Siv"]=np.array(temp_Siv)
    data_dictionary["Siv_Data"]=np.array(temp_Siv_Org_Data)
    NdataPoints = len(data_dictionary["Siv"])
    Tempchi2 = chi2(data_dictionary["Siv"],data_dictionary["Siv_Data"],data_dictionary["tot_err"])
    return pd.DataFrame(data_dictionary),Tempchi2, NdataPoints


def param_samples_new(datafilesarray,pars, pars_err,Nsamples):
    par_sample = []
    temp_ch2 = 
    for i in range(0,Nsamples):
        temp_pars = np.random.normal(pars, pars_err)
        for j in range(len(datafilesarray)):


    org_theory=totalfitfunc(datafilesarray,*pars)
    org_chi2=totchi2(SiversVals(datafilesarray),org_theory,SiversErrVals(datafilesarray))
    #print(org_chi2)
    #temp_pars = np.random.normal(pars, 0.03*np.array(err))
    #temp_pars = np.random.normal(pars, np.diag(err))
    #print(temp_pars)
    #print(len(SiversVals(datafilesarray)))
    for i in range(100):
        temp_pars = np.random.normal(pars, err)
        print(temp_pars)
        par_sample.append(temp_pars) 
        temp_theory=totalfitfunc(datafilesarray,*temp_pars)
        temp_chi2=totchi2(SiversVals(datafilesarray),temp_theory,SiversErrVals(datafilesarray))
        #print(temp_chi2)
        if(np.abs(org_chi2-temp_chi2)<len(SiversVals(datafilesarray))):
            par_sample.append(temp_pars)
    return np.array(par_sample)

Pseudo_SIDIS_HERMES2009=Create_SIDIS_P_Data(Dat1,m1v,Nuv,auv,buv,Nubv,Ndv,adv,bdv,Ndbv)
Pseudo_SIDIS_HERMES2020=Create_SIDIS_P_Data(Dat2,m1v,Nuv,auv,buv,Nubv,Ndv,adv,bdv,Ndbv)
Pseudo_SIDIS_COMPASS2009=Create_SIDIS_P_Data(Dat3,m1v,Nuv,auv,buv,Nubv,Ndv,adv,bdv,Ndbv)
Pseudo_SIDIS_COMPASS2015=Create_SIDIS_P_Data(Dat4,m1v,Nuv,auv,buv,Nubv,Ndv,adv,bdv,Ndbv)

def Create_DY_Data(datafile, m1, Nu, au, bu, Nub, Nd, ad, bd, Ndb):
    tempdf=pd.read_csv(datafile)
    tempDEP=np.array(tempdf['Dependence'],dtype=object)
    tempX1=np.array(tempdf['x1'],dtype=object)
    tempX2=np.array(tempdf['x2'],dtype=object)
    tempXF=np.array(tempdf['xF'],dtype=object)
    tempQT=np.array(tempdf['QT'],dtype=object)
    tempQM=np.array(tempdf['QM'],dtype=object)
    tempSivErr=np.array(tempdf['tot_err'],dtype=object)
    data_dictionary={"Dependence":[],"x1":[],"x2":[],"xF":[],"QT":[],"QM":[],"Siv":[],"tot_err":[]}
    data_dictionary["Dependence"]=tempDEP
    data_dictionary["x1"]=tempX1
    data_dictionary["x2"]=tempX2
    data_dictionary["xF"]=tempXF
    data_dictionary["QT"]=tempQT
    data_dictionary["QM"]=tempQM
    data_dictionary["tot_err"]=tempSivErr
    PiP=copy.deepcopy(data_dictionary)
    PiM=copy.deepcopy(data_dictionary)
    Pi0=copy.deepcopy(data_dictionary)
    KP=copy.deepcopy(data_dictionary)
    KM=copy.deepcopy(data_dictionary)
    SivDY=functions_develop.Sivers_DY()
    ############################################
    temp_Siv=[]
    for i in range(len(tempDEP)):
        temp=np.array([[data_dictionary["x1"][i],data_dictionary["x2"][i],
                        data_dictionary["QT"][i],data_dictionary["QM"][i]]])
        temp_Siv.append(SivDY.sivers(temp, m1, Nu, au, bu, Nub, Nd, ad, bd, Ndb)[0])
    ############################################
    data_dictionary["Siv"]=np.array(temp_Siv)
    return pd.DataFrame(data_dictionary)

Pseudo_DY_COMPASS2017=Create_DY_Data(DY_DataFilesArray[0],m1v,Nuv,auv,buv,Nubv,Ndv,adv,bdv,Ndbv,Nsv,asv,bsv,Nsbv)


OutputFolder='Pseudo_Data_Set'
os.mkdir(OutputFolder)

Pseudo_SIDIS_HERMES2009.to_csv(str(OutputFolder)+'/Pseudo_SIDIS_HERMES2009.csv')
Pseudo_SIDIS_HERMES2020.to_csv(str(OutputFolder)+'/Pseudo_SIDIS_HERMES2020.csv')
Pseudo_SIDIS_COMPASS2009.to_csv(str(OutputFolder)+'/Pseudo_SIDIS_COMPASS2009.csv')
Pseudo_SIDIS_COMPASS2015.to_csv(str(OutputFolder)+'/Pseudo_SIDIS_COMPASS2015.csv')

Pseudo_DY_COMPASS2017.to_csv(str(OutputFolder)+'/Pseudo_DY_COMPASS2017.csv')
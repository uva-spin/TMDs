import lhapdf
import numpy as np
import pandas as pd
from Sivers_SIDIS_Definitions_Parm2 import *
import os

#parms_df = pd.read_csv('Parameters.csv')
test_pars=([7.0, 0.89, 2.50, 20, -0.12, 0.4,  20, -2.4, 2.7, 17, -0.7, 1.5, 20, -20, 4.7, 2.3, 20, 9.5, 20])
test_errs=([  4, 0.06, 0.11,  2,  0.06, 0.35, 16,  0.4, 0.6,  4,  0.5, 0.6, 17,  40, 3.0, 2.2,  5, 1.4, 14])
# These parameters gave chi2/N = 531.2/313 = 1.69 for MINUIT fit with 
# HERMES2009, COMPASS2009, COMPASS2015 data (HERMES2020 wasn't included)
#parms_df = pd.read_csv('./PseudoData/Parameters.csv')
#Parameters_array = parms_df.to_numpy()

N_Trial_Samples = 1000
Chi2_diff = 30.52

HERMES2009 = './Data/HERMES_p_2009.csv'
HERMES2020 = './Data/HERMES_p_2020.csv'
COMPASS2009 = './Data/COMPASS_d_2009.csv'
COMPASS2015 = './Data/COMPASS_p_2015.csv'

def Create_SIDIS_Asym_Data(datafile, m1, Nu, au, bu, Nub, aub, bub, Nd, ad, bd, Ndb,adb, bdb, Ns, aS, bS, Nsb, asb, bsb):
    tempdf=pd.read_csv(datafile)
    temphad=np.array(tempdf['hadron'],dtype=object)
    tempQ2=np.array(tempdf['Q2'],dtype=object)
    tempX=np.array(tempdf['x'],dtype=object)
    tempY=np.array(tempdf['y'],dtype=object)
    tempZ=np.array(tempdf['z'],dtype=object)
    tempPHT=np.array(tempdf['phT'],dtype=object)
    #tempSivData=np.array(tempdf['Siv'],dtype=object)
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
    temp_Siv = totalfitDataSet_All(datafile, m1=m1, Nu=Nu, au=au, bu=bu,
              Nub=Nub, aub=aub, bub=bub,
              Nd=Nd, ad=ad, bd=bd, 
              Ndb=Ndb, adb=adb, bdb=bdb,
              Ns=Ns,aS=aS,bS=bS,
              Nsb=Nsb, asb=asb, bsb=bsb)
    ############################################
    data_dictionary["Siv"]=temp_Siv
    return pd.DataFrame(data_dictionary)
    


OutputFolder = 'PseudoData_Parm2'
os.mkdir(OutputFolder)

R_SIDIS_HERMES2009=Create_SIDIS_Asym_Data(HERMES2009, *test_pars)
R_SIDIS_HERMES2009.to_csv(str(OutputFolder)+'/'+'HERMES2009_Pseudo.csv')

R_SIDIS_HERMES2020=Create_SIDIS_Asym_Data(HERMES2020, *test_pars)
R_SIDIS_HERMES2020.to_csv(str(OutputFolder)+'/'+'HERMES2020_Pseudo.csv')

R_SIDIS_COMPASS2009=Create_SIDIS_Asym_Data(COMPASS2009, *test_pars)
R_SIDIS_COMPASS2009.to_csv(str(OutputFolder)+'/'+'COMPASS2009_Pseudo.csv')

R_SIDIS_COMPASS2015=Create_SIDIS_Asym_Data(COMPASS2015, *test_pars)
R_SIDIS_COMPASS2015.to_csv(str(OutputFolder)+'/'+'COMPASS2015_Pseudo.csv')


def SIDIS_param_samples(pars, pars_err,Nsamples):
    par_sample = []
    chi2_array = []
    parm_dictionary={"m1":[],"Nu":[],"au":[],"bu":[],
    "Nub":[], "aub": [], "bub": [],
    "Nd":[],"ad":[],"bd":[],
    "Ndb":[], "adb": [], "bdb": [],
    "Ns":[],"aS":[],"bS":[],
    "Nsb":[], "asb": [], "bsb": []}
    m1a = []
    Nua = []
    aua = []
    bua = []
    Nuba = []
    auba = []
    buba = []
    Nda = []
    ada = []
    bda = []
    Ndba = []
    adba = []
    bdba = []
    Nsa = []
    aSa = []
    bSa = []
    Nsba = []
    asba = []
    bsba = []
    for i in range(0,Nsamples):
        temp_pars = np.random.normal(pars, pars_err)
        #print(temp_pars)
        temp_chi2_central = SIDIStotalchi2Minuit(*pars)
        temp_chi2_dist = SIDIStotalchi2Minuit(*temp_pars)
        temp_chi2_diff = np.abs(temp_chi2_central - temp_chi2_dist)
        print("checking chi2 on sample "+str(i)+" out of "+str(N_Trial_Samples)+ ":"+ str(temp_chi2_diff))
        nn = 0
        if (temp_chi2_diff <= Chi2_diff):
            nn = nn + 1
            m1a.append(temp_pars[0])
            Nua.append(temp_pars[1])
            aua.append(temp_pars[2])
            bua.append(temp_pars[3])
            Nuba.append(temp_pars[4])
            auba.append(temp_pars[5])
            buba.append(temp_pars[6])
            Nda.append(temp_pars[7])
            ada.append(temp_pars[8])
            bda.append(temp_pars[9])
            Ndba.append(temp_pars[10])
            adba.append(temp_pars[11])
            bdba.append(temp_pars[12])
            Nsa.append(temp_pars[13])
            aSa.append(temp_pars[14])
            bSa.append(temp_pars[15])
            Nsba.append(temp_pars[16])
            asba.append(temp_pars[17])
            bsba.append(temp_pars[18])
            #parm_dictionary.append(temp_pars)
        parm_dictionary["m1"] = np.array(m1a)
        parm_dictionary["Nu"] = np.array(Nua)
        parm_dictionary["au"] = np.array(aua)
        parm_dictionary["bu"] = np.array(bua)
        parm_dictionary["Nub"] = np.array(Nuba)
        parm_dictionary["aub"] = np.array(auba)
        parm_dictionary["bub"] = np.array(buba)
        parm_dictionary["Nd"] = np.array(Nda)
        parm_dictionary["ad"] = np.array(ada)
        parm_dictionary["bd"] = np.array(bda)
        parm_dictionary["Ndb"] = np.array(Ndba)
        parm_dictionary["adb"] = np.array(adba)
        parm_dictionary["bdb"] = np.array(adba)
        parm_dictionary["Ns"] = np.array(Nsa)
        parm_dictionary["aS"] = np.array(aSa)
        parm_dictionary["bS"] = np.array(bSa)
        parm_dictionary["Nsb"] = np.array(Nsba)
        parm_dictionary["asb"] = np.array(asba)
        parm_dictionary["bsb"] = np.array(bsba)
    return pd.DataFrame(parm_dictionary)


Paramters_DF = SIDIS_param_samples(test_pars, test_errs, N_Trial_Samples)
Paramters_DF.to_csv(str(OutputFolder)+'/'+'Parameters.csv')




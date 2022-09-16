#import os
#import functions_develop
import numpy as np
import pandas as pd
from Input_Parameterization import *
from Sivers_SIDIS_Definitions_for_Pseudo import *


test_pars=([7.0, 0.89, 2.78, 19.4, -0.07, -2.33, 2.5, 15.8, -0.29, -14, 4.9, 3, 0])
test_errs=([0.6, 0.05, 0.17, 1.6, 0.06, 0.31, 0.4, 3.2, 0.17, 10, 3.3, 2, 0.18])
N_Trial_Samples = 300
Chi2_diff = 17.21
OutputFolder = 'PseudoData'

def SIDIS_param_samples(pars, pars_err,Nsamples):
    par_sample = []
    chi2_array = []
    parm_dictionary={"m1":[],"Nu":[],"au":[],"bu":[],"Nub":[],
    "Nd":[],"ad":[],"bd":[],"Ndb":[], 
    "Ns":[],"aS":[],"bS":[],"Nsb":[]}
    m1a = []
    Nua = []
    aua = []
    bua = []
    Nuba = []
    Nda = []
    ada = []
    bda = []
    Ndba = []
    Nsa = []
    aSa = []
    bSa = []
    Nsba = []
    for i in range(0,Nsamples):
        temp_pars = np.random.normal(pars, pars_err)
        #print(temp_pars)
        temp_chi2_central = SIDIStotalchi2Minuit(*pars)
        temp_chi2_dist = SIDIStotalchi2Minuit(*temp_pars)
        temp_chi2_diff = np.abs(temp_chi2_central - temp_chi2_dist)
        print("checking chi2 on sample "+str(i)+" out of "+str(N_Trial_Samples))
        nn = 0
        if (temp_chi2_diff <= Chi2_diff):
            nn = nn + 1
            m1a.append(temp_pars[0])
            Nua.append(temp_pars[1])
            aua.append(temp_pars[2])
            bua.append(temp_pars[3])
            Nuba.append(temp_pars[4])
            Nda.append(temp_pars[5])
            ada.append(temp_pars[6])
            bda.append(temp_pars[7])
            Ndba.append(temp_pars[8])
            Nsa.append(temp_pars[9])
            aSa.append(temp_pars[10])
            bSa.append(temp_pars[11])
            Nsba.append(temp_pars[12])
            #parm_dictionary.append(temp_pars)
        parm_dictionary["m1"] = np.array(m1a)
        parm_dictionary["Nu"] = np.array(Nua)
        parm_dictionary["au"] = np.array(aua)
        parm_dictionary["bu"] = np.array(bua)
        parm_dictionary["Nub"] = np.array(Nuba)
        parm_dictionary["Nd"] = np.array(Nda)
        parm_dictionary["ad"] = np.array(ada)
        parm_dictionary["bd"] = np.array(bda)
        parm_dictionary["Ndb"] = np.array(Ndba)
        parm_dictionary["Ns"] = np.array(Nsa)
        parm_dictionary["aS"] = np.array(aSa)
        parm_dictionary["bS"] = np.array(bSa)
        parm_dictionary["Nsb"] = np.array(Nsba)
    return pd.DataFrame(parm_dictionary)


Paramters_DF = SIDIS_param_samples(test_pars, test_errs, N_Trial_Samples)
Paramters_DF.to_csv(str(OutputFolder)+'/'+'Parameters.csv')

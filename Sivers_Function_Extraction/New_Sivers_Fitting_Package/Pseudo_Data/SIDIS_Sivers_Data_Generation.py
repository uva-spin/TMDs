#import tensorflow as tf
import pandas as pd
import numpy as np
import lhapdf
import matplotlib.pyplot as plt

from Sivers_SIDIS_Definitions import *
from Input_Parameterization import *


import copy
def Create_SIDIS_P_Data(datafile,m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar):
    tempdf=pd.read_csv(datafile)
    temphad=np.array(tempdf['hadron'],dtype=object)
    tempQ2=np.array(tempdf['Q2'],dtype=object)
    tempX=np.array(tempdf['x'],dtype=object)
    tempY=np.array(tempdf['y'],dtype=object)
    tempZ=np.array(tempdf['z'],dtype=object)
    tempPHT=np.array(tempdf['phT'],dtype=object)
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
    #SivHad=functions_develop.Sivers_Hadron()
    ############################################
    temp_Siv=totalfitDataSet(datafile,m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
    #temp_Siv=np.random.normal
    ############################################
    data_dictionary["Siv"]=np.array(temp_Siv)
    return pd.DataFrame(data_dictionary)


Pseudo_SIDIS_HERMES2009=Create_SIDIS_P_Data(Dat1,m1=m1v,Nu=Nuv,alphau=auv,betau=buv,Nubar=Nubv,Nd=Ndv,alphad=adv,betad=bdv,Ndbar=Ndbv,Ns=Nsv,alphas=asv,betas=bsv,Nsbar=Nsbv)
Pseudo_SIDIS_HERMES2020=Create_SIDIS_P_Data(Dat2,m1=m1v,Nu=Nuv,alphau=auv,betau=buv,Nubar=Nubv,Nd=Ndv,alphad=adv,betad=bdv,Ndbar=Ndbv,Ns=Nsv,alphas=asv,betas=bsv,Nsbar=Nsbv)
Pseudo_SIDIS_COMPASS2009=Create_SIDIS_P_Data(Dat3,m1=m1v,Nu=Nuv,alphau=auv,betau=buv,Nubar=Nubv,Nd=Ndv,alphad=adv,betad=bdv,Ndbar=Ndbv,Ns=Nsv,alphas=asv,betas=bsv,Nsbar=Nsbv)
Pseudo_SIDIS_COMPASS2015=Create_SIDIS_P_Data(Dat4,m1=m1v,Nu=Nuv,alphau=auv,betau=buv,Nubar=Nubv,Nd=Ndv,alphad=adv,betad=bdv,Ndbar=Ndbv,Ns=Nsv,alphas=asv,betas=bsv,Nsbar=Nsbv)

# Pseudo_SIDIS_HERMES2009.to_csv('Pseudo_SIDIS_HERMES2009.csv')
# Pseudo_SIDIS_HERMES2020.to_csv('Pseudo_SIDIS_HERMES2020.csv')
# Pseudo_SIDIS_COMPASS2009.to_csv('Pseudo_SIDIS_COMPASS2009.csv')
# Pseudo_SIDIS_COMPASS2015.to_csv('Pseudo_SIDIS_COMPASS2015.csv')
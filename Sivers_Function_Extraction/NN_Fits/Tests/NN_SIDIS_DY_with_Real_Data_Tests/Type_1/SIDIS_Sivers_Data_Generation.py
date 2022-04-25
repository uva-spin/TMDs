######################################################################################
#################      Written by Ishara Fernando       ##############################
#################       February, 2022                  ##############################
######################################################################################

import tensorflow as tf
import pandas as pd
import numpy as np
import lhapdf
import matplotlib.pyplot as plt
### Make sure to change the Sivers Function formulation in the functions_devep.py ###
import functions_develop
from Global_Variables import *

Dat1='Data/HERMES_p_2009.csv'
Dat2='Data/HERMES_p_2020.csv'
Dat3='Data/COMPASS_d_2009.csv'
Dat4='Data/COMPASS_p_2015.csv'


import copy
def Create_SIDIS_Data(datafile,m1, Nu, au, bu, Nubar, Nd, ad, bd, Ndbar, NS, aS, bS, NSbar):
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
    SivHad=functions_develop.Sivers_Hadron()
    ############################################
    temp_Siv=[]
    for i in range(len(temphad)):
        temp=np.array([[data_dictionary["x"][i],data_dictionary["z"][i],
                        data_dictionary["phT"][i],data_dictionary["Q2"][i]]])
        temp_had=data_dictionary["hadron"][i]  
        temp_Siv.append(SivHad.sivers(temp_had,temp, m1, Nu, au, bu, Nubar, Nd, ad, bd, Ndbar, NS, aS, bS, NSbar)[0])
    ############################################
    data_dictionary["Siv"]=np.array(temp_Siv)
    return pd.DataFrame(data_dictionary)


Pseudo_SIDIS_HERMES2009=Create_SIDIS_Data(Dat1,m1v,Nuv,auv,buv,Nubv,Ndv,adv,bdv,Ndbv,Nsv,asv,bsv,Nsbv)
Pseudo_SIDIS_HERMES2020=Create_SIDIS_Data(Dat2,m1v,Nuv,auv,buv,Nubv,Ndv,adv,bdv,Ndbv,Nsv,asv,bsv,Nsbv)
Pseudo_SIDIS_COMPASS2009=Create_SIDIS_Data(Dat3,m1v,Nuv,auv,buv,Nubv,Ndv,adv,bdv,Ndbv,Nsv,asv,bsv,Nsbv)
Pseudo_SIDIS_COMPASS2015=Create_SIDIS_Data(Dat4,m1v,Nuv,auv,buv,Nubv,Ndv,adv,bdv,Ndbv,Nsv,asv,bsv,Nsbv)


Pseudo_SIDIS_HERMES2009.to_csv('Pseudo_Data/Pseudo_SIDIS_HERMES2009.csv')
Pseudo_SIDIS_HERMES2020.to_csv('Pseudo_Data/Pseudo_SIDIS_HERMES2020.csv')
Pseudo_SIDIS_COMPASS2009.to_csv('Pseudo_Data/Pseudo_SIDIS_COMPASS2009.csv')
Pseudo_SIDIS_COMPASS2015.to_csv('Pseudo_Data/Pseudo_SIDIS_COMPASS2015.csv')
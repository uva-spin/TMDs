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


DY_Dat1='Data/COMPASS_p_DY_2017.csv'

##### assumption for N(q) =  Nq x^a (1-x)^b ######
m1v = 1

Nuv = 1
auv = 0.2
buv = 0.2
Nubv = 0.5

Ndv = 1
adv = 0.2
bdv = 0.2
Ndbv = 0.5

Nsv = 1 
asv = 0.2
bsv = 0.2
Nsbv = 0.5


import copy
def Create_DY_Data(datafile,m1, Nu, au, bu, Nubar, Nd, ad, bd, Ndbar, NS, aS, bS, NSbar):
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
        temp_Siv.append(SivDY.sivers(temp, m1, Nu, au, bu, Nubar, Nd, ad, bd, Ndbar, NS, aS, bS, NSbar)[0])
    ############################################
    data_dictionary["Siv"]=np.array(temp_Siv)
    return pd.DataFrame(data_dictionary)


Pseudo_DY_COMPASS2017=Create_DY_Data(DY_Dat1,m1v,Nuv,auv,buv,Nubv,Ndv,adv,bdv,Ndbv,Nsv,asv,bsv,Nsbv)

Pseudo_DY_COMPASS2017.to_csv('Pseudo_Data/Pseudo_DY_COMPASS2017.csv')
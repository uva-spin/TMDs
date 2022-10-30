import lhapdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Input_Parameterization import *
from Sivers_SIDIS_Definitions import *
from Paths import *
from Constants import *

from iminuit import Minuit
import numpy as np


def totalchi2Minuit(m1,Nu,au,bu,Nub,aub,bub,Nd,ad,bd,Ndb,adb,bdb,Ns,aS,bS,Nsb,aSb,bSb):
    tempchi2=SIDIStotalchi2Minuit(m1=m1,Nu=Nu,au=au,bu=bu,Nub=Nub,aub=aub,bub=bub,Nd=Nd,ad=ad,bd=bd,
                                  Ndb=Ndb,adb=adb,bdb=bdb,Ns=Ns,aS=aS,bS=bS,Nsb=Nsb,asb=aSb,bsb=bSb)
    return tempchi2


Data_points_SIDIS = SIDIS_Data_points() 
Data_points_DY = DY_Data_points()
Total_data_points = Data_points_SIDIS 
#+ Data_points_DY

par_name_array=('m1','Nu','alphau','betau','Nubar','alphaub','betaub','Nd','alphad','betad','Ndbar','alphadb','betadb','Ns','alphas','betas','Nsbar','alphasb','betasb')

m1v=7.0
Nuv=0.89
auv=2.78
buv=19.4
Nubv=-0.07
aubv=0
bubv=0
Ndv=-2.33
adv=2.5
bdv=15.8
Ndbv=-0.29
adbv=0
bdbv=0
Nsv=14
asv=4.9
bsv=3
Nsbv=0.18
asbv=0
bsbv=0

from datetime import datetime

# datetime object containing current date and time
start = datetime.now()
print("start =", start)

def generate_file(n_array):
    ms = Minuit(totalchi2Minuit,m1=m1v,Nu=Nuv,au=auv,bu=buv,Nub=Nubv,aub=aubv,bub=bubv,
    Nd=Ndv,ad=adv,bd=bdv,Ndb=Ndbv,adb=adbv,bdb=bdbv,
    Ns=Nsv,aS=asv,bS=bsv,Nsb=Nsbv,aSb=asbv,bSb=bsbv,
    limit_m1=(3,7),limit_Ns=(-20,20), limit_Nsb = (-20,20),
    limit_au=(0,20),limit_bu=(0,20),limit_aub=(0, 20),limit_bub=(0,20),
    limit_ad=(0,20),limit_bd=(0,20),limit_adb=(0, 20),limit_bdb=(0,20),
    limit_aS=(0,20),limit_bS=(0,20),limit_aSb=(0, 20),limit_bSb=(0,20),
    errordef=1)
    ms.migrad()
    temp_df=pd.DataFrame({'parameter':[],'value':[],'error':[],'chi2':[],'N_data':[]})
    temp_val=[]
    temp_err=[]
    for i in range(0,len(n_array)):
        temp_val.append(ms.values[i])
        temp_err.append(ms.errors[i])
    temp_df['parameter'] = n_array
    temp_df['value'] = temp_val
    temp_df['error'] = temp_err
    temp_df['chi2'] = ms.fval
    temp_df['N_data'] = Total_data_points
    #return temp_df
    finish = datetime.now()
    print("finish =", finish)
    return temp_df.to_csv('Fit_Results.csv')

print(generate_file(par_name_array))
done = datetime.now()
print("done =", done)
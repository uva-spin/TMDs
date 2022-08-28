import lhapdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Input_Parameterization import *
from Sivers_SIDIS_Definitions import *
from Sivers_DY_Definitions import * 
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

m1v = 1.0

Nuv = 2.4
auv = 1.8
buv = 0.05

Nubv = 2.2
aubv = 2.1
bubv = 0.06

Ndv = 2.45
adv = 1.95
bdv = 0.07

Ndbv = -1.6
adbv = 2.6
bdbv = 0.04

Nsv = 0.4
asv = 1.8
bsv = 0.51

Nsbv = -0.07
asbv = 0.55
bsbv = 0.07

from datetime import datetime

# datetime object containing current date and time
start = datetime.now()
print("start =", start)

def generate_file(n_array):
    ms = Minuit(totalchi2Minuit,m1=m1v,Nu=Nuv,au=auv,bu=buv,Nub=Nubv,aub=aubv,bub=bubv,
    Nd=Ndv,ad=adv,bd=bdv,Ndb=Ndbv,adb=adbv,bdb=bdbv,
    Ns=Nsv,aS=asv,bS=bsv,Nsb=Nsbv,aSb=asbv,bSb=bsbv,
    limit_m1=(0.7,1.2),
    limit_au=(0,20),limit_bu=(0,20),limit_aub=(0, 20),limit_bub=(0,20),
    limit_ad=(0,20),limit_bd=(0,20),limit_adb=(0, 20),limit_bdb=(0,20),
    limit_aS=(0,20),limit_bS=(0,20),limit_aSb=(0, 20),limit_bSb=(0,20),
    errordef=1)
    #ms = Minuit(totalchi2Minuit,m1=M1_init,Nu=NU_init,alphau=AlphaU_init,betau=BetaU_init,Nubar=NUbar_init,Nd=ND_init,alphad=AlphaD_init,betad=BetaD_init,Ndbar=NDbar_init,Ns=NS_init,alphas=AlphaS_init,betas=BetaS_init,Nsbar=NSbar_init)
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
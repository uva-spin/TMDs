import lhapdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Input_Parameterization import *
from Sivers_SIDIS_Chi2_3D import *
#from Sivers_SIDIS_Chi2_R import *
#from PathsX import *
#from PathsR import *
from Constants import *

from iminuit import Minuit
import numpy as np

def totalchi2Minuit(m1,Nu,au,bu,Nub,aub,bub,Nd,ad,bd,Ndb,adb,bdb,Ns,aS,bS,Nsb,aSb,bSb):
    tempchi2=SIDIStotalchi2Minuit_3D(m1=m1,Nu=Nu,au=au,bu=bu,Nub=Nub,aub=aub,bub=bub,Nd=Nd,ad=ad,bd=bd,
                                  Ndb=Ndb,adb=adb,bdb=bdb,Ns=Ns,aS=aS,bS=bS,Nsb=Nsb,asb=aSb,bsb=bSb)   
    return tempchi2


Data_points_SIDIS = SIDIS_Data_points()
# + SIDIS_Data_points_R()
#Data_points_DY = DY_Data_points()
Total_data_points = Data_points_SIDIS 
#+ Data_points_DY

par_name_array=('m1','Nu','alphau','betau','Nubar','alphaub','betaub','Nd','alphad','betad','Ndbar','alphadb','betadb','Ns','alphas','betas','Nsbar','alphasb','betasb')

#m1v=7.0
#Nuv=0.89
#auv=2.75
#buv=20
#Nubv=-0.12
#aubv=0.4
#bubv=20
#Ndv=-2.2
#adv=3.0
#bdv=15.5
#Ndbv=-0.7
#adbv=1.5
#bdbv=15
#Nsv=-20
#asv=4.7
#bsv=2.3
#Nsbv=20
#asbv=9.5
#bsbv=20

par_array = np.array([7,0.89,2.50,20,-0.12,0.4,20,-2.4,2.7,17,-0.7,1.5,20,-20,4.7,2.3,20,9.5,20])

m1v=par_array[0]
Nuv=par_array[1]
auv=par_array[2]
buv=par_array[3]
Nubv=par_array[4]
aubv=par_array[5]
bubv=par_array[6]
Ndv=par_array[7]
adv=par_array[8]
bdv=par_array[9]
Ndbv=par_array[10]
adbv=par_array[11]
bdbv=par_array[12]
Nsv=par_array[13]
asv=par_array[14]
bsv=par_array[15]
Nsbv=par_array[16]
asbv=par_array[17]
bsbv=par_array[18]

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

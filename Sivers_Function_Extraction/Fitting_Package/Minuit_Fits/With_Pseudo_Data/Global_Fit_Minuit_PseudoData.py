import lhapdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Sivers_SIDIS_Definitions import *
from Sivers_DY_Definitions import * 

from iminuit import Minuit
import numpy as np

#### Here define the output file name ####
Results_FileName = 'original'


### Define Initial Starting points here ###
m1v = 6.31
Nuv = 0.75
auv = 2.68
buv = 14.71
Nubv = -0.094
Ndv = -1.32
adv = 1.39
bdv = 5.21
Ndbv = -0.01
Nsv = 12.08
asv = 0.91
bsv = 0.54
Nsbv = 0.25




from datetime import datetime

def totalchi2Minuit(m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar):
    tempchi2=SIDIStotalchi2Minuit(m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)+ DYtotalchi2Minuit(m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
    return tempchi2

Data_points_SIDIS = SIDIS_Data_points() 
Data_points_DY = DY_Data_points()
Total_data_points = Data_points_SIDIS + Data_points_DY

par_name_array=('m1','Nu','alphau','betau','Nubar','Nd','alphad','betad','Ndbar','Ns','alphas','betas','Nsbar')

# datetime object containing current date and time
start = datetime.now()
print("start =", start)

def generate_file(n_array):
    ms = Minuit(totalchi2Minuit,m1=m1v,Nu=Nuv,alphau=auv,betau=buv,Nubar=Nubv,
    Nd=Ndv,alphad=adv,betad=bdv,Ndbar=Ndbv,
    Ns=Nsv,alphas=asv,betas=bsv,Nsbar=Nsbv,
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
    return temp_df.to_csv('Fit_Results_'+str(Results_FileName)+'.csv')

print(generate_file(par_name_array))
done = datetime.now()
print("done =", done)
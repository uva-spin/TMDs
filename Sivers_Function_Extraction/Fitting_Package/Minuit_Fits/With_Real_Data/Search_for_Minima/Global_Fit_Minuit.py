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



def totalchi2Minuit(m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar):
    tempchi2=SIDIStotalchi2Minuit(m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)+ DYtotalchi2Minuit(m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
    return tempchi2


Data_points_SIDIS = SIDIS_Data_points() 
Data_points_DY = DY_Data_points()
Total_data_points = Data_points_SIDIS + Data_points_DY

par_name_array=('m1','Nu','alphau','betau','Nubar','Nd','alphad','betad','Ndbar','Ns','alphas','betas','Nsbar')



### Initial values ######

# m1v = 3.73
# auv=2.45
# buv=13
# adv=2.08
# bdv=7.8
# asv=0.421
# bsv=0.052
# Nuv=0.435
# Nubv=-0.1202
# Ndv=-0.89
# Ndbv=-0.10
# Nsv=5.76
# Nsbv=0.17


# m1v = 6.1
# Nuv = 0.72
# auv = 2.71
# buv = 15.05
# Nubv = -0.096
# Ndv = -1.30
# adv = 1.36
# bdv = 4.7
# Ndbv = -0.04
# Nsv = 12
# asv = 0.91
# bsv = 0.52
# Nsbv = 0.25

# m1v = 6.32
# Nuv = 0.75
# auv = 2.68
# buv = 14.71
# Nubv = -0.09
# Ndv = -1.32
# adv = 1.39
# bdv = 5.21
# Ndbv = -0.01
# Nsv = 12.08
# asv = 0.91
# bsv = 0.54
# Nsbv = 0.25

# m1v = 13.05
# Nuv = 1.51
# auv = 2.45
# buv = 13.38
# Nubv = -0.33
# Ndv = -3.15
# adv = 1.48
# bdv = 4.25
# Ndbv = -0.31
# Nsv = 12.64
# asv = 0.83
# bsv = 2.16
# Nsbv = 0.004

m1v = 17.71
Nuv = 2.04
auv = 2.62
buv = 14.97
Nubv = -0.14
Ndv = -3.23
adv = 1.15
bdv = 4.68
Ndbv = -0.10
Nsv = 12.96
asv = 1.92
bsv = 8.88
Nsbv = 0.15

# PseudoData_Set = "FromCalcGrids/Pseudo_Data/Set_2"
# Results_FileName = 'Set_2'

# m1v = 5.03250768
# Nuv = 0.70305177
# auv = 2.66220696
# buv = 15.07358796
# Nubv = -0.10143253
# Ndv = -1.28031833
# adv = 1.33923229
# bdv = 3.30827989
# Ndbv = -0.05256805
# Nsv = 11.18602047
# asv = 0.89546474
# bsv = 0.65188853
# Nsbv = 0.27352807


from datetime import datetime

# datetime object containing current date and time
start = datetime.now()
print("start =", start)

def generate_file(n_array):
    ms = Minuit(totalchi2Minuit,m1=m1v,Nu=Nuv,alphau=auv,betau=buv,Nubar=Nubv,
    Nd=Ndv,alphad=adv,betad=bdv,Ndbar=Ndbv,
    Ns=Nsv,alphas=asv,betas=bsv,Nsbar=Nsbv,
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
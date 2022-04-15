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

m1v = 34.19
Nuv = 3.53
auv = 2.50
buv = 15.49
Nubv = -0.19
Ndv = -5.34
adv = 1.26
bdv = 6.49
Ndbv = -0.89
Nsv = 18.00
asv = 3.60
bsv = 26.27
Nsbv = -1.79









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
    return temp_df.to_csv('Result_Fit2PseudoData.csv')

print(generate_file(par_name_array))
done = datetime.now()
print("done =", done)
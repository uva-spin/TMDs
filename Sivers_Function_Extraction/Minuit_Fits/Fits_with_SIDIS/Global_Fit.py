import lhapdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from Global_Constants import *
from Sivers_SIDIS_Definitions import *
#from Sivers_DY_Definitions import *
from iminuit import Minuit

def SIDIStotalchi2Minuit(m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar):
    datfilesarray=SIDIS_DataFilesArray
    datfilesnum=len(datfilesarray)
    temptotal=[]
    temptotaldata=[]
    temptotalerr=[]
    for i in range(0,datfilesnum):
        temptotal.append(totalfitDataSet(datfilesarray[i],m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar))
        temptotaldata.append(np.concatenate(ASiv_Val(datfilesarray[i])))
        temptotalerr.append(np.concatenate(ASiv_Err(datfilesarray[i])))
    tempTheory=np.concatenate((temptotal))
    tempY=np.concatenate((temptotaldata))
    tempYErr=np.concatenate((temptotalerr))
    tempChi2=np.sum(((tempY-tempTheory)/tempYErr)**2)
    return tempChi2


# def DYtotalchi2Minuit(m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar):
#     DY_datfilesarray=DY_DataFilesArray
#     DY_datfilesnum=len(DY_datfilesarray)
#     temptotal=[]
#     for i in range(0,DY_datfilesnum):
#         temptotal.append(DYtotalfitDataSet(DY_datfilesarray[i],m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar))
#     tempTheory=np.concatenate((temptotal), axis=None)
#     tempY=DYSiversVals(DY_datfilesarray)
#     tempYErr=DYSiversErrVals(DY_datfilesarray)
#     tempChi2=np.sum(((tempY-tempTheory)/tempYErr)**2)
#     return tempChi2

# def totalchi2Minuit(m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar):
#     tempchi2=SIDIStotalchi2Minuit(m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)+ DYtotalchi2Minuit(m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
#     return tempchi2


def SIDIS_Data_points():
    datfilesarray=SIDIS_DataFilesArray
    datfilesnum=len(datfilesarray)
    temptotaldata=[]
    temptotalerr=[]
    for i in range(0,datfilesnum):
        temptotaldata.append(np.concatenate(ASiv_Val(datfilesarray[i])))
        temptotalerr.append(np.concatenate(ASiv_Err(datfilesarray[i])))
    tempY=np.concatenate((temptotaldata))
    Data_points=len(tempY)
    #tempYErr=np.concatenate((temptotalerr))
    return Data_points

# def DY_Data_points():
#     DY_datfilesarray=DY_DataFilesArray
#     DY_datfilesnum=len(DY_datfilesarray)
#     tempY=DYSiversVals(DY_datfilesarray)
#     DY_Data_points=len(tempY)
#     return DY_Data_points

Data_points_SIDIS = SIDIS_Data_points() 
#Data_points_DY = DY_Data_points()
Total_data_points = Data_points_SIDIS

par_name_array=('m1','Nu','alphau','betau','Nubar','Nd','alphad','betad','Ndbar','Ns','alphas','betas','Nsbar')


### Define Initial Starting points here ###
M1_init = M1_test
AlphaU_init= AlphaU_test
BetaU_init = BetaU_test
AlphaD_init = AlphaD_test
BetaD_init = BetaD_test
AlphaS_init = AlphaS_test
BetaS_init = BetaS_test
NU_init = NU_test
NUbar_init = NUbar_test
ND_init = ND_test
NDbar_init = NDbar_test
NS_init = NS_test
NSbar_init = NSbar_test



def generate_file(n_array):
    #ms = Minuit(totalchi2Minuit,m1=M1_t2,Nu=NU_t2,alphau=AlphaU_t2,betau=BetaU_t2,Nubar=NUbar_t2,Nd=ND_t2,alphad=AlphaD_t2,betad=BetaD_t2,Ndbar=NDbar_t2,Ns=NS_t2,alphas=AlphaS_t2,betas=BetaS_t2,Nsbar=NSbar_t2)
    ms = Minuit(SIDIStotalchi2Minuit,m1=M1_init,Nu=NU_init,alphau=AlphaU_init,betau=BetaU_init,Nubar=NUbar_init,Nd=ND_init,alphad=AlphaD_init,betad=BetaD_init,Ndbar=NDbar_init,Ns=NS_init,alphas=AlphaS_init,betas=BetaS_init,Nsbar=NSbar_init)
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
    return temp_df.to_csv('Fit_Results.csv')

print(generate_file(par_name_array))
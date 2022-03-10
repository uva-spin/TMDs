import lhapdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from Global_Constants import *
from Sivers_SIDIS_Definitions import *
from Sivers_DY_Definitions import *

from iminuit import Minuit
import numpy as np
import tabulate as tab



#######################################################
############# This is the chi2 for SIDIS ##############
#######################################################

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

#######################################################
############# This is the chi2 for DY #################
#######################################################

def DYtotalchi2Minuit(m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar):
    DY_datfilesarray=DY_DataFilesArray
    DY_datfilesnum=len(DY_datfilesarray)
    temptotal=[]
    for i in range(0,DY_datfilesnum):
        temptotal.append(DYtotalfitDataSets(DY_datfilesarray,m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar))
    tempTheory=np.concatenate((temptotal), axis=None)
    tempY=DYSiversVals(DY_datfilesarray)
    tempYErr=DYSiversErrVals(DY_datfilesarray)
    tempChi2=np.sum(((tempY-tempTheory)/tempYErr)**2)
    return tempChi2


def totalchi2Minuit(m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar):
    tempchi2=SIDIStotalchi2Minuit(m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)+ DYtotalchi2Minuit(m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar)
    return tempchi2

m = Minuit(totalchi2Minuit,m1=M1_t2,Nu=NU_t2,alphau=AlphaU_t2,betau=BetaU_t2,Nubar=NUbar_t2,Nd=ND_t2,alphad=AlphaD_t2,betad=BetaD_t2,Ndbar=NDbar_t2,Ns=NS_t2,alphas=AlphaS_t2,betas=BetaS_t2,Nsbar=NSbar_t2)

# m = Minuit(totalchi2Minuit,m1=M1_test,Nu=NU_test,alphau=AlphaU_test,betau=BetaU_test,Nubar=NUbar_test,Nd=ND_test,alphad=AlphaD_test,betad=BetaD_test,Ndbar=NDbar_test,Ns=NS_test,alphas=AlphaS_test,betas=BetaS_test,Nsbar=NSbar_test)

# m.limits=((23.8,27.8),(2.52,2.72),(2.38,2.86),(15.36,16.68),(-.267,-.151),(-4.06,-3.06),(1.43,1.83),(7.6,11.2),(-1.4,.2),(13.44,14.72),(2.8,3.44),(20.6,22.6),(-2.7,1.3))
    
m.migrad()

f = open("Global_Fit_Sivers_SIDIS_DY_Plus_Without_H09_v1_1.txt","w")
for i in range(1):
    f.write(str(m.values))
    f.write("\n")
    f.write(str(m.errors))
    f.write("\n")
    f.write(str(m.params))
    f.write("\n")
    f.write(str(m.covariance))
    f.write("\n")
    f.write(str(m.fmin))
f.close()


with open("Global_Fit_Sivers_SIDIS_DY_Parameters_Plus_Without_H09_v1_1.txt", "w") as f2:
    f2.write(tab.tabulate(*m.params.to_table()))
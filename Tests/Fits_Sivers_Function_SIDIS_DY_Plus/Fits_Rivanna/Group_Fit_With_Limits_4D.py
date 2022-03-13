#!/usr/bin/env python
# coding: utf-8

# In[1]:


import Def_1 as SA
from iminuit import Minuit
import numpy as np
import tabulate as tab

# Here the data sets are listed and collected into and Array called "DataFilesArray"
Dat1='HERMES_p_2009.csv'
Dat2='HERMES_p_2020.csv'
Dat3='COMPASS_d_2009.csv'
Dat4='COMPASS_p_2015.csv'
DataFilesArray=[Dat1,Dat2,Dat3,Dat4]

print(DataFilesArray)


# In[2]:


def totalchi2Minuit(m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar):
    datfilesarray=DataFilesArray
    datfilesnum=len(datfilesarray)
    temptotal=[]
    for i in range(0,datfilesnum):
        temptotal.append(SA.totalfitDataSet(datfilesarray[i],m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar))
    tempTheory=np.concatenate((temptotal), axis=None)
    tempY=SA.SiversVals(datfilesarray)
    tempYErr=SA.SiversErrVals(datfilesarray)
    tempChi2=np.sum(((tempY-tempTheory)/tempYErr)**2)
    return tempChi2
    
# totalchi2Minuit(1,1,1,1,1,1,1,1,1,1,1,1,1)



# In[ ]:


m = Minuit(totalchi2Minuit,m1=SA.M1_t2,Nu=SA.NU_t2,alphau=SA.AlphaU_t2,betau=SA.BetaU_t2,Nubar=SA.NUbar_t2,Nd=SA.ND_t2,alphad=SA.AlphaD_t2,betad=SA.BetaD_t2,Ndbar=SA.NDbar_t2,Ns=SA.NS_t2,alphas=SA.AlphaS_t2,betas=SA.BetaS_t2,Nsbar=SA.NSbar_t2)

m.limits=((2.86,4.6),(.366,.504),(2.21,2.69),(11.8,14.2),(-.1295,-.1109),(-1.19,-.59),(1.48,2.68),(3.9,11.7).(-.34,.14),(5.07,6.45),(.412,.43),(.001,.103),(-.49,.83))
m.migrad()


# In[ ]:

f = open("Group_Fit_with_limits_4D_Increased_Limits_v9.txt","w")
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


with open("Group_Fit_4D_table_v9.txt", "w") as f2:
    f2.write(tab.tabulate(*m.params.to_table()))

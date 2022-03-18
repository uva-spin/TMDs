import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Dat1='Result_1.csv'
Dat2='Result_2.csv'
# Dat3=
# Dat4=
# ...

DF1=pd.read_csv(Dat1)
DF2=pd.read_csv(Dat2)
#DF3=pd.read_csv(Dat3)

### Here you can define array of results .cvs files ###
DFArray=[DF1,DF2]

def SortParameters(par,dfarray):
    alength=len(dfarray)
    pararray=[]
    chi2array=[]
    for i in range(0,alength):
        tempdf=DFArray[i]
        tempparslice=tempdf[(tempdf["parameter"]==par)]
        temppar=np.array(tempparslice['value'])[0]
        tempchi2=np.array(tempparslice['chi2'])[0]
        pararray.append(temppar)
        chi2array.append(tempchi2)
    return pararray,chi2array

f1=plt.figure(1)
plt.hist(SortParameters("m1",DFArray)[1],bins=25)
f1.savefig('Chi2_dist.pdf')

f2=plt.figure(2)
plt.hist(SortParameters("m1",DFArray)[0],bins=20)
f2.savefig('par_m1.pdf')
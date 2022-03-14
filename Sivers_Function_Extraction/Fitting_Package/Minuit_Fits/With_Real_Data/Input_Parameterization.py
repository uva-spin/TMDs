import numpy as np


Mp=0.938272
alpha_s=1/(137.0359998)

Kp2A=0.57
Pp2A=0.12
p2unp=0.25

ee=1
eU=2/3
eUbar=-2/3
eD=-1/3
eDbar=1/3
eS=-1/3
eSbar=1/3

qCharge=np.array([eSbar,eUbar,eDbar,eU,eD,eS])
qFlavor=np.array([-3,-2,-1,1,2,3])


def NNq(x,Nq,aq,bq):
    tempNNq = Nq*(x**aq)*((1-x)**(bq))*((aq+bq)**(aq+bq))/((aq**aq)*(bq**bq))
    return tempNNq

def NNqbar(x,Nqbar):
    tempNNqbar = Nqbar
    return tempNNqbar

##### SIGN of DY-Sivers relative to SIDIS-Sivers ######
SIGN = 1    

## DY ###
DY_DataFilesArray_Input=np.array(['../../Data/COMPASS_p_DY_2017.csv'])

## SIDIS ###
Dat1='../../Data/HERMES_p_2009.csv'
Dat2='../../Data/HERMES_p_2020.csv'
Dat3='../../Data/COMPASS_d_2009.csv'
Dat4='../../Data/COMPASS_p_2015.csv'
SIDIS_DataFilesArray_Input=[Dat1,Dat2,Dat3,Dat4]
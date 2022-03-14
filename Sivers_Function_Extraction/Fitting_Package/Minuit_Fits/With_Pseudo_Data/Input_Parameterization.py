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
DY_DataFilesArray_Input=np.array(['../../Pseudo_Data/Set_1/Pseudo_DY_COMPASS2017.csv'])

## SIDIS ###
Dat1='../../Pseudo_Data/Set_1/Pseudo_SIDIS_HERMES2009.csv'
Dat2='../../Pseudo_Data/Set_1/Pseudo_SIDIS_HERMES2020.csv'
Dat3='../../Pseudo_Data/Set_1/Pseudo_SIDIS_COMPASS2009.csv'
Dat4='../../Pseudo_Data/Set_1/Pseudo_SIDIS_COMPASS2015.csv'
SIDIS_DataFilesArray_Input=[Dat1,Dat2,Dat3,Dat4]
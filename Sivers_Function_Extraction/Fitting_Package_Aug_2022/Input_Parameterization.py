import numpy as np
import pandas as pd
#import os

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

SIGN = 1

# def NN(x,Nq,aq,bq):
#     tempNNq = Nq*(x**aq)*((1-x)**(bq))
#     return tempNNq


# def NNanti(x,Nq,aq,bq):
#     tempNNq = Nq*(x**aq)*((1-x)**(bq))
#     return tempNNq

### NNq parameterization ####

def NNq(x,Nq,aq,bq):
    tempNNq = Nq*(x**aq)*((1-x)**(bq))*((aq+bq)**(aq+bq))/((aq**aq)*(bq**bq))
    return tempNNq

def NNqbar(x,Nqbar):
    tempNNqbar = Nqbar
    return tempNNqbar


def NN(x,Nq,aq,bq):
    tempNNq = Nq*(x**aq)*((1-x)**(bq))*np.exp(((aq+bq)**(aq+bq))/((aq**aq)*(bq**bq)))
    return tempNNq

def NNanti(x,Nq,aq,bq):
    tempNNq = Nq*(x**aq)*((1-x)**(bq))*np.exp(((aq+bq)**(aq+bq))/((aq**aq)*(bq**bq)))
    return tempNNq



m1v = 1.0

Nuv = 2.4
auv = 1.8
buv = 0.05

Nubv = 2.2
aubv = 2.1
bubv = 0.06

Ndv = 2.45
adv = 1.95
bdv = 0.07

Ndbv = -1.6
adbv = 2.6
bdbv = 0.04

Nsv = 0.4
asv = 1.8
bsv = 0.51

Nsbv = -0.07
asbv = 0.55
bsbv = 0.07

# m1v = 0.9

# Nuv = 2.4
# auv = 1.8
# buv = 0.05

# Nubv = 2.2
# aubv = 2.1
# bubv = 0.06

# Ndv = 2.45
# adv = 1.95
# bdv = 0.07

# Ndbv = -1.6
# adbv = 2.6
# bdbv = 0.04

# Nsv = 0.4
# asv = 1.8
# bsv = 0.51

# Nsbv = -0.07
# asbv = 0.55
# bsbv = 0.07



## DY ###
DY_DataFilesArray=np.array(['Data/COMPASS_p_DY_2017.csv'])

## SIDIS ###
Dat1='Data/HERMES_p_2009.csv'
Dat2='Data/HERMES_p_2020.csv'
Dat3='Data/COMPASS_d_2009.csv'
Dat4='Data/COMPASS_p_2015.csv'
SIDIS_DataFilesArray=[Dat1,Dat2,Dat3,Dat4]

import numpy as np
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

def NN(x,Nq,aq,bq):
    tempNNq = Nq*(x**aq)*((1-x)**(bq))
    return tempNNq


def NNanti(x,Nq,aq,bq):
    tempNNq = Nq*(x**aq)*((1-x)**(bq))
    return tempNNq

# def NNq(x,Nq,aq,bq):
#     tempNNq = Nq*(x**aq)*((1-x)**(bq))*np.exp(((aq+bq)**(aq+bq))/((aq**aq)*(bq**bq)))
#     return tempNNq


# def NNqbar(x,Nq,aq,bq):
#     tempNNq = Nq*(x**aq)*((1-x)**(bq))*np.exp(((aq+bq)**(aq+bq))/((aq**aq)*(bq**bq)))
#     return tempNNq




m1v = 1

Nuv = 0.3
auv = 1
buv = 1

Nubv = 0.25
aubv = 1
bubv = 1

Ndv = 0.2
adv = 1
bdv = 1

Ndbv = 0.1
adbv = 1
bdbv = 1

Nsv = 0.2
asv = 1
bsv = 1

Nsbv = -0.1
asbv = 1
bsbv = 1


# m1v = 1

# Nuv = 0.3
# auv = 1
# buv = 1

# Nubv = 0.25
# aubv = 1
# bubv = 1

# Ndv = 0.2
# adv = 1
# bdv = 1

# Ndbv = 0.1
# adbv = 1
# bdbv = 1

# Nsv = 0.2
# asv = 1
# bsv = 1

# Nsbv = -0.1
# asbv = 1
# bsbv = 1

### Data set arrays

## DY ###
DY_DataFilesArray=np.array(['Data/COMPASS_p_DY_2017.csv'])

## SIDIS ###
Dat1='Data/HERMES_p_2009.csv'
Dat2='Data/HERMES_p_2020.csv'
Dat3='Data/COMPASS_d_2009.csv'
Dat4='Data/COMPASS_p_2015.csv'
SIDIS_DataFilesArray=[Dat1,Dat2,Dat3,Dat4]
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

def NNq(x,Nq,aq,bq):
    tempNNq = Nq*(x**aq)*((1-x)**(bq))*((aq+bq)**(aq+bq))/((aq**aq)*(bq**bq))
    return tempNNq

def NNqbar(x,Nqbar):
    tempNNqbar = x*Nqbar
    return tempNNqbar

#OutputFolder='Set_1'
# OutputFolder='Set_1'


### Pseudodata-Generation ###
# m1v = 6.31
# Nuv = 0.75
# auv = 2.68
# buv = 14.71
# Nubv = -0.094
# Ndv = -1.32
# adv = 1.39
# bdv = 5.21
# Ndbv = -0.01
# Nsv = 12.08
# asv = 0.91
# bsv = 0.54
# Nsbv = 0.25

# m1v = 40.40
# Nuv = 4.36
# auv = 2.37
# buv = 12.79
# Nubv = -0.24
# Ndv = -10.56
# adv = 1.23
# bdv = 3.11
# Ndbv = 0.32
# Nsv = 25.33
# asv = 1.38
# bsv = 5.35
# Nsbv = 0.06


# m1v = 25.07
# Nuv = 2.83
# auv = 2.40
# buv = 14.29
# Nubv = -0.16
# Ndv = -4.72
# adv = 1.30
# bdv = 5.29
# Ndbv = -0.93
# Nsv = 14.76
# asv = 3.07
# bsv = 20.62
# Nsbv = -1.82

# m1v = 34.19
# Nuv = 3.53
# auv = 2.50
# buv = 15.49
# Nubv = -0.19
# Ndv = -5.34
# adv = 1.26
# bdv = 6.49
# Ndbv = -0.89
# Nsv = 18.00
# asv = 3.60
# bsv = 26.27
# Nsbv = -1.79


m1v = 1
Nuv = 0.5
auv = 2
buv = 3
Nubv = -0.5
Ndv = 0.4
adv = 2.5
bdv = 3.5
Ndbv = -0.4
Nsv = 0.1
asv = 1.2
bsv = 1.3
Nsbv = -0.1


### Data set arrays

## DY ###
DY_DataFilesArray=np.array(['Data/COMPASS_p_DY_2017.csv'])

## SIDIS ###
Dat1='Data/HERMES_p_2009.csv'
Dat2='Data/HERMES_p_2020.csv'
Dat3='Data/COMPASS_d_2009.csv'
Dat4='Data/COMPASS_p_2015.csv'
SIDIS_DataFilesArray=[Dat1,Dat2,Dat3,Dat4]
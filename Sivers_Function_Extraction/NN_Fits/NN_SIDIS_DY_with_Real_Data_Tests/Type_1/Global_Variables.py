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


###############################################
########    Parameters   ######################
####### for Pseudodata-Generation #############
####### and for plotting  #####################
###############################################

m1v = 6.1
Nuv = 0.72
auv = 2.71
buv = 15.05
Nubv = -0.096
Ndv = -1.30
adv = 1.36
bdv = 4.7
Ndbv = -0.04
Nsv = 12
asv = 0.91
bsv = 0.52
Nsbv = 0.25

### Formula ####
def NN(x, n, a, b):
    return n * x**a * (1 - x)**b * (((a + b)**(a + b))/(a**a * b**b))

def NNanti(x,n):
    return n

## DY ###
DY_DataFilesArray=np.array(['Data/COMPASS_p_DY_2017.csv'])

## SIDIS ###
Dat1='Data/HERMES_p_2009.csv'
Dat2='Data/HERMES_p_2020.csv'
Dat3='Data/COMPASS_d_2009.csv'
Dat4='Data/COMPASS_p_2015.csv'
SIDIS_DataFilesArray=[Dat2,Dat3,Dat4]
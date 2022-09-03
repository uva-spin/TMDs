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


#def NNq(x,Nq,aq,bq):
#    tempNNq = Nq*(x**aq)*((1-x)**(bq))*np.exp(((aq+bq)**(aq+bq))/((aq**aq)*(bq**bq)))
#    return tempNNq

#def NNqbar(x,Nq,aq,bq):
#    tempNNq = Nq*(x**aq)*((1-x)**(bq))*np.exp(((aq+bq)**(aq+bq))/((aq**aq)*(bq**bq)))
#    return tempNNq



M1_test=1.303
AlphaU_test=0.6455
BetaU_test=3.122
AlphaD_test=1.777
BetaD_test=7.788
AlphaS_test=0.00006887
BetaS_test=0.0000000005537
NU_test=0.1695
NUbar_test=0.007605
ND_test=-0.4345
NDbar_test=-0.1420
NS_test=0.5626
NSbar_test=-0.1221


m1v = M1_test

Nuv = NU_test
auv = AlphaU_test
buv = BetaU_test

Nubv = NUbar_test
# aubv = 1
# bubv = 1

Ndv = ND_test
adv = AlphaD_test
bdv = BetaD_test

Ndbv = NDbar_test
# adbv = 1
# bdbv = 1

Nsv = NS_test
asv = AlphaS_test
bsv = BetaS_test

Nsbv = NSbar_test
# asbv = 1
# bsbv = 1


## DY ###
DY_DataFilesArray=np.array(['../../Data/COMPASS_p_DY_2017.csv'])

## SIDIS ###
Dat1='../../Data/HERMES_p_2009.csv'
Dat2='../../Data/HERMES_p_2020.csv'
Dat3='../../Data/COMPASS_d_2009.csv'
Dat4='../../Data/COMPASS_p_2015.csv'
SIDIS_DataFilesArray=[Dat2,Dat3,Dat4]

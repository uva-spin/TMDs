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


### Initial values for Sivers-SIDIS ######

## Set 1
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


## Set 2
# M1_t2 = 8.19
# AlphaU_t2=2.49
# BetaU_t2=13.7
# AlphaD_t2=1.88
# BetaD_t2=7.4
# AlphaS_t2=0.738
# BetaS_t2=0.633
# NU_t2=0.966
# NUbar_t2=-0.209
# ND_t2=-1.76
# NDbar_t2=-0.25
# NS_t2=12.33
# NSbar_t2=-0.1

## Set 3
# M1_t2 = 3.73
# AlphaU_t2=2.45
# BetaU_t2=13
# AlphaD_t2=2.08
# BetaD_t2=7.8
# AlphaS_t2=0.421
# BetaS_t2=0.052
# NU_t2=0.435
# NUbar_t2=-0.1202
# ND_t2=-0.89
# NDbar_t2=-0.10
# NS_t2=5.76
# NSbar_t2=0.17

# M1_t2 = 5.35
# AlphaU_t2=2.537
# BetaU_t2=13.82
# AlphaD_t2=2.2
# BetaD_t2=9.12
# AlphaS_t2=0.497
# BetaS_t2=0.066
# NU_t2=0.62
# NUbar_t2=-0.189
# ND_t2=-1.15
# NDbar_t2=-0.16
# NS_t2=10.09
# NSbar_t2=-0.18

M1_t2 = 22.2
AlphaU_t2=2.378
BetaU_t2=14.25
AlphaD_t2=1.45
BetaD_t2=6.3
AlphaS_t2=2.80
BetaS_t2=18.8
NU_t2=2.51
NUbar_t2=-0.202
ND_t2=-4.10
NDbar_t2=-1.1
NS_t2=13.44
NSbar_t2=-1.9

### Data set arrays

## DY ###
DY_DataFilesArray=np.array(['../Data/COMPASS_p_DY_2017.csv'])

## SIDIS ###
Dat1='../Data/HERMES_p_2009.csv'
Dat2='../Data/HERMES_p_2020.csv'
Dat3='../Data/COMPASS_d_2009.csv'
Dat4='../Data/COMPASS_p_2015.csv'
SIDIS_DataFilesArray=[Dat1,Dat2,Dat3,Dat4]
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
# M1_t2 = 7.97
# AlphaU_t2=2.497
# BetaU_t2=13.60
# AlphaD_t2=1.59
# BetaD_t2=6.1
# AlphaS_t2=0.802
# BetaS_t2=0.437
# NU_t2=0.940
# NUbar_t2=-0.162
# ND_t2=-1.67
# NDbar_t2=-0.20
# NS_t2=14.73
# NSbar_t2=0.1

# # ## Set 3
# # M1_t2 = 3.73
# # AlphaU_t2=2.45
# # BetaU_t2=13
# # AlphaD_t2=2.08
# # BetaD_t2=7.8
# # AlphaS_t2=0.421
# # BetaS_t2=0.052
# # NU_t2=0.435
# # NUbar_t2=-0.1202
# # ND_t2=-0.89
# # NDbar_t2=-0.10
# # NS_t2=5.76
# # NSbar_t2=0.17

# M1_t2 = 24.8
# AlphaU_t2=2.5
# BetaU_t2=15.69
# AlphaD_t2=1.53
# BetaD_t2=8.5
# AlphaS_t2=2.96
# BetaS_t2=21.1
# NU_t2=2.57
# NUbar_t2=-.238
# ND_t2=-3.81
# NDbar_t2=-1
# NS_t2=13.76
# NSbar_t2=-1.7

M1_t2 = 6.2
AlphaU_t2=2.73
BetaU_t2=15.9
AlphaD_t2=1.71
BetaD_t2=8.1
AlphaS_t2=.82
BetaS_t2=.49
NU_t2=.66
NUbar_t2=-.127
ND_t2=-1.11
NDbar_t2=-0.07
NS_t2=10.9
NSbar_t2=.1


### Data set arrays

## DY ###
DY_DataFilesArray=np.array(['../Data/COMPASS_p_DY_2017.csv'])

## SIDIS ###
#Dat1='../Data/HERMES_p_2009.csv'
Dat2='../Data/HERMES_p_2020.csv'
Dat3='../Data/COMPASS_d_2009.csv'
Dat4='../Data/COMPASS_p_2015.csv'
SIDIS_DataFilesArray=[Dat2,Dat3,Dat4]
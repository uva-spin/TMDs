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


### Initial values ######


# m1v = 6.1
# Nuv = 0.72
# auv = 2.71
# buv = 15.05
# Nubv = -0.096
# Ndv = -1.30
# adv = 1.36
# bdv = 4.7
# Ndbv = -0.04
# Nsv = 12
# asv = 0.91
# bsv = 0.52
# Nsbv = 0.25

# m1v = 34.19
# #m1v = 1
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

# PseudoData_Set = "FromCalcGrids/Pseudo_Data/Set_1"
# Results_FileName = 'Set_1'

# m1v = 5.03250768
# Nuv = 0.70305177
# auv = 2.66220696
# buv = 15.07358796
# Nubv = -0.10143253
# Ndv = -1.28031833
# adv = 1.33923229
# bdv = 3.30827989
# Ndbv = -0.05256805
# Nsv = 11.18602047
# asv = 0.89546474
# bsv = 0.65188853
# Nsbv = 0.27352807
 
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

# def NNq(x,Nq,aq,bq):
#     tempNNq = Nq*(x**aq)*((1-x)**(bq))*((aq+bq)**(aq+bq))/((aq**aq)*(bq**bq))
#     return tempNNq

# def NNqbar(x,Nqbar):
#     tempNNqbar = Nqbar
#     return tempNNqbar


def NN(x,Nq,aq,bq):
    tempNNq = Nq*(x**aq)*((1-x)**(bq))*((aq+bq)**(aq+bq))/((aq**aq)*(bq**bq))
    return tempNNq

def NNanti(x,Nqbar):
    tempNNqbar = x*Nqbar
    return tempNNqbar

##### SIGN of DY-Sivers relative to SIDIS-Sivers ######
SIGN = 1    

# ## DY ###
# DY_DataFilesArray=np.array(['../../Pseudo_Data/'+str(PseudoData_Set)+'/Pseudo_DY_COMPASS2017.csv'])

# ## SIDIS ###
# Dat1='../../Pseudo_Data/'+str(PseudoData_Set)+'/Pseudo_SIDIS_HERMES2009.csv'
# Dat2='../../Pseudo_Data/'+str(PseudoData_Set)+'/Pseudo_SIDIS_HERMES2020.csv'
# Dat3='../../Pseudo_Data/'+str(PseudoData_Set)+'/Pseudo_SIDIS_COMPASS2009.csv'
# Dat4='../../Pseudo_Data/'+str(PseudoData_Set)+'/Pseudo_SIDIS_COMPASS2015.csv'
# SIDIS_DataFilesArray=[Dat1,Dat2,Dat3,Dat4]
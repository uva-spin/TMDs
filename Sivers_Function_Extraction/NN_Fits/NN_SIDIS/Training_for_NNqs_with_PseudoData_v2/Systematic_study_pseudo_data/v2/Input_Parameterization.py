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

def NN(x,Nq,aq,bq):
    tempNNq = Nq*(x**aq)*((1-x)**(bq))*np.exp(((aq+bq)**(aq+bq))/((aq**aq)*(bq**bq)))
    return tempNNq


def NNanti(x,Nq,aq,bq):
    tempNNq = Nq*(x**aq)*((1-x)**(bq))*np.exp(((aq+bq)**(aq+bq))/((aq**aq)*(bq**bq)))
    return tempNNq



# m1v = np.random.normal(1,0.3)

# Nuv = np.random.normal(2.5,0.5)
# auv = np.random.normal(2.0,0.2)
# buv = np.random.normal(0.05,0.03)

# Nubv = np.random.normal(2.1,0.5)
# aubv = np.random.normal(2.0,0.2)
# bubv = np.random.normal(0.05,0.03)

# Ndv = np.random.normal(2.5,0.5)
# adv = np.random.normal(2.0,0.2)
# bdv = np.random.normal(0.05,0.03)

# Ndbv = np.random.normal(-1.5,0.5)
# adbv = np.random.normal(2.5,0.2)
# bdbv = np.random.normal(0.05,0.03)

# Nsv = np.random.normal(0.2,0.5)
# asv = np.random.normal(1.8,0.2)
# bsv = np.random.normal(0.5,0.03)

# Nsbv = np.random.normal(-0.05,0.1)
# asbv = np.random.normal(0.5,0.2)
# bsbv = np.random.normal(0.1,0.05)

m1v = 1.17790408346687

Nuv = 3.48242656160677
auv = 2.0576359497584
buv = 0.046590933687603

Nubv = 2.66930457203584
aubv = 1.9678384269736
bubv = 0.090731319545579

Ndv = 1.95386108532782
adv = 2.39317587098313
bdv = 0.08780215144728

Ndbv = -1.76359128485984
adbv = 2.59938807579905
bdbv = 0.040523150758288

Nsv = -0.045456135562756
asv = 1.64621551303519
bsv = 0.522766051460815

Nsbv = -0.122828733697722
asbv = 0.792213664324234
bsbv = 0.083062749152512


# par_name_array=('m1','Nu','alphau','betau','Nubar','alphaub','betaub','Nd','alphad','betad','Ndbar','alphadb','betadb','Ns','alphas','betas','Nsbar','alphasb','betasb')
# temp_df=pd.DataFrame({'parameter':[],'value':[]})
# temp_df['parameter']=par_name_array
# temp_df['value']=np.array([m1v,Nuv,auv,buv,Nubv,aubv,bubv,Ndv,adv,bdv,Ndbv,adbv,bdbv,Nsv,asv,bsv,Nsbv,asbv,bsbv])
# temp_df.to_csv('parameters.csv')

# m1v = 1

# Nuv = 2.0
# auv = 2.0
# buv = 0.05

# Nubv = -2.3
# aubv = 2.0
# bubv = 0.02

# Ndv = -2.3
# adv = 2.0
# bdv = 0.01

# Ndbv = 0.5
# adbv = 2.0
# bdbv = 0.05

# Nsv = -0.5
# asv = 2
# bsv = 1

# Nsbv = -0.02
# asbv = 2
# bsbv = 1




# m1v = 1

# Nuv = 0.03
# auv = 0.01
# buv = 0.01

# Nubv = 0.028
# aubv = 0.01
# bubv = 0.01

# Ndv = -0.03
# adv = 0.01
# bdv = 0.01

# Ndbv = 0.012
# adbv = 0.01
# bdbv = 0.01

# Nsv = 0.025
# asv = 0.01
# bsv = 0.01

# Nsbv = -0.015
# asbv = 0.01
# bsbv = 0.01


# m1v = 1

# Nuv = 0.18
# auv = 1
# buv = 6.6

# Nubv = -0.01
# aubv = 0.01
# bubv = 0.01

# Ndv = -0.52
# adv = 2
# bdv = 10

# Ndbv = -0.06
# adbv = 0.01
# bdbv = 0.01

# Nsv = 0.025
# asv = 0.01
# bsv = 0.01

# Nsbv = -0.015
# asbv = 0.01
# bsbv = 0.01

### Data set arrays

## DY ###
DY_DataFilesArray=np.array(['Data/COMPASS_p_DY_2017.csv'])

## SIDIS ###
Dat1='Data/HERMES_p_2009.csv'
Dat2='Data/HERMES_p_2020.csv'
Dat3='Data/COMPASS_d_2009.csv'
Dat4='Data/COMPASS_p_2015.csv'
SIDIS_DataFilesArray=[Dat1,Dat2,Dat3,Dat4]
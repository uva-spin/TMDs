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

m1v = 1.00316673307616

Nuv = 1.49446270438738
auv = 1.98103771230181
buv = 0.127916440563156

Nubv = 2.37603080110487
aubv = 1.75168710689433
bubv = 0.062408936933261

Ndv = 2.7387230469617
adv = 1.90966768227514
bdv = 0.034628504775884

Ndbv = -1.93421677905926
adbv = 2.2857733434424
bdbv = 0.03086718447667

Nsv = -0.495919098099663
asv = 1.76388117957333
bsv = 0.495928514574857

Nsbv = 0.11016391953608
asbv = 0.460801777550646
bsbv = 0.120299701573669

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
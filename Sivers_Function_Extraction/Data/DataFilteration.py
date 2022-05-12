import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## DY ###
DY_DataFilesArray=np.array(['Data/COMPASS_p_DY_2017.csv'])

## SIDIS ###
Dat1='Data/HERMES_p_2009.csv'
Dat2='Data/HERMES_p_2020.csv'
Dat3='Data/COMPASS_d_2009.csv'
Dat4='Data/COMPASS_p_2015.csv'
SIDIS_DataFilesArray=[Dat1,Dat2,Dat3,Dat4]


def Filter_SIDIS_Data(datafile):
    tempdf=pd.read_csv(datafile)
    #tempCUTdf=tempdf[(tempdf['z'] > 0.2) & (tempdf['z'] < 0.6) | (tempdf['Q2'] > 1.63) | (tempdf['phT'] > 0.2) | (tempdf['phT'] < 0.9) ]
    tempCUTdf=tempdf[(tempdf['z'] > 0.2) & (tempdf['z'] < 0.6)]
    tempCUTdf=tempCUTdf[(tempCUTdf['phT'] > 0.2) & (tempCUTdf['phT'] < 0.9)]
    tempCUTdf=tempCUTdf[(tempCUTdf['Q2'] > 1.63)]
    return tempCUTdf


SIDIS_HERMES2009_Cut=Filter_SIDIS_Data(Dat1)
SIDIS_HERMES2020_Cut=Filter_SIDIS_Data(Dat2)
SIDIS_COMPASS2009_Cut=Filter_SIDIS_Data(Dat3)
SIDIS_COMPASS2015_Cut=Filter_SIDIS_Data(Dat4)

OutputFolder='Filtered_Data'

SIDIS_HERMES2009_Cut.to_csv(str(OutputFolder)+'/SIDIS_HERMES2009.csv')
SIDIS_HERMES2020_Cut.to_csv(str(OutputFolder)+'/SIDIS_HERMES2020.csv')
SIDIS_COMPASS2009_Cut.to_csv(str(OutputFolder)+'/SIDIS_COMPASS2009.csv')
SIDIS_COMPASS2015_Cut.to_csv(str(OutputFolder)+'/SIDIS_COMPASS2015.csv')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
## DY ###
DY_Data1='Data/COMPASS_p_DY_2017.csv'

## SIDIS ###
SIDIS_Dat1='Data/HERMES_p_2009.csv'
SIDIS_Dat2='Data/HERMES_p_2020.csv'
SIDIS_Dat3='Data/COMPASS_d_2009.csv'
SIDIS_Dat4='Data/COMPASS_p_2015.csv'


def PlotDependence(filename,Had,Var,Kin1,Kin2):
    tempdf=pd.read_csv(filename)
    temp_slice=tempdf[(tempdf["hadron"]==Had)&(tempdf["1D_dependence"]==Var)]
    tempKin1=np.array(temp_slice[Kin1])
    tempKin2=np.array(temp_slice[Kin2])
    tempplot=plt.plot(tempKin1,tempKin2,'.')
    return tempplot

def PlotFullKin(filename,Kin1,Kin2):
    tempdf=pd.read_csv(filename)
    x=np.array(tempdf[Kin1])
    y=np.array(tempdf[Kin2])
    nbins=len(x)
    k = kde.gaussian_kde([x,y])
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    tempplot=plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
    return tempplot

#PlotDependence(SIDIS_Dat1,"pi+","x","x","Q2")
PlotFullKin(SIDIS_Dat1,"x","Q2")
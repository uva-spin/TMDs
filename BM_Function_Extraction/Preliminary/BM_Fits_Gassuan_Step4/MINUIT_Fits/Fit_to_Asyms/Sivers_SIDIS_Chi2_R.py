import lhapdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PathsR import *
from Constants import *
from Input_Parameterization import *
from Sivers_SIDIS_Definitions_R import *



def SIDIStotalchi2Minuit_R(m1,Nu,au,bu,Nub,aub,bub,Nd,ad,bd,Ndb,adb,bdb,Ns,aS,bS,Nsb,asb,bsb):
    datfilesarray=SIDIS_DataFilesArrayR
    datfilesnum=len(datfilesarray)
    temptotal=[]
    temptotaldata=[]
    temptotalerr=[]
    for i in range(0,datfilesnum):
        temptotal.append(totalfitDataSet(datfilesarray[i],m1=m1,Nu=Nu,au=au,bu=bu,Nub=Nub,aub=aub,bub=bub,
        Nd=Nd,ad=ad,bd=bd,Ndb=Ndb,adb=adb,bdb=bdb,Ns=Ns,aS=aS,bS=bS,Nsb=Nsb,asb=asb,bsb=bsb))
        temptotaldata.append(np.concatenate(ASiv_Val(datfilesarray[i])[0]))
        temptotalerr.append(np.concatenate(ASiv_Val(datfilesarray[i])[1]))
    tempTheory=np.concatenate((temptotal))
    tempY=np.concatenate((temptotaldata))
    tempYErr=np.concatenate((temptotalerr))
    tempChi2=np.sum(((tempY-tempTheory)/tempYErr)**2)
    return tempChi2


def SIDIS_Data_points_R():
    datfilesarray=SIDIS_DataFilesArrayR
    datfilesnum=len(datfilesarray)
    temptotaldata=[]
    for i in range(0,datfilesnum):
        temptotaldata.append(np.concatenate(ASiv_Val(datfilesarray[i])[0]))
    tempY=np.concatenate((temptotaldata))
    Data_points=len(tempY)
    return Data_points



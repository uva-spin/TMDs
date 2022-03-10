import lhapdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from Global_Constants import *

### Notes
# Make sure to use (<kT>2 correct value)

PDFdataset = lhapdf.mkPDF("cteq61")


def ks2Avg(m1,kperp2Avg):
    test_ks2Avg=((m1**2)*kperp2Avg)/((m1**2)+kperp2Avg)
    return test_ks2Avg

def NNq(x,Nq,aq,bq):
    tempNNq = Nq*(x**aq)*((1-x)**(bq))*((aq+bq)**(aq+bq))/((aq**aq)*(bq**bq))
    return tempNNq

def NNqbar(x,Nqbar):
    tempNNqbar = Nqbar
    return tempNNqbar

def xFxQ2(dataset,flavor,x,QQ):
    temp_parton_dist_x=np.array(dataset.xfxQ2(flavor, x, QQ),dtype=object)
    return temp_parton_dist_x

def hfunc(kp,m1):
    temphfunc=np.sqrt(2*ee)*(kp/m1)*(np.exp((-kp**2)/(m1**2)))
    return temphfunc

def unpol_fxkT(dataset,flavor,x,kp,QQ):
    temp_unpol=(np.exp((-kp**2)/(Kp2A**2)))*(1/((np.pi)*(Kp2A**2)))*(xFxQ2(dataset,flavor,x,QQ))
    return temp_unpol

def SiversFuncQ_DY(dataset,flavor,x,QQ,kp,**parms):
    m1= parms["m1"]
    tempM1=m1
    if(flavor==2):
        Nq = parms["Nu"]
        aq= parms["alphau"]
        bq = parms["betau"]        
        tempsiv=2*NNq(x,Nq,aq,bq)*hfunc(kp,tempM1)*unpol_fxkT(dataset,flavor,x,kp,QQ)
    if(flavor==1):
        Nq=parms["Nd"]
        aq=parms["alphad"]
        bq=parms["betad"]
        tempsiv=2*NNq(x,Nq,aq,bq)*hfunc(kp,tempM1)*unpol_fxkT(dataset,flavor,x,kp,QQ)
    if(flavor==3):
        Nq=parms["Ns"]
        aq=parms["alphas"]
        bq=parms["betas"]
        tempsiv=2*NNq(x,Nq,aq,bq)*hfunc(kp,tempM1)*unpol_fxkT(dataset,flavor,x,kp,QQ)
    return x*tempsiv

        
def SiversFuncAntiQ_DY(dataset,flavor,x,QQ,kp,**parms):
    m1= parms["m1"]
    tempM1=m1
    if(flavor==-2):
        Nq=parms["Nubar"]
        tempsiv=2*NNqbar(x,Nq)*hfunc(kp,tempM1)*unpol_fxkT(dataset,flavor,x,kp,QQ)
    if(flavor==-1):
        Nq=parms["Ndbar"]
        tempsiv=2*NNqbar(x,Nq)*hfunc(kp,tempM1)*unpol_fxkT(dataset,flavor,x,kp,QQ)
    if(flavor==-3):
        Nq=parms["Nsbar"]
        tempsiv=2*NNqbar(x,Nq)*hfunc(kp,tempM1)*unpol_fxkT(dataset,flavor,x,kp,QQ)
    return x*tempsiv


def Int_Sivers_DY_Q(dataset,flavor,x,QQ,**parms):
    m1= parms["m1"]
    tempM1=m1
    if(flavor==2):
        Nq = parms["Nu"]
        aq= parms["alphau"]
        bq = parms["betau"]        
        tempsiv=2*NNq(x,Nq,aq,bq)*(xFxQ2(dataset,flavor,x,QQ))
    if(flavor==1):
        Nq=parms["Nd"]
        aq=parms["alphad"]
        bq=parms["betad"]
        tempsiv=2*NNq(x,Nq,aq,bq)*(xFxQ2(dataset,flavor,x,QQ))
    if(flavor==3):
        Nq=parms["Ns"]
        aq=parms["alphas"]
        bq=parms["betas"]
        tempsiv=2*NNq(x,Nq,aq,bq)*(xFxQ2(dataset,flavor,x,QQ))
    return x*tempsiv


def Int_Sivers_DY_AntiQ(dataset,flavor,x,QQ,**parms):
    m1= parms["m1"]
    tempM1=m1
    if(flavor==-2):
        Nq=parms["Nubar"]
        tempsiv=2*NNqbar(x,Nq)*(xFxQ2(dataset,flavor,x,QQ))
    if(flavor==-1):
        Nq=parms["Ndbar"]
        tempsiv=2*NNqbar(x,Nq)*(xFxQ2(dataset,flavor,x,QQ))
    if(flavor==-3):
        Nq=parms["Nsbar"]
        tempsiv=2*NNqbar(x,Nq)*(xFxQ2(dataset,flavor,x,QQ))
    return x*tempsiv


#print(SiversFuncAntiQ_DY(PDFdataset,-2,0.2,10,0.25,m1=1,Nu=0.2,alphau=2,betau=2,Nubar=2))


### A common numerator and denominator of DY Sivers Asymmetry
### (4*(np.pi)*(alpha_s)**2)/(9*(Mp**2)*ss)

## Note that, QQ here is just Q not Q2!

def Numerator_Siv_DY_mod(x1,x2,qT,QQ,**parms):
    m1= parms["m1"]
    BB0=((np.sqrt(2*ee))*qT/m1)*(1/(x1+x2))
    BBexp=np.square(ks2Avg(m1,Kp2A))*(np.exp(-np.square(qT)/(ks2Avg(m1,Kp2A)+Kp2A)))/((np.pi)*Kp2A*np.square(ks2Avg(m1,Kp2A)+Kp2A))
    tempSum=0
    for i in range(0,len(qFlavor)-3):
        tempSum = tempSum + (np.square(qCharge[i]))*Int_Sivers_DY_AntiQ(PDFdataset,qFlavor[i],x1,QQ,**parms)*(xFxQ2(PDFdataset,-qFlavor[i],x2,QQ))
    for i in range(len(qFlavor)-3,len(qFlavor)):
        tempSum = tempSum + (np.square(qCharge[i]))*Int_Sivers_DY_Q(PDFdataset,qFlavor[i],x1,QQ,**parms)*(xFxQ2(PDFdataset,-qFlavor[i],x2,QQ))
    return tempSum*BB0*BBexp*((np.pi)/2)

#print(Numerator_Siv_DY_mod(0.2,0.5,2,10,m1=1,Nu=1,alphau=1,betau=1,Nubar=1,Nd=1,alphad=1,betad=1,Ndbar=1,Ns=1,alphas=1,betas=1,Nsbar=1))

## Note that, QQ here is just Q not Q2!

def Denominator_Siv_DY_mod(x1,x2,qT,QQ,**parms):
    m1= parms["m1"]
    BBexp=(np.exp(-np.square(qT)/(Kp2A+Kp2A)))/((np.pi)*(Kp2A+Kp2A))
    tempSum=0
    for i in range(0,len(qFlavor)-3):
        tempSum = tempSum + (np.square(qCharge[i]))*(xFxQ2(PDFdataset,qFlavor[i],x1,QQ))*(xFxQ2(PDFdataset,-qFlavor[i],x2,QQ))
    for i in range(len(qFlavor)-3,len(qFlavor)):
        tempSum = tempSum + (np.square(qCharge[i]))*(xFxQ2(PDFdataset,qFlavor[i],x1,QQ))*(xFxQ2(PDFdataset,-qFlavor[i],x2,QQ))
    return tempSum*BBexp*(np.pi)


def DY_Sivers_Asym(x1,x2,xF,qT,QQ,**parms):
    # Here the xF variable is set to have no impact because we have x1 and x2
    tempSiv_DY=xF*0+(Numerator_Siv_DY_mod(x1,x2,qT,QQ,**parms))/(Denominator_Siv_DY_mod(x1,x2,qT,QQ,**parms))
    return tempSiv_DY

#print(DY_Sivers_Asym(0.2,0.5,0.3,2,10,m1=1,Nu=1,alphau=1,betau=1,Nubar=1,Nd=1,alphad=1,betad=1,Ndbar=1,Ns=1,alphas=1,betas=1,Nsbar=1))


###########################################################
###### Calculating Theory Values for DY Asymmetries  ######
###########################################################

##### This function will calculate the theory values for each data set


def Dep_array(filename):
    tempdf=pd.read_csv(filename)
    temphad_data=tempdf["Dependence"]
    temphad=temphad_data.dropna().unique()
    refined_had_array=[]
    for i in range(0,len(temphad)):
        if((temphad[i]=="x1") or (temphad[i]=="x2") or (temphad[i]=="xF") or (temphad[i]=="QT") or (temphad[i]=="QM")):
            refined_had_array.append(temphad[i])
    return refined_had_array


#print(Dep_array('../Data/COMPASS_p_DY_2017.csv'))

def mergekins(list1,list2,list3,list4,list5):
    mergedkins=tuple(zip(list1,list2,list3,list4,list5))
    return mergedkins


def DYkinslice(filename,dep_var):
    tempdf=pd.read_csv(filename)
    temp_slice=tempdf[(tempdf["Dependence"]==dep_var)]
    tempx1=np.array(temp_slice["x1"],dtype=object)
    tempx2=np.array(temp_slice["x2"],dtype=object)
    tempxF=np.array(temp_slice["xF"],dtype=object)
    tempQT=np.array(temp_slice["QT"],dtype=object)
    tempQ=np.array(temp_slice["QM"],dtype=object)
    tempDYkins=np.array((tempx1,tempx2,tempxF,tempQT,tempQ))
    return tempDYkins


#print(DYkinslice('../Data/COMPASS_p_DY_2017.csv','x1')[0])

# Following function looks fine
def DYSivFitDep(KV,**parms):
    m1= parms["m1"]
    Nu = parms["Nu"]
    alphau= parms["alphau"]
    betau = parms["betau"]
    Nubar = parms["Nubar"]
    Nd = parms["Nd"]
    alphad= parms["alphad"]
    betad = parms["betad"]
    Ndbar = parms["Ndbar"]
    Ns = parms["Ns"]
    alphas= parms["alphas"]
    betas = parms["betas"]
    Nsbar = parms["Nsbar"]
    ################
    x1,x2,xF,QT,QM = KV
    array_size=len(xF)
    tempDYSiv_val=[]
    for i in range(0,array_size):
        tempDYSiv=DY_Sivers_Asym(x1[i],x2[i],xF[i],QT[i],QM[i],**parms)       
        tempDYSiv_val.append(tempDYSiv)
    return tempDYSiv_val

#print(DYSivFitDep(DYkinslice('../Data/COMPASS_p_DY_2017.csv','x1'),m1=1,Nu=1,alphau=1,betau=1,Nubar=1,Nd=1,alphad=1,betad=1,Ndbar=1,Ns=1,alphas=1,betas=1,Nsbar=1))

def DYtotalfitDataSet(datfile,**parms):
    m1= parms["m1"]
    Nu = parms["Nu"]
    alphau= parms["alphau"]
    betau = parms["betau"]
    Nubar = parms["Nubar"]
    Nd = parms["Nd"]
    alphad= parms["alphad"]
    betad = parms["betad"]
    Ndbar = parms["Ndbar"]
    Ns = parms["Ns"]
    alphas= parms["alphas"]
    betas = parms["betas"]
    Nsbar = parms["Nsbar"]
    dep_len=len(Dep_array(datfile))
    tempdep=Dep_array(datfile)
    fittot=[]
    for i in range(0,dep_len):
        if tempdep[i]=="x1":
            tempfit=DYSivFitDep(DYkinslice(datfile,'x1'),**parms)
            fittot.append(tempfit)
        elif tempdep[i]=="x2":
            tempfit=DYSivFitDep(DYkinslice(datfile,'x2'),**parms)
            fittot.append(tempfit)
        elif tempdep[i]=="xF":
            tempfit=DYSivFitDep(DYkinslice(datfile,'xF'),**parms)
            fittot.append(tempfit)
        elif tempdep[i]=="QT":
            tempfit=DYSivFitDep(DYkinslice(datfile,'QT'),**parms)
            fittot.append(tempfit)
        elif tempdep[i]=="QM":
            tempfit=DYSivFitDep(DYkinslice(datfile,'QM'),**parms)
            fittot.append(tempfit)
    return np.concatenate((fittot), axis=None)



#print(DYtotalfitDataSet('../Data/COMPASS_p_DY_2017.csv',m1=1,Nu=1,alphau=1,betau=1,Nubar=1,Nd=1,alphad=1,betad=1,Ndbar=1,Ns=1,alphas=1,betas=1,Nsbar=1))


###########################################################
########## DY Asymmetry (Data)  ########################
###########################################################


def DYdataslice(filename,dep_var):
    tempdf=pd.read_csv(filename)
    temp_slice=tempdf[(tempdf["Dependence"]==dep_var)]
    tempSiv=np.array(temp_slice["Siv"],dtype=object)
    temperrSiv=np.array(temp_slice["tot_err"],dtype=object)
    tempDYdata=np.array((tempSiv,temperrSiv))
    return tempDYdata


def DYSiv_data_oneset(datfile):
    tempx1data=DYdataslice(datfile,"x1")
    tempx2data=DYdataslice(datfile,"x2")
    tempxFdata=DYdataslice(datfile,"xF")
    tempQTdata=DYdataslice(datfile,"QT")
    tempQMdata=DYdataslice(datfile,"QM")   
    ##### Asy ################
    tempDYAsy_x1=np.array(tempx1data[0],dtype=object)
    tempDYAsy_x2=np.array(tempx2data[0],dtype=object)
    tempDYAsy_xF=np.array(tempxFdata[0],dtype=object)
    tempDYAsy_QT=np.array(tempQTdata[0],dtype=object)
    tempDYAsy_QM=np.array(tempQMdata[0],dtype=object)
    tempAsy=np.concatenate((tempDYAsy_x1,tempDYAsy_x2,tempDYAsy_xF,tempDYAsy_QT,tempDYAsy_QM))
    ##### err ################
    tempDYAsyerr_x1=np.array(tempx1data[1],dtype=object)
    tempDYAsyerr_x2=np.array(tempx2data[1],dtype=object)
    tempDYAsyerr_xF=np.array(tempxFdata[1],dtype=object)
    tempDYAsyerr_QT=np.array(tempQTdata[1],dtype=object)
    tempDYAsyerr_QM=np.array(tempQMdata[1],dtype=object)
    tempAsyErr=np.concatenate((tempDYAsyerr_x1,tempDYAsyerr_x2,tempDYAsyerr_xF,tempDYAsyerr_QT,tempDYAsyerr_QM))
    return tempAsy,tempAsyErr


def DYSiversVals(datafilesarray):
    datfilesnum=len(datafilesarray)
    tempSiv=[]
    for i in range(0,datfilesnum):
        tempSiv.append(DYSiv_data_oneset(datafilesarray[i])[0])
    tempflatSiv=np.concatenate((tempSiv), axis=None)
    return np.concatenate((tempflatSiv), axis=None)

def DYSiversErrVals(datafilesarray):
    datfilesnum=len(datafilesarray)
    tempSivErr=[]
    for i in range(0,datfilesnum):
        tempSivErr.append(DYSiv_data_oneset(datafilesarray[i])[1])
    tempflatSivErr=np.concatenate((tempSivErr), axis=None)
    return np.concatenate((tempflatSivErr), axis=None)

#testdatarray1=(['../Data/COMPASS_p_DY_2017.csv'])
#print(DYSiversVals(testdatarray1))
#print(DYSiv_data_oneset('../Data/COMPASS_p_DY_2017.csv'))
#print(DY_Sivers_Asym(0.2,0.5,2,10,m1=1,Nu=1,alphau=1,betau=1,Nubar=1,Nd=1,alphad=1,betad=1,Ndbar=1,Ns=1,alphas=1,betas=1,Nsbar=1))
#print(DYdataslice('../Data/COMPASS_p_DY_2017.csv','x1')[1])    
#print(DY_Sivers_Asym(0.2,0.5,2,10,m1=1,Nu=1,alphau=1,betau=1,Nubar=1,Nd=1,alphad=1,betad=1,Ndbar=1,Ns=1,alphas=1,betas=1,Nsbar=1))
    

####################################################################################
###########################   Chi2 Function(s)  ####################################
####################################################################################


## This should go to the file that is going to be called by the slurm file

DY_DataFilesArray1=np.array(['../Data/COMPASS_p_DY_2017.csv'])

def DYtotalchi2Minuit(**parms):
    m1= parms["m1"]
    Nu = parms["Nu"]
    alphau= parms["alphau"]
    betau = parms["betau"]
    Nubar = parms["Nubar"]
    Nd = parms["Nd"]
    alphad= parms["alphad"]
    betad = parms["betad"]
    Ndbar = parms["Ndbar"]
    Ns = parms["Ns"]
    alphas= parms["alphas"]
    betas = parms["betas"]
    Nsbar = parms["Nsbar"]
    DY_datfilesarray=DY_DataFilesArray1
    DY_datfilesnum=len(DY_datfilesarray)
    temptotal=[]
    for i in range(0,DY_datfilesnum):
        temptotal.append(DYtotalfitDataSet(DY_datfilesarray[i],**parms))
    tempTheory=np.concatenate((temptotal), axis=None)
    tempY=DYSiversVals(DY_datfilesarray)
    tempYErr=DYSiversErrVals(DY_datfilesarray)
    tempChi2=np.sum(((tempY-tempTheory)/tempYErr)**2)
    return tempChi2

    
#print(DYtotalchi2Minuit(m1=1,Nu=1,alphau=1,betau=1,Nubar=1,Nd=1,alphad=1,betad=1,Ndbar=1,Ns=1,alphas=1,betas=1,Nsbar=1))

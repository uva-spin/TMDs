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


def Denominator_Siv_DY_mod(x1,x2,qT,QQ,**parms):
    m1= parms["m1"]
    BBexp=(np.exp(-np.square(qT)/(Kp2A+Kp2A)))/((np.pi)*(Kp2A+Kp2A))
    tempSum=0
    for i in range(0,len(qFlavor)-3):
        tempSum = tempSum + (np.square(qCharge[i]))*(xFxQ2(PDFdataset,qFlavor[i],x1,QQ))*(xFxQ2(PDFdataset,-qFlavor[i],x2,QQ))
    for i in range(len(qFlavor)-3,len(qFlavor)):
        tempSum = tempSum + (np.square(qCharge[i]))*(xFxQ2(PDFdataset,qFlavor[i],x1,QQ))*(xFxQ2(PDFdataset,-qFlavor[i],x2,QQ))
    return tempSum*BBexp*(np.pi)


def DY_Sivers_Asym(x1,x2,qT,QQ,**parms):
    tempSiv_DY=(Numerator_Siv_DY_mod(x1,x2,qT,QQ,**parms))/(Denominator_Siv_DY_mod(x1,x2,qT,QQ,**parms))
    return tempSiv_DY


print(DY_Sivers_Asym(0.2,0.5,2,10,m1=1,Nu=1,alphau=1,betau=1,Nubar=1,Nd=1,alphad=1,betad=1,Ndbar=1,Ns=1,alphas=1,betas=1,Nsbar=1))
    
    

    

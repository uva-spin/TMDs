import lhapdf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from Input_Parameterization import *
#from Sivers_SIDIS_Definitions_for_Pseudo import *

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


def NNq(x,Nq,aq,bq):
    tempNNq = Nq*(x**aq)*((1-x)**(bq))*((aq+bq)**(aq+bq))/((aq**aq)*(bq**bq))
    return tempNNq

def NNqbar(x,Nqbar):
    tempNNqbar = Nqbar
    return tempNNqbar


#parms_df = pd.read_csv('Parameters.csv')
test_pars=([7.0, 0.89, 2.78, 19.4, -0.07, -2.33, 2.5, 15.8, -0.29, -14, 4.9, 12, -0.1])
#test_pars=([7.0, 0.89, 2.78, 19.4, -0.07, -2.33, 2.5, 15.8, -0.29, -14, 4.9, 3, -0.1])
test_errs=([0.6, 0.05, 0.17, 1.6, 0.06, 0.31, 0.4, 3.2, 0.27, 10, 3.3, 10, 0.2])
# These parameters gave chi2/N = 531.2/313 = 1.69 for MINUIT fit with 
# HERMES2009, COMPASS2009, COMPASS2015 data (HERMES2020 wasn't included)
parms_df = pd.read_csv('./PseudoData_Parm1/Parameters.csv')
Parameters_array = parms_df.to_numpy()

HERMES2009 = './Data/HERMES_p_2009.csv'
HERMES2020 = './Data/HERMES_p_2020.csv'
COMPASS2009 = './Data/COMPASS_d_2009.csv'
COMPASS2015 = './Data/COMPASS_p_2015.csv'

#######################################################################
########## Sivers Function ############################################
#######################################################################

PDFdataset = lhapdf.mkPDF("cteq61")

def xFxQ2(dataset,flavor,x,QQ):
    temp_parton_dist_x=np.array(dataset.xfxQ2(flavor, x, QQ))
    return temp_parton_dist_x
    
def hfunc(kp,m1):
    temphfunc=np.sqrt(2*ee)*(kp/m1)*(np.exp((-kp**2)/(m1**2)))
    return temphfunc

def SiversFuncQ(dataset,flavor,x,QQ,kp,fitresult):
    tempM1=fitresult[0]
    if(flavor==2):
        Nq=fitresult[1]
        aq=fitresult[2]
        bq=fitresult[3]
        tempsiv=2*NNq(x,Nq,aq,bq)*hfunc(kp,tempM1)*(np.exp((-kp**2)/(Kp2A**2)))*(1/((np.pi)*(Kp2A**2)))*(xFxQ2(dataset,flavor,x,QQ))
    if(flavor==1):
        Nq=fitresult[5]
        aq=fitresult[6]
        bq=fitresult[7]
        tempsiv=2*NNq(x,Nq,aq,bq)*hfunc(kp,tempM1)*(np.exp((-kp**2)/(Kp2A**2)))*(1/((np.pi)*(Kp2A**2)))*(xFxQ2(dataset,flavor,x,QQ))
    if(flavor==3):
        Nq=fitresult[9]
        aq=fitresult[10]
        bq=fitresult[11]
        tempsiv=2*NNq(x,Nq,aq,bq)*hfunc(kp,tempM1)*(np.exp((-kp**2)/(Kp2A**2)))*(1/((np.pi)*(Kp2A**2)))*(xFxQ2(dataset,flavor,x,QQ))
    return tempsiv
    
    
def SiversFuncAntiQ(dataset,flavor,x,QQ,kp,fitresult):
    tempM1=fitresult[0]
    if(flavor==-2):
        tempM1=fitresult[0]
        Nq=fitresult[4]
        tempsiv=2*NNqbar(x,Nq)*hfunc(kp,tempM1)*(np.exp((-kp**2)/(Kp2A**2)))*(1/((np.pi)*(Kp2A**2)))*(xFxQ2(dataset,flavor,x,QQ))
    if(flavor==-1):
        tempM1=fitresult[0]
        Nq=fitresult[8]
        tempsiv=2*NNqbar(x,Nq)*hfunc(kp,tempM1)*(np.exp((-kp**2)/(Kp2A**2)))*(1/((np.pi)*(Kp2A**2)))*(xFxQ2(dataset,flavor,x,QQ))
    if(flavor==-3):
        tempM1=fitresult[0]
        Nq=fitresult[12]
        tempsiv=2*NNqbar(x,Nq)*hfunc(kp,tempM1)*(np.exp((-kp**2)/(Kp2A**2)))*(1/((np.pi)*(Kp2A**2)))*(xFxQ2(dataset,flavor,x,QQ))
    return tempsiv
    

def plotSiversQ(flavor,ParmResults,col):
    tempkT=np.linspace(0, 1.5)
    tempSiv=[SiversFuncQ(PDFdataset,flavor,0.1,2.4,tempkT[i],ParmResults) for i in range(0,len(tempkT))]
    plt.plot(tempkT,tempSiv, color = col)
    #return tempSiv

def plotSiversAntiQ(flavor,ParmResults):
    tempkT=np.linspace(0, 1.5)
    tempSiv=[SiversFuncAntiQ(PDFdataset,flavor,0.1,2.4,tempkT[i],ParmResults) for i in range(0,len(tempkT))]
    plt.plot(tempkT,tempSiv, color = col)



    
# ### Different style band

def plotSiversQBandFill(flavor,array,col,lbl,ParmResults):
    tempkT=np.linspace(0, 1.5)
    lenarray=len(array)
    tempASivVal=[]
    tempSiv=[SiversFuncQ(PDFdataset,flavor,0.1,2.4,tempkT[i],ParmResults) for i in range(0,len(tempkT))]
    Smax=[]
    Smin=[]
    for i in range(0,len(tempkT)):
        tempmax = SiversFuncQ(PDFdataset,flavor,0.1,2.4,tempkT[i],ParmResults)
        tempmin = SiversFuncQ(PDFdataset,flavor,0.1,2.4,tempkT[i],ParmResults)
        for j in range(0,lenarray):
            modified_array = np.delete(np.array(array[j]), 0)
            ttt=SiversFuncQ(PDFdataset,flavor,0.1,2.4,tempkT[i],modified_array)
            if ttt > tempmax:
                tempmax = ttt
            elif ttt < tempmin:
                tempmin = ttt
        Smax.append(tempmax)
        Smin.append(tempmin)
    plt.fill_between(tempkT,Smin,Smax,alpha=0.4,color=col,linewidth=0.01)    
    plt.plot(tempkT,tempSiv,col,label=lbl)

    
def plotSiversAntiQBandFill(flavor,array,col,lbl,ParmResults):
    tempkT=np.linspace(0, 1.5)
    lenarray=len(array)
    tempASivVal=[]
    tempSiv=[SiversFuncAntiQ(PDFdataset,flavor,0.1,2.4,tempkT[i],ParmResults) for i in range(0,len(tempkT))]
    #print(tempSiv)
    Smax=[]
    Smin=[]
    for i in range(0,len(tempkT)):
        tempmax = SiversFuncAntiQ(PDFdataset,flavor,0.1,2.4,tempkT[i],ParmResults)
        tempmin = SiversFuncAntiQ(PDFdataset,flavor,0.1,2.4,tempkT[i],ParmResults)
        for j in range(0,lenarray):
            modified_array = np.delete(np.array(array[j]), 0)
            ttt=SiversFuncAntiQ(PDFdataset,flavor,0.1,2.4,tempkT[i],modified_array)
            if ttt > tempmax:
                tempmax = ttt
            elif ttt < tempmin:
                tempmin = ttt
        Smax.append(tempmax)
        Smin.append(tempmin)
    plt.fill_between(tempkT,Smin,Smax,alpha=0.4,color=col,linewidth=0.01)    
    plt.plot(tempkT,tempSiv,col,label=lbl)
    
plt.figure(1)
plotSiversQBandFill(2,Parameters_array,'b','$u$',test_pars) 
plotSiversQBandFill(1,Parameters_array,'r','$d$',test_pars)
plotSiversQBandFill(3,Parameters_array,'g','$s$',test_pars)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(-0.12,0.12)
plt.legend(loc=4,fontsize=20,handlelength=3)
plt.savefig('PseudoData_Parm1/SiversQ.pdf')

plt.figure(2)
plotSiversAntiQBandFill(-2,Parameters_array,'b','$u$',test_pars) 
plotSiversAntiQBandFill(-1,Parameters_array,'r','$d$',test_pars)
plotSiversAntiQBandFill(-3,Parameters_array,'g','$s$',test_pars)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(-0.012,0.012)
plt.legend(loc=4,fontsize=20,handlelength=3)
plt.savefig('PseudoData_Parm1/SiversAntiQ.pdf')
    


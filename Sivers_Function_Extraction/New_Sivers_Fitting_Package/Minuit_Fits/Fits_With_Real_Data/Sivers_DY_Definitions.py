import lhapdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Paths import *
from Constants import *
from Input_Parameterization import *

### Notes
# Make sure to use (<kT>2 correct value)
Kp2A=p2unp 
#PDFdataset = lhapdf.mkPDF("cteq61")


###########################################################################################
##### Note: make sure to have this order in the same order as the data files array order ##
###########################################################################################
DY_PDFs_Files=(DY_PDFs_COMPASS_p_2017_x1,DY_PDFs_COMPASS_p_2017_x2)

##########################################################################################

def ks2Avg(m1,kperp2Avg):
    test_ks2Avg=((m1**2)*kperp2Avg)/((m1**2)+kperp2Avg)
    return test_ks2Avg


## Here we need to make sure whether we provide Q or QQ. For DY it looks like it uses Q
# def xFxQ2(dataset,flavor,x,QQ):
#     #temp_parton_dist_x=np.array(dataset.xfxQ2(flavor, x, QQ),dtype=object)
#     temp_parton_dist_x=np.array(dataset.xfxQ(flavor, x, QQ),dtype=object)
#     return temp_parton_dist_x


def DY_xFxQ(datafile,flavor):
    tempvals=pd.read_csv(datafile)
    tempx=tempvals['x']
    #tempQQ=tempvals['QQ']
    if(flavor==-3):
        temp_PDF=tempvals['sbar']
    elif(flavor==-2):
        temp_PDF=tempvals['ubar']   
    elif(flavor==-1):
        temp_PDF=tempvals['dbar']
    if(flavor==1):
        temp_PDF=tempvals['d']
    elif(flavor==2):
        temp_PDF=tempvals['u']
    elif(flavor==3):
        temp_PDF=tempvals['s']                 
    return np.array(temp_PDF)



def hfunc(kp,m1):
    temphfunc=np.sqrt(2*ee)*(kp/m1)*(np.exp((-kp**2)/(m1**2)))
    return temphfunc

def unpol_fxkT(dataset,flavor):
    tempvals=pd.read_csv(dataset)
    kp=tempvals['QT']
    temp_unpol=(np.exp((-kp**2)/(Kp2A**2)))*(1/((np.pi)*(Kp2A**2)))*(DY_xFxQ(dataset,flavor))
    return temp_unpol

def SiversFuncQ_DY(dataset,flavor,**parms):
    tempvals=pd.read_csv(dataset)
    x=tempvals['x']
    kp=tempvals['QT']
    m1= parms["m1"]
    tempM1=m1
    if(flavor==2):
        Nq = parms["Nu"]
        aq= parms["alphau"]
        bq = parms["betau"]        
        tempsiv=2*NNq(x,Nq,aq,bq)*hfunc(kp,tempM1)*unpol_fxkT(dataset,flavor)
    if(flavor==1):
        Nq=parms["Nd"]
        aq=parms["alphad"]
        bq=parms["betad"]
        tempsiv=2*NNq(x,Nq,aq,bq)*hfunc(kp,tempM1)*unpol_fxkT(dataset,flavor)
    if(flavor==3):
        Nq=parms["Ns"]
        aq=parms["alphas"]
        bq=parms["betas"]
        tempsiv=2*NNq(x,Nq,aq,bq)*hfunc(kp,tempM1)*unpol_fxkT(dataset,flavor)
    return SIGN*tempsiv

        
def SiversFuncAntiQ_DY(dataset,flavor,**parms):
    tempvals=pd.read_csv(dataset)
    x=tempvals['x']
    kp=tempvals['QT']
    m1= parms["m1"]
    tempM1=m1
    if(flavor==-2):
        Nq=parms["Nubar"]
        tempsiv=2*NNqbar(x,Nq)*hfunc(kp,tempM1)*unpol_fxkT(dataset,flavor)
    if(flavor==-1):
        Nq=parms["Ndbar"]
        tempsiv=2*NNqbar(x,Nq)*hfunc(kp,tempM1)*unpol_fxkT(dataset,flavor)
    if(flavor==-3):
        Nq=parms["Nsbar"]
        tempsiv=2*NNqbar(x,Nq)*hfunc(kp,tempM1)*unpol_fxkT(dataset,flavor)
    return SIGN*tempsiv


def Int_Sivers_DY_Q(dataset,flavor,**parms):
    tempvals=pd.read_csv(dataset)
    x=tempvals['x']
    kp=tempvals['QT']
    m1= parms["m1"]
    tempM1=m1
    if(flavor==2):
        Nq = parms["Nu"]
        aq= parms["alphau"]
        bq = parms["betau"]        
        tempsiv=2*NNq(x,Nq,aq,bq)*(DY_xFxQ(dataset,flavor))
    if(flavor==1):
        Nq=parms["Nd"]
        aq=parms["alphad"]
        bq=parms["betad"]
        tempsiv=2*NNq(x,Nq,aq,bq)*(DY_xFxQ(dataset,flavor,))
    if(flavor==3):
        Nq=parms["Ns"]
        aq=parms["alphas"]
        bq=parms["betas"]
        tempsiv=2*NNq(x,Nq,aq,bq)*(DY_xFxQ(dataset,flavor))
    return SIGN*tempsiv

#print(Int_Sivers_DY_Q(DY_PDFs_COMPASS_p_2017_x2,2,m1=1,Nu=0.2,alphau=2,betau=2,Nubar=2))

def Int_Sivers_DY_AntiQ(dataset,flavor,**parms):
    tempvals=pd.read_csv(dataset)
    x=tempvals['x']
    kp=tempvals['QT']
    m1= parms["m1"]
    tempM1=m1
    if(flavor==-2):
        Nq=parms["Nubar"]
        tempsiv=2*NNqbar(x,Nq)*(DY_xFxQ(dataset,flavor))
    if(flavor==-1):
        Nq=parms["Ndbar"]
        tempsiv=2*NNqbar(x,Nq)*(DY_xFxQ(dataset,flavor))
    if(flavor==-3):
        Nq=parms["Nsbar"]
        tempsiv=2*NNqbar(x,Nq)*(DY_xFxQ(dataset,flavor))
    return SIGN*tempsiv

#print(Int_Sivers_DY_AntiQ(DY_PDFs_COMPASS_p_2017_x2,-2,m1=1,Nu=0.2,alphau=2,betau=2,Nubar=2))

### A common numerator and denominator of DY Sivers Asymmetry
### (4*(np.pi)*(alpha_s)**2)/(9*(Mp**2)*ss)

## Note that, QQ here is just Q not Q2!

def Numerator_Siv_DY_mod(PDFdataset_x1,PDFdataset_x2,**parms):
    m1= parms["m1"]
    tempvals_x1=pd.read_csv(PDFdataset_x1)
    tempvals_x2=pd.read_csv(PDFdataset_x2)
    x1=tempvals_x1['x']
    x2=tempvals_x2['x']
    qT=tempvals_x1['QT']
    BB0=((np.sqrt(2*ee))*qT/m1)*(1/(x1+x2))
    BBexp=np.square(ks2Avg(m1,Kp2A))*(np.exp(-np.square(qT)/(ks2Avg(m1,Kp2A)+Kp2A)))/((np.pi)*Kp2A*np.square(ks2Avg(m1,Kp2A)+Kp2A))
    tempSum=0
    for i in range(0,len(qFlavor)-3):
        tempSum = tempSum + (np.square(qCharge[i]))*Int_Sivers_DY_AntiQ(PDFdataset_x1,qFlavor[i],**parms)*(DY_xFxQ(PDFdataset_x2,-qFlavor[i]))
    for i in range(len(qFlavor)-3,len(qFlavor)):
        tempSum = tempSum + (np.square(qCharge[i]))*Int_Sivers_DY_Q(PDFdataset_x1,qFlavor[i],**parms)*(DY_xFxQ(PDFdataset_x2,-qFlavor[i]))
    return tempSum*BB0*BBexp*((np.pi)/2)

#print(Numerator_Siv_DY_mod(DY_PDFs_COMPASS_p_2017_x1,DY_PDFs_COMPASS_p_2017_x2,m1=1,Nu=1,alphau=1,betau=1,Nubar=1,Nd=1,alphad=1,betad=1,Ndbar=1,Ns=1,alphas=1,betas=1,Nsbar=1))

## Note that, QQ here is just Q not Q2!

def Denominator_Siv_DY_mod(PDFdataset_x1,PDFdataset_x2,**parms):
    m1= parms["m1"]
    tempvals_x1=pd.read_csv(PDFdataset_x1)
    tempvals_x2=pd.read_csv(PDFdataset_x2)
    x1=tempvals_x1['x']
    x2=tempvals_x2['x']
    qT=tempvals_x1['QT']
    BBexp=(np.exp(-np.square(qT)/(Kp2A+Kp2A)))/((np.pi)*(Kp2A+Kp2A))
    tempSum=0
    for i in range(0,len(qFlavor)-3):
        tempSum = tempSum + (np.square(qCharge[i]))*(DY_xFxQ(PDFdataset_x1,qFlavor[i]))*(DY_xFxQ(PDFdataset_x2,-qFlavor[i]))
    for i in range(len(qFlavor)-3,len(qFlavor)):
        tempSum = tempSum + (np.square(qCharge[i]))*(DY_xFxQ(PDFdataset_x1,qFlavor[i]))*(DY_xFxQ(PDFdataset_x2,-qFlavor[i]))
    return tempSum*BBexp*(np.pi)

#print(Denominator_Siv_DY_mod(DY_PDFs_COMPASS_p_2017_x1,DY_PDFs_COMPASS_p_2017_x2,m1=1,Nu=1,alphau=1,betau=1,Nubar=1,Nd=1,alphad=1,betad=1,Ndbar=1,Ns=1,alphas=1,betas=1,Nsbar=1))


def DY_Sivers_Asym(PDFdataset_x1,PDFdataset_x2,**parms):
    # Here the xF variable is set to have no impact because we have x1 and x2
    xF=1
    tempSiv_DY=xF*0+(Numerator_Siv_DY_mod(PDFdataset_x1,PDFdataset_x2,**parms))/(Denominator_Siv_DY_mod(PDFdataset_x1,PDFdataset_x2,**parms))
    return tempSiv_DY

#print(DY_Sivers_Asym(DY_PDFs_COMPASS_p_2017_x1,DY_PDFs_COMPASS_p_2017_x2,m1=1,Nu=1,alphau=1,betau=1,Nubar=1,Nd=1,alphad=1,betad=1,Ndbar=1,Ns=1,alphas=1,betas=1,Nsbar=1))


# ###########################################################
# ###### Calculating Theory Values for DY Asymmetries  ######
# ###########################################################

# ##### This function will calculate the theory values for each data set


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


def DYtotalfitDataSets(datafilesarray,**parms):
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
    data_len=len(datafilesarray)
    fittot=[]
    for i in range(0,data_len):
        fittot.append(DY_Sivers_Asym(DY_PDFs_Files[2*i+0],DY_PDFs_Files[2*i+1],**parms))
    return np.concatenate((fittot), axis=None)


def DYPseudoData(datafile,**parms):
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
    #data_len=len(DY_DataFilesArray)
    index = np.where(DY_DataFilesArray == datafile)[0][0]
    fittot=DY_Sivers_Asym(DY_PDFs_Files[2*index+0],DY_PDFs_Files[2*index+1],**parms)
    return np.array(fittot)
#print(DYtotalfitDataSets(DY_DataFilesArray,m1=1,Nu=1,alphau=1,betau=1,Nubar=1,Nd=1,alphad=1,betad=1,Ndbar=1,Ns=1,alphas=1,betas=1,Nsbar=1))


# ###########################################################
# ########## DY Asymmetry (Data)  ########################
# ###########################################################


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
    

# ####################################################################################
# ###########################   Chi2 Function(s)  ####################################
# ####################################################################################

def DYtotalchi2Minuit(m1,Nu,alphau,betau,Nubar,Nd,alphad,betad,Ndbar,Ns,alphas,betas,Nsbar):
    DY_datfilesarray=DY_DataFilesArray
    DY_datfilesnum=len(DY_datfilesarray)
    temptotal=[]
    for i in range(0,DY_datfilesnum):
        temptotal.append(DYtotalfitDataSets(DY_datfilesarray,m1=m1,Nu=Nu,alphau=alphau,betau=betau,Nubar=Nubar,Nd=Nd,alphad=alphad,betad=betad,Ndbar=Ndbar,Ns=Ns,alphas=alphas,betas=betas,Nsbar=Nsbar))
    tempTheory=np.concatenate((temptotal), axis=None)
    tempY=DYSiversVals(DY_datfilesarray)
    tempYErr=DYSiversErrVals(DY_datfilesarray)
    tempChi2=np.sum(((tempY-tempTheory)/tempYErr)**2)
    return tempChi2

def DY_Data_points():
    DY_datfilesarray=DY_DataFilesArray
    DY_datfilesnum=len(DY_datfilesarray)
    tempY=DYSiversVals(DY_datfilesarray)
    DY_Data_points=len(tempY)
    return DY_Data_points
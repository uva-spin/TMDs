from sys import setrecursionlimit
import lhapdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.integrate as sciint
from scipy import integrate
from Global_Constants import *

PDFdataset = lhapdf.mkPDF("cteq61")


def xFxQ2(dataset,flavor,x,QQ):
    temp_parton_dist_x=np.array(dataset.xfxQ2(flavor, x, QQ),dtype=object)
    return temp_parton_dist_x


####################################################
############### pD Cross-Section ###################
####################################################


###### General Expression interms of x1,x2,QT,QQ #######################
def vpD(x1,x2,QT,QQ,**parms):
    CC= parms["cc"]
    p2bm=parms["p2bm"]
    Hu= parms["Hu"]
    Hubar= parms["Hubar"]
    Hd= parms["Hd"]
    Hdbar= parms["Hdbar"]
    Hs= parms["Hs"]
    Hsbar= parms["Hsbar"]
    alpha=(x1**CC)*(1-x1)*(x2**CC)*(1-x2)*(((4*Hu*xFxQ2(PDFdataset,2,x1,QQ))+(Hd*xFxQ2(PDFdataset,1,x1,QQ))+(Hs*xFxQ2(PDFdataset,3,x1,QQ)))*((Hubar*xFxQ2(PDFdataset,-2,x2,QQ))+(Hdbar*xFxQ2(PDFdataset,-1,x2,QQ))+(Hsbar*xFxQ2(PDFdataset,-3,x2,QQ)))+((4*Hubar*xFxQ2(PDFdataset,-2,x1,QQ))+(Hdbar*xFxQ2(PDFdataset,-1,x1,QQ))+(Hsbar*xFxQ2(PDFdataset,-3,x1,QQ)))*((Hu*xFxQ2(PDFdataset,2,x2,QQ))+(Hd*xFxQ2(PDFdataset,1,x2,QQ))+(Hs*xFxQ2(PDFdataset,3,x2,QQ))))
    beta=((4*xFxQ2(PDFdataset,2,x1,QQ))+(4*xFxQ2(PDFdataset,1,x1,QQ))+(4*xFxQ2(PDFdataset,3,x1,QQ)))*((xFxQ2(PDFdataset,-2,x1,QQ))+(xFxQ2(PDFdataset,-1,x1,QQ))+(xFxQ2(PDFdataset,-3,x1,QQ)))
    test_vpD=(p2unp*alpha*(np.square(QT))*np.exp(-np.square(QT)/(2*p2bm)))/(2*(np.square(Mp))*p2bm*beta*(np.square(QT))*np.exp(-np.square(QT)/(2*p2unp)))
    return test_vpD                                                                          
#print(vpD(0.5,0.5,2,10,cc=1,p2bm=2,Hu=1,Hubar=1,Hd=1,Hdbar=1,Hs=1,Hsbar=1))


###### QT dependence #######################
def vpDQT(QT,QQ,**parms):
    x1low=0.1
    x1high=0.5
    x2low=0.5
    x2high=0.1
    CC= parms["cc"]
    p2bm=parms["p2bm"]
    Hu= parms["Hu"]
    Hubar= parms["Hubar"]
    Hd= parms["Hd"]
    Hdbar= parms["Hdbar"]
    Hs= parms["Hs"]
    Hsbar= parms["Hsbar"]
    def vpDQTx1x2(x1,x2):
        alpha=(x1**CC)*(1-x1)*(x2**CC)*(1-x2)*(((4*Hu*xFxQ2(PDFdataset,2,x1,QQ))+(Hd*xFxQ2(PDFdataset,1,x1,QQ))+(Hs*xFxQ2(PDFdataset,3,x1,QQ)))*((Hubar*xFxQ2(PDFdataset,-2,x2,QQ))+(Hdbar*xFxQ2(PDFdataset,-1,x2,QQ))+(Hsbar*xFxQ2(PDFdataset,-3,x2,QQ)))+((4*Hubar*xFxQ2(PDFdataset,-2,x1,QQ))+(Hdbar*xFxQ2(PDFdataset,-1,x1,QQ))+(Hsbar*xFxQ2(PDFdataset,-3,x1,QQ)))*((Hu*xFxQ2(PDFdataset,2,x2,QQ))+(Hd*xFxQ2(PDFdataset,1,x2,QQ))+(Hs*xFxQ2(PDFdataset,3,x2,QQ))))
        beta=((4*xFxQ2(PDFdataset,2,x1,QQ))+(4*xFxQ2(PDFdataset,1,x1,QQ))+(4*xFxQ2(PDFdataset,3,x1,QQ)))*((xFxQ2(PDFdataset,-2,x1,QQ))+(xFxQ2(PDFdataset,-1,x1,QQ))+(xFxQ2(PDFdataset,-3,x1,QQ)))
        test_vpD=(p2unp*alpha*(np.square(QT))*np.exp(-np.square(QT)/(2*p2bm)))/(2*(np.square(Mp))*p2bm*beta*(np.square(QT))*np.exp(-np.square(QT)/(2*p2unp)))
        return test_vpD
    int_result,err=sciint.dblquad(vpDQTx1x2,x1low,x1high,x2low,x2high)
    return int_result                                                                          
#print(vpDQT(2,10,cc=1,p2bm=2,Hu=1,Hubar=1,Hd=1,Hdbar=1,Hs=1,Hsbar=1))


###### x1 dependence #######################
def vpDx1(x1,QQ,**parms):
    QTlow=2
    QThigh=5
    x2low=0.5
    x2high=0.1
    CC= parms["cc"]
    p2bm=parms["p2bm"]
    Hu= parms["Hu"]
    Hubar= parms["Hubar"]
    Hd= parms["Hd"]
    Hdbar= parms["Hdbar"]
    Hs= parms["Hs"]
    Hsbar= parms["Hsbar"]
    def vpDQTx2(QT,x2):
        alpha=(x1**CC)*(1-x1)*(x2**CC)*(1-x2)*(((4*Hu*xFxQ2(PDFdataset,2,x1,QQ))+(Hd*xFxQ2(PDFdataset,1,x1,QQ))+(Hs*xFxQ2(PDFdataset,3,x1,QQ)))*((Hubar*xFxQ2(PDFdataset,-2,x2,QQ))+(Hdbar*xFxQ2(PDFdataset,-1,x2,QQ))+(Hsbar*xFxQ2(PDFdataset,-3,x2,QQ)))+((4*Hubar*xFxQ2(PDFdataset,-2,x1,QQ))+(Hdbar*xFxQ2(PDFdataset,-1,x1,QQ))+(Hsbar*xFxQ2(PDFdataset,-3,x1,QQ)))*((Hu*xFxQ2(PDFdataset,2,x2,QQ))+(Hd*xFxQ2(PDFdataset,1,x2,QQ))+(Hs*xFxQ2(PDFdataset,3,x2,QQ))))
        beta=((4*xFxQ2(PDFdataset,2,x1,QQ))+(4*xFxQ2(PDFdataset,1,x1,QQ))+(4*xFxQ2(PDFdataset,3,x1,QQ)))*((xFxQ2(PDFdataset,-2,x1,QQ))+(xFxQ2(PDFdataset,-1,x1,QQ))+(xFxQ2(PDFdataset,-3,x1,QQ)))
        test_vpD=(p2unp*alpha*(np.square(QT))*np.exp(-np.square(QT)/(2*p2bm)))/(2*(np.square(Mp))*p2bm*beta*(np.square(QT))*np.exp(-np.square(QT)/(2*p2unp)))
        return test_vpD
    int_result,err=sciint.dblquad(vpDQTx2,x2low,x2high,QTlow,QThigh)
    return int_result                                                                          
#print(vpDx1(0.5,10,cc=1,p2bm=2,Hu=1,Hubar=1,Hd=1,Hdbar=1,Hs=1,Hsbar=1))


###### x2 dependence #######################
def vpDx2(x2,QQ,**parms):
    QTlow=2
    QThigh=5
    x1low=0.5
    x1high=0.1
    CC= parms["cc"]
    p2bm=parms["p2bm"]
    Hu= parms["Hu"]
    Hubar= parms["Hubar"]
    Hd= parms["Hd"]
    Hdbar= parms["Hdbar"]
    Hs= parms["Hs"]
    Hsbar= parms["Hsbar"]
    def vpDQTx1(QT,x1):
        alpha=(x1**CC)*(1-x1)*(x2**CC)*(1-x2)*(((4*Hu*xFxQ2(PDFdataset,2,x1,QQ))+(Hd*xFxQ2(PDFdataset,1,x1,QQ))+(Hs*xFxQ2(PDFdataset,3,x1,QQ)))*((Hubar*xFxQ2(PDFdataset,-2,x2,QQ))+(Hdbar*xFxQ2(PDFdataset,-1,x2,QQ))+(Hsbar*xFxQ2(PDFdataset,-3,x2,QQ)))+((4*Hubar*xFxQ2(PDFdataset,-2,x1,QQ))+(Hdbar*xFxQ2(PDFdataset,-1,x1,QQ))+(Hsbar*xFxQ2(PDFdataset,-3,x1,QQ)))*((Hu*xFxQ2(PDFdataset,2,x2,QQ))+(Hd*xFxQ2(PDFdataset,1,x2,QQ))+(Hs*xFxQ2(PDFdataset,3,x2,QQ))))
        beta=((4*xFxQ2(PDFdataset,2,x1,QQ))+(4*xFxQ2(PDFdataset,1,x1,QQ))+(4*xFxQ2(PDFdataset,3,x1,QQ)))*((xFxQ2(PDFdataset,-2,x1,QQ))+(xFxQ2(PDFdataset,-1,x1,QQ))+(xFxQ2(PDFdataset,-3,x1,QQ)))
        test_vpD=(p2unp*alpha*(np.square(QT))*np.exp(-np.square(QT)/(2*p2bm)))/(2*(np.square(Mp))*p2bm*beta*(np.square(QT))*np.exp(-np.square(QT)/(2*p2unp)))
        return test_vpD
    int_result,err=sciint.dblquad(vpDQTx1,x1low,x1high,QTlow,QThigh)
    return int_result                                                                          
#print(vpDx2(0.5,10,cc=1,p2bm=2,Hu=1,Hubar=1,Hd=1,Hdbar=1,Hs=1,Hsbar=1))



####################################################
############### pp Cross-Section ###################
####################################################


###### General Expression interms of x1,x2,QT,QQ #######################
def vpp(x1,x2,QT,QQ,**parms):
    CC= parms["cc"]
    p2bm=parms["p2bm"]
    Hu= parms["Hu"]
    Hubar= parms["Hubar"]
    Hd= parms["Hd"]
    Hdbar= parms["Hdbar"]
    Hs= parms["Hs"]
    Hsbar= parms["Hsbar"]
    alpha=(x1**CC)*(1-x1)*(x2**CC)*(1-x2)*(4*Hu*xFxQ2(PDFdataset,2,x1,QQ)*Hubar*xFxQ2(PDFdataset,-2,x2,QQ) + Hd*xFxQ2(PDFdataset,1,x1,QQ)*Hdbar*xFxQ2(PDFdataset,-2,x2,QQ) + Hs*xFxQ2(PDFdataset,3,x1,QQ)*Hsbar*xFxQ2(PDFdataset,-3,x2,QQ))
    beta= 4*xFxQ2(PDFdataset,2,x1,QQ)*xFxQ2(PDFdataset,-2,x2,QQ) + xFxQ2(PDFdataset,1,x1,QQ)*xFxQ2(PDFdataset,-1,x2,QQ) + xFxQ2(PDFdataset,3,x1,QQ)*xFxQ2(PDFdataset,-3,x2,QQ)
    test_vpp=(p2unp*alpha*(np.square(QT))*np.exp(-np.square(QT)/(2*p2bm)))/(2*(np.square(Mp))*p2bm*beta*(np.square(QT))*np.exp(-np.square(QT)/(2*p2unp)))
    return test_vpp                                                                          
#print(vpD(0.5,0.5,2,10,cc=1,p2bm=2,Hu=1,Hubar=1,Hd=1,Hdbar=1,Hs=1,Hsbar=1))



###### QT dependence #######################
def vppQT(QT,QQ,**parms):
    x1low=0.1
    x1high=0.5
    x2low=0.5
    x2high=0.1
    CC= parms["cc"]
    p2bm=parms["p2bm"]
    Hu= parms["Hu"]
    Hubar= parms["Hubar"]
    Hd= parms["Hd"]
    Hdbar= parms["Hdbar"]
    Hs= parms["Hs"]
    Hsbar= parms["Hsbar"]
    def vppQTx1x2(x1,x2):
        alpha=(x1**CC)*(1-x1)*(x2**CC)*(1-x2)*(4*Hu*xFxQ2(PDFdataset,2,x1,QQ)*Hubar*xFxQ2(PDFdataset,-2,x2,QQ) + Hd*xFxQ2(PDFdataset,1,x1,QQ)*Hdbar*xFxQ2(PDFdataset,-2,x2,QQ) + Hs*xFxQ2(PDFdataset,3,x1,QQ)*Hsbar*xFxQ2(PDFdataset,-3,x2,QQ))
        beta= 4*xFxQ2(PDFdataset,2,x1,QQ)*xFxQ2(PDFdataset,-2,x2,QQ) + xFxQ2(PDFdataset,1,x1,QQ)*xFxQ2(PDFdataset,-1,x2,QQ) + xFxQ2(PDFdataset,3,x1,QQ)*xFxQ2(PDFdataset,-3,x2,QQ)
        test_vpp=(p2unp*alpha*(np.square(QT))*np.exp(-np.square(QT)/(2*p2bm)))/(2*(np.square(Mp))*p2bm*beta*(np.square(QT))*np.exp(-np.square(QT)/(2*p2unp)))
        return test_vpp
    int_result,err=sciint.dblquad(vppQTx1x2,x1low,x1high,x2low,x2high)
    return int_result                                                                          
#print(vppQT(2,10,cc=0.5,p2bm=2,Hu=1,Hubar=-1,Hd=1,Hdbar=1,Hs=1,Hsbar=1))


###### x1 dependence #######################
def vppx1(x1,QQ,**parms):
    QTlow=2
    QThigh=5
    x2low=0.5
    x2high=0.1
    CC= parms["cc"]
    p2bm=parms["p2bm"]
    Hu= parms["Hu"]
    Hubar= parms["Hubar"]
    Hd= parms["Hd"]
    Hdbar= parms["Hdbar"]
    Hs= parms["Hs"]
    Hsbar= parms["Hsbar"]
    def vppQTx2(QT,x2):
        alpha=(x1**CC)*(1-x1)*(x2**CC)*(1-x2)*(4*Hu*xFxQ2(PDFdataset,2,x1,QQ)*Hubar*xFxQ2(PDFdataset,-2,x2,QQ) + Hd*xFxQ2(PDFdataset,1,x1,QQ)*Hdbar*xFxQ2(PDFdataset,-2,x2,QQ) + Hs*xFxQ2(PDFdataset,3,x1,QQ)*Hsbar*xFxQ2(PDFdataset,-3,x2,QQ))
        beta= 4*xFxQ2(PDFdataset,2,x1,QQ)*xFxQ2(PDFdataset,-2,x2,QQ) + xFxQ2(PDFdataset,1,x1,QQ)*xFxQ2(PDFdataset,-1,x2,QQ) + xFxQ2(PDFdataset,3,x1,QQ)*xFxQ2(PDFdataset,-3,x2,QQ)
        test_vpp=(p2unp*alpha*(np.square(QT))*np.exp(-np.square(QT)/(2*p2bm)))/(2*(np.square(Mp))*p2bm*beta*(np.square(QT))*np.exp(-np.square(QT)/(2*p2unp)))
        return test_vpp
    int_result,err=sciint.dblquad(vppQTx2,x2low,x2high,QTlow,QThigh)
    return int_result                                                                          
#print(vppx1(0.5,10,cc=1,p2bm=2,Hu=1,Hubar=1,Hd=1,Hdbar=1,Hs=1,Hsbar=1))


###### x2 dependence #######################
def vppx2(x2,QQ,**parms):
    QTlow=2
    QThigh=5
    x1low=0.5
    x1high=0.1
    CC= parms["cc"]
    p2bm=parms["p2bm"]
    Hu= parms["Hu"]
    Hubar= parms["Hubar"]
    Hd= parms["Hd"]
    Hdbar= parms["Hdbar"]
    Hs= parms["Hs"]
    Hsbar= parms["Hsbar"]
    def vppQTx1(QT,x1):
        alpha=(x1**CC)*(1-x1)*(x2**CC)*(1-x2)*(4*Hu*xFxQ2(PDFdataset,2,x1,QQ)*Hubar*xFxQ2(PDFdataset,-2,x2,QQ) + Hd*xFxQ2(PDFdataset,1,x1,QQ)*Hdbar*xFxQ2(PDFdataset,-2,x2,QQ) + Hs*xFxQ2(PDFdataset,3,x1,QQ)*Hsbar*xFxQ2(PDFdataset,-3,x2,QQ))
        beta= 4*xFxQ2(PDFdataset,2,x1,QQ)*xFxQ2(PDFdataset,-2,x2,QQ) + xFxQ2(PDFdataset,1,x1,QQ)*xFxQ2(PDFdataset,-1,x2,QQ) + xFxQ2(PDFdataset,3,x1,QQ)*xFxQ2(PDFdataset,-3,x2,QQ)
        test_vpp=(p2unp*alpha*(np.square(QT))*np.exp(-np.square(QT)/(2*p2bm)))/(2*(np.square(Mp))*p2bm*beta*(np.square(QT))*np.exp(-np.square(QT)/(2*p2unp)))
        return test_vpp
    int_result,err=sciint.dblquad(vppQTx1,x1low,x1high,QTlow,QThigh)
    return int_result                                                                          
#print(vppx2(0.5,10,cc=1,p2bm=2,Hu=1,Hubar=1,Hd=1,Hdbar=1,Hs=1,Hsbar=1))
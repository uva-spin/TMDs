import numpy as np

Num_Replicas =  50


eu2 = (2/3)**2
ed2 = (-1/3)**2
es2 = (-1/3)**2



def fDNNQ(QM, b=0.5):
    return b*QM

# def fDNNQ(QM, b=4):
#     return 10*np.exp(-((QM-b)**2)/b**2)

# def fDNNQ(QM, b=0.1):
#     return 10*np.exp(-b*QM)
    
# def fDNNQ(QM, b=0.5):
#     return b*QM


alpha = 1/137

hc_factor = 3.89 * 10**8

factor = ((4*np.pi*alpha)**2)/(9*2*np.pi)


# Compute S(qT) Contribution
mm = 0.55


## Sk1 ####
def Sk(k):
    return ((k**2)/(mm*np.pi))*np.exp(-(k**2)/mm)


# ## Sk2 ####
# def Sk(k):
#     return ((1)/(mm*np.pi))*np.exp(-(k**2)/mm)

# SqT 1 ##
def SqT(qT):
    return (8 * mm * mm + qT**4) / (32 * np.pi * mm) * np.exp(-qT**2 / (2 * mm))


# ## SqT 2 ##
# def SqT(qT):
#     return 1/ (2 * np.pi * mm) * np.exp(-qT**2 / (2 * mm))

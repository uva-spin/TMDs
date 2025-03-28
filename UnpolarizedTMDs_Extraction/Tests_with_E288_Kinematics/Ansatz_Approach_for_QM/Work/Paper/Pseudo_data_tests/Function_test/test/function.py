import numpy as np


def fDNNQ(QM, b=5, c=2):
    return 10 / (1 + ((QM - b) / b)**c) 


# def fDNNQ(QM, b=4):
#     return 10*np.exp(-((QM-b)**2)/b**2)

# def fDNNQ(QM, b=0.1):
#     return 10*np.exp(-b*QM)
    
# def fDNNQ(QM, b=0.5):
#     return b*QM

import numpy as np

# def fDNNQ(QM, b=4):
#     return 10*np.exp(-((QM-b)**2)/b**2)

# def fDNNQ(QM, b=0.1):
#     return 10*np.exp(-b*QM)
    
def fDNNQ(QM, b=0.5):
    return b*QM

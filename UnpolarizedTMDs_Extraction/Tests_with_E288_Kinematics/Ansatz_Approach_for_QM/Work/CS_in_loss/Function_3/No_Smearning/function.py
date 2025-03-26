import numpy as np

# def fDNNQ(QM, b=4):
#     return 10*np.exp(-((QM-b)**2)/b**2)

def fDNNQ(QM, a=5, mu=6.0, sigma=2):
    return a*np.exp(-(QM-mu)**2/(2*sigma**2))
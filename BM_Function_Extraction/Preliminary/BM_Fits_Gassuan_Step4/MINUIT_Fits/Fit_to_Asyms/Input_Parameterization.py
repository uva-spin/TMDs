import numpy as np


# m1v = 1

# Nuv = 1
# auv = 1
# buv = 1

# Nubv = 1
# aubv = 1
# bubv = 1

# Ndv = 1
# adv = 1
# bdv = 1

# Ndbv = 1
# adbv = 1
# bdbv = 1

# Nsv = 1
# asv = 1
# bsv = 1

# Nsbv = 1
# asbv = 1
# bsbv = 1

### NNq parameterization ####

# def NNq(x,Nq,aq,bq):
#     tempNNq = Nq*(x**aq)*((1-x)**(bq))
#     return tempNNq

# def NNqbar(x,Nq,aq,bq):
#     tempNNq = Nq*(x**aq)*((1-x)**(bq))
#     return tempNNq


def NNq(x,Nq,aq,bq):
    tempNNq = Nq*(x**aq)*((1-x)**(bq))*((aq+bq)**(aq+bq))/((aq**aq)*(bq**bq))
    return tempNNq

def NNqbar(x,Nq,aq,bq):
    tempNNq = Nq*(x**aq)*((1-x)**(bq))*((aq+bq)**(aq+bq))/((aq**aq)*(bq**bq))
    return tempNNq


##### SIGN of DY-Sivers relative to SIDIS-Sivers ######
SIGN = 1    


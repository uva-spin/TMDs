import pandas as pd
import numpy as np

herm9_DATA = pd.read_csv('../Data/HERMES_p_2009.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
herm20_DATA = pd.read_csv('../Data/HERMES_p_2020.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
comp9_DATA = pd.read_csv('../Data/COMPASS_d_2009.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
comp15_DATA = pd.read_csv('../Data/COMPASS_p_2015.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')

def GenPseudo(datasetdf):
    data_dictionary = {'hadron':[],
                      'Q2': [],
                      'x': [],
                      'z': [],
                      'phT': [],
                      '1D_dependence': []}
    temp_hads = pd.unique(datasetdf['hadron'])
    #temp_deps = pd.unique(datasetdf['1D_dependence'])
    for i in temp_hads:
        tempdf=datasetdf[(datasetdf["hadron"]==str(i))&(datasetdf["1D_dependence"]=='x')]
        Npoints = 100
        tempxMin=min(np.array(tempdf['x']))
        tempxMax=max(np.array(tempdf['x']))
        tempxarray=np.array(np.linspace(tempxMin,tempxMax,Npoints))
        tempzarray=np.array([np.mean((np.array(tempdf['z'])))+j*0 for j in range(0,Npoints)])
        tempphTarray=np.array([np.mean((np.array(tempdf['phT'])))+j*0 for j in range(0,Npoints)])
        tempQ2rray=np.array([np.mean((np.array(tempdf['Q2'])))+j*0 for j in range(0,Npoints)])
        tempHADarray = np.array([str(i)]*Npoints)
        tempDEParray = np.array(['x']*Npoints)
        data_dictionary['hadron']+= list(tempHADarray)
        data_dictionary['Q2']+= list(tempQ2rray)
        data_dictionary['x']+= list(tempxarray)
        data_dictionary['z']+= list(tempzarray)
        data_dictionary['phT']+= list(tempphTarray)
        #data_dictionary['phT']+= list(tempdf['phT'])
        data_dictionary['1D_dependence']+= list(tempDEParray)
    return pd.DataFrame(data_dictionary)

H09 = GenPseudo(herm9_DATA)
H09.to_csv('HERMES2009xGrid')
H20 = GenPseudo(herm20_DATA)
H20.to_csv('HERMES2020xGrid')
C09 = GenPseudo(comp9_DATA)
C09.to_csv('COMPASS2009xGrid')
C15 = GenPseudo(comp15_DATA)
C15.to_csv('COMPASS2015xGrid')
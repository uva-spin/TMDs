import tensorflow as tf
import pandas as pd
import numpy as np
#import lhapdf
import matplotlib.pyplot as plt
import os
#import plotly.express as px  # Import plotly express
import plotly.graph_objects as go

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")

folder_name ='Evaluations_TMDs_3D_Plots'
create_folders(str(folder_name))


CSVs_folder = 'Evaluations_TMDs_dfs'



#### .csv files for k=0 #####
dfk00_fxAk_true = pd.read_csv(str(CSVs_folder)+'/k_00_fxAk_true.csv')
dfk00_fxAk_pred = pd.read_csv(str(CSVs_folder)+'/k_00_fxAk_pred.csv')
dfk00_fbxBk_true = pd.read_csv(str(CSVs_folder)+'/k_00_fbxBk_true.csv')
dfk00_fbxBk_true = pd.read_csv(str(CSVs_folder)+'/k_00_fbxBk_pred.csv')
dfk00_fxBk_true = pd.read_csv(str(CSVs_folder)+'/k_00_fxBk_true.csv')
dfk00_fxBk_pred = pd.read_csv(str(CSVs_folder)+'/k_00_fxBk_pred.csv')
dfk00_fbxAk_true = pd.read_csv(str(CSVs_folder)+'/k_00_fbxAk_true.csv')
dfk00_fbxAk_pred = pd.read_csv(str(CSVs_folder)+'/k_00_fbxAk_pred.csv')


#### .csv files for k=0.5 #####
dfk05_fxAk_true = pd.read_csv(str(CSVs_folder)+'/k_05_fxAk_true.csv')
dfk05_fxAk_pred = pd.read_csv(str(CSVs_folder)+'/k_05_fxAk_pred.csv')
dfk05_fbxBk_true = pd.read_csv(str(CSVs_folder)+'/k_05_fbxBk_true.csv')
dfk05_fbxBk_true = pd.read_csv(str(CSVs_folder)+'/k_05_fbxBk_pred.csv')
dfk05_fxBk_true = pd.read_csv(str(CSVs_folder)+'/k_05_fxBk_true.csv')
dfk05_fxBk_pred = pd.read_csv(str(CSVs_folder)+'/k_05_fxBk_pred.csv')
dfk05_fbxAk_true = pd.read_csv(str(CSVs_folder)+'/k_05_fbxAk_true.csv')
dfk05_fbxAk_pred = pd.read_csv(str(CSVs_folder)+'/k_05_fbxAk_pred.csv')


#### .csv files for k=1.0 #####
dfk10_fxAk_true = pd.read_csv(str(CSVs_folder)+'/k_10_fxAk_true.csv')
dfk10_fxAk_pred = pd.read_csv(str(CSVs_folder)+'/k_10_fxAk_pred.csv')
dfk10_fbxBk_true = pd.read_csv(str(CSVs_folder)+'/k_10_fbxBk_true.csv')
dfk10_fbxBk_true = pd.read_csv(str(CSVs_folder)+'/k_10_fbxBk_pred.csv')
dfk10_fxBk_true = pd.read_csv(str(CSVs_folder)+'/k_10_fxBk_true.csv')
dfk10_fxBk_pred = pd.read_csv(str(CSVs_folder)+'/k_10_fxBk_pred.csv')
dfk10_fbxAk_true = pd.read_csv(str(CSVs_folder)+'/k_10_fbxAk_true.csv')
dfk10_fbxAk_pred = pd.read_csv(str(CSVs_folder)+'/k_10_fbxAk_pred.csv')


#### .csv files for k=1.5 #####
dfk15_fxAk_true = pd.read_csv(str(CSVs_folder)+'/k_15_fxAk_true.csv')
dfk15_fxAk_pred = pd.read_csv(str(CSVs_folder)+'/k_15_fxAk_pred.csv')
dfk15_fbxBk_true = pd.read_csv(str(CSVs_folder)+'/k_15_fbxBk_true.csv')
dfk15_fbxBk_true = pd.read_csv(str(CSVs_folder)+'/k_15_fbxBk_pred.csv')
dfk15_fxBk_true = pd.read_csv(str(CSVs_folder)+'/k_15_fxBk_true.csv')
dfk15_fxBk_pred = pd.read_csv(str(CSVs_folder)+'/k_15_fxBk_pred.csv')
dfk15_fbxAk_true = pd.read_csv(str(CSVs_folder)+'/k_15_fbxAk_true.csv')
dfk15_fbxAk_pred = pd.read_csv(str(CSVs_folder)+'/k_15_fbxAk_pred.csv')


#### .csv files for k=2.0 #####
dfk20_fxAk_true = pd.read_csv(str(CSVs_folder)+'/k_20_fxAk_true.csv')
dfk20_fxAk_pred = pd.read_csv(str(CSVs_folder)+'/k_20_fxAk_pred.csv')
dfk20_fbxBk_true = pd.read_csv(str(CSVs_folder)+'/k_20_fbxBk_true.csv')
dfk20_fbxBk_true = pd.read_csv(str(CSVs_folder)+'/k_20_fbxBk_pred.csv')
dfk20_fxBk_true = pd.read_csv(str(CSVs_folder)+'/k_20_fxBk_true.csv')
dfk20_fxBk_pred = pd.read_csv(str(CSVs_folder)+'/k_20_fxBk_pred.csv')
dfk20_fbxAk_true = pd.read_csv(str(CSVs_folder)+'/k_20_fbxAk_true.csv')
dfk20_fbxAk_pred = pd.read_csv(str(CSVs_folder)+'/k_20_fbxAk_pred.csv')



def create_3D_Comparison_Plots(output_file_name,strX,df1,df2,df3,df4,df5,df6,df7,df8,df9,df10):
    sc01 = go.Scatter3d(x=df1[str(strX)], y=df1['k'], z=df1['fxk'], mode='markers', name = 'fxk_true', marker=dict(size=2, color='blue'))
    sc02 = go.Scatter3d(x=df2[str(strX)], y=df2['k'], z=df2['fxk'], mode='markers', name = 'fxk_pred', marker=dict(size=2, color='red'))
    sc03 = go.Scatter3d(x=df3[str(strX)], y=df3['k'], z=df3['fxk'], mode='markers', name = 'fxk_true', marker=dict(size=2, color='blue'))
    sc04 = go.Scatter3d(x=df4[str(strX)], y=df4['k'], z=df4['fxk'], mode='markers', name = 'fxk_pred', marker=dict(size=2, color='red'))  
    sc05 = go.Scatter3d(x=df5[str(strX)], y=df5['k'], z=df5['fxk'], mode='markers', name = 'fxk_true', marker=dict(size=2, color='blue'))
    sc06 = go.Scatter3d(x=df6[str(strX)], y=df6['k'], z=df6['fxk'], mode='markers', name = 'fxk_pred', marker=dict(size=2, color='red'))
    sc07 = go.Scatter3d(x=df7[str(strX)], y=df7['k'], z=df7['fxk'], mode='markers', name = 'fxk_true', marker=dict(size=2, color='blue'))
    sc08 = go.Scatter3d(x=df8[str(strX)], y=df8['k'], z=df8['fxk'], mode='markers', name = 'fxk_pred', marker=dict(size=2, color='red'))   
    sc09 = go.Scatter3d(x=df9[str(strX)], y=df9['k'], z=df9['fxk'], mode='markers', name = 'fxk_true', marker=dict(size=2, color='blue'))
    sc10 = go.Scatter3d(x=df10[str(strX)], y=df10['k'], z=df10['fxk'], mode='markers', name = 'fxk_pred', marker=dict(size=2, color='red'))   
    fig = go.Figure(data=[sc01,sc02,sc03,sc04,sc05,sc06,sc07,sc08,sc09,sc10])
    fig.write_html(str(folder_name)+'/'+str(output_file_name)+'.html')


create_3D_Comparison_Plots('test','xA',dfk00_fxAk_true,dfk00_fxAk_pred,dfk05_fxAk_true,dfk05_fxAk_pred,dfk10_fxAk_true,dfk10_fxAk_pred,dfk15_fxAk_true,dfk15_fxAk_pred,dfk20_fxAk_true,dfk20_fxAk_pred)
import tensorflow as tf
import pandas as pd
import numpy as np
#import lhapdf
import matplotlib.pyplot as plt
import os

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")


create_folders('Evaluations_TMDs')


def mse_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return mse_loss


model = tf.keras.models.load_model('model.h5',custom_objects={'mse_loss': mse_loss})
modnnu = model.get_layer('nnu')
modnnubar = model.get_layer('nnubar')


## Load the pdf grids
pdfs0 = pd.read_csv('Eval_PDFs/Eval_pdfs_0.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
pdfs1 = pd.read_csv('Eval_PDFs/Eval_pdfs_1.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
pdfs2 = pd.read_csv('Eval_PDFs/Eval_pdfs_2.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
pdfs3 = pd.read_csv('Eval_PDFs/Eval_pdfs_3.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
pdfs4 = pd.read_csv('Eval_PDFs/Eval_pdfs_4.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
pdfs5 = pd.read_csv('Eval_PDFs/Eval_pdfs_5.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
pdfs6 = pd.read_csv('Eval_PDFs/Eval_pdfs_6.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
pdfs7 = pd.read_csv('Eval_PDFs/Eval_pdfs_7.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
pdfs8 = pd.read_csv('Eval_PDFs/Eval_pdfs_8.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
pdfs9 = pd.read_csv('Eval_PDFs/Eval_pdfs_9.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')

#### S(k) used for pseudo-data ####

def Skq(k):
    return np.exp(-4*k**2/(4*k**2 + 4))

def Skqbar(k):
    return np.exp(-4*k**2/(4*k**2 + 1))


def Generate_Comparison_Plots(df,output_name):
    x1vals = df['xA']
    x2vals = df['xB']
    pTvals = df['PT']
    QMvals = df['QM']

    fu_xA = df['fu_xA']
    fubar_xB = df['fubar_xB']
    Kvals = np.linspace(0.1,2,len(x1vals))
    pT_k_vals = pTvals - Kvals

    true_values_1 = fu_xA * Skq(Kvals)  
    true_values_2 = fubar_xB * Skqbar(pT_k_vals) 

    concatenated_inputs_1 = np.column_stack((x1vals,Kvals,QMvals))
    concatenated_inputs_2 = np.column_stack((x2vals,pT_k_vals,QMvals))

    predicted_values_1 = modnnu.predict(concatenated_inputs_1)
    predicted_values_2 = modnnubar.predict(concatenated_inputs_2)

    plt.figure(1, figsize=(10, 6))
    plt.plot(x1vals, true_values_1, label='True nnu', linestyle='--')
    plt.plot(x1vals, predicted_values_1, label='Predicted nnu')
    plt.title('Comparison of True and Predicted nnu Values')
    plt.xlabel('x1')
    plt.ylabel('nnu')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig('Evaluations_TMDs/True_Pred_fxk_q_'+str(output_name)+'.pdf')
    plt.close()

    plt.figure(2, figsize=(10, 6))
    plt.plot(x2vals, true_values_2, label='True nnubar', linestyle='--')
    plt.plot(x2vals, predicted_values_2, label='Predicted nnubar')
    plt.title('Comparison of True and Predicted nnubar Values')
    plt.xlabel('x2')
    plt.ylabel('nnubar')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig('Evaluations_TMDs/True_Pred_fxk_qbar_'+str(output_name)+'.pdf')
    plt.close()


Generate_Comparison_Plots(pdfs0,'set_0')
Generate_Comparison_Plots(pdfs0,'set_1')
Generate_Comparison_Plots(pdfs0,'set_2')
Generate_Comparison_Plots(pdfs0,'set_3')
Generate_Comparison_Plots(pdfs0,'set_4')
Generate_Comparison_Plots(pdfs0,'set_5')
Generate_Comparison_Plots(pdfs0,'set_6')
Generate_Comparison_Plots(pdfs0,'set_7')
Generate_Comparison_Plots(pdfs0,'set_8')
Generate_Comparison_Plots(pdfs0,'set_9')

# x1vals = pdfs1['xA']
# x2vals = pdfs1['xB']
# pTvals = pdfs1['PT']
# QMvals = pdfs1['QM']
# kk_values_loss = np.array(np.linspace(0,0,len(x1vals)))

# fu_xA = pdfs1['fu_xA']
# fubar_xB = pdfs1['fubar_xB']
# Kvals = np.linspace(0.1,2,len(x1vals))
# pT_k_vals = pTvals - Kvals

# true_values_1 = fu_xA * Skq(Kvals)  
# true_values_2 = fubar_xB * Skqbar(pT_k_vals) 

# concatenated_inputs_1 = np.column_stack((x1vals,Kvals,QMvals))
# concatenated_inputs_2 = np.column_stack((x2vals,pT_k_vals,QMvals))

# predicted_values_1 = modnnu.predict(concatenated_inputs_1)
# predicted_values_2 = modnnubar.predict(concatenated_inputs_2)


# plt.figure(3, figsize=(10, 6))
# plt.plot(x1vals, true_values_1, label='True nnu', linestyle='--')
# plt.plot(x1vals, predicted_values_1, label='Predicted nnu')
# plt.title('Comparison of True and Predicted nnu Values')
# plt.xlabel('x1')
# plt.ylabel('nnu')
# plt.legend()
# plt.grid(True)
# #plt.show()
# plt.savefig('True_Pred_fxk_q.pdf')

# plt.figure(4, figsize=(10, 6))
# plt.plot(x2vals, true_values_2, label='True nnubar', linestyle='--')
# plt.plot(x2vals, predicted_values_2, label='Predicted nnubar')
# plt.title('Comparison of True and Predicted nnubar Values')
# plt.xlabel('x2')
# plt.ylabel('nnubar')
# plt.legend()
# plt.grid(True)
# #plt.show()
# plt.savefig('True_Pred_fxk_qbar.pdf')
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


create_folders('Evaluations_TMDs_DataPoints')


Models_folder = 'DNNmodels'
folders_array=os.listdir(Models_folder)
numreplicas=len(folders_array)
# numreplicas=3

def mse_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return mse_loss


modelsArray = []
for i in range(numreplicas):
    testmodel = tf.keras.models.load_model(str(Models_folder)+'/' + str(folders_array[i]),custom_objects={'mse_loss': mse_loss})
    modelsArray.append(testmodel)

modelsArray = np.array(modelsArray)

# model = tf.keras.models.load_model('model.h5',custom_objects={'mse_loss': mse_loss})
# modnnu = model.get_layer('nnu')
# modnnubar = model.get_layer('nnubar')


########### Import pseudodata file 
df = pd.read_csv('E288_pseudo_data.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')

## Load the pdf grids
# pdfs0 = pd.read_csv('Eval_PDFs/Eval_pdfs_0.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# pdfs1 = pd.read_csv('Eval_PDFs/Eval_pdfs_1.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# pdfs2 = pd.read_csv('Eval_PDFs/Eval_pdfs_2.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# pdfs3 = pd.read_csv('Eval_PDFs/Eval_pdfs_3.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# pdfs4 = pd.read_csv('Eval_PDFs/Eval_pdfs_4.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# pdfs5 = pd.read_csv('Eval_PDFs/Eval_pdfs_5.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# pdfs6 = pd.read_csv('Eval_PDFs/Eval_pdfs_6.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# pdfs7 = pd.read_csv('Eval_PDFs/Eval_pdfs_7.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# pdfs8 = pd.read_csv('Eval_PDFs/Eval_pdfs_8.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# pdfs9 = pd.read_csv('Eval_PDFs/Eval_pdfs_9.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')

#### S(k) used for pseudo-data ####

def Skq(k):
    return np.exp(-4*k**2/(4*k**2 + 4))

def Skqbar(k):
    return np.exp(-4*k**2/(4*k**2 + 1))




# def xsivdistFromReplicas(numReplicas, x, QQ, kperp2avg, kperp):
#     tempfu = []
#     tempfd = []
#     tempfs = []
#     tempfubar = []
#     tempfdbar = []
#     tempfsbar = []
#     for i in range(numReplicas):
#         #t = tf.keras.models.load_model(Models_folder+'/'+ str(folders_array[i]), 
#         #                                 custom_objects={'A0': A0, 'Quotient': Quotient})
#         t = SIDISmodelsArray[i]
#         tempfu.append(list(xsivdist(t, x, QQ, kperp2avg, 2, kperp)))
#         tempfd.append(list(xsivdist(t, x, QQ, kperp2avg, 1, kperp)))
#         tempfs.append(list(xsivdist(t, x, QQ, kperp2avg, 3, kperp)))
#         tempfubar.append(list(xsivdist(t, x, QQ, kperp2avg, -2, kperp)))
#         tempfdbar.append(list(xsivdist(t, x, QQ, kperp2avg, -1, kperp)))
#         tempfsbar.append(list(xsivdist(t, x, QQ, kperp2avg, -3, kperp)))
#     return np.array(tempfu),np.array(tempfubar),np.array(tempfd),np.array(tempfdbar),np.array(tempfs),np.array(tempfsbar)


def Generate_Comparison_Plots(df, num_replicas, output_name):
    x1vals = df['x1']
    x2vals = df['x2']
    pTvals = df['pT']
    QMvals = df['QM']

    Avals = df['A']

    fu_xA = df['fu_xA']
    fubar_xB = df['fubar_xB']

    fu_xB = df['fu_xB']
    fubar_xA = df['fubar_xA']

    Kvals = np.linspace(1,1,len(x1vals))
    #Kvals = np.linspace(0,0,len(x1vals))
    # pT_k_vals = pTvals - Kvals
    pT_k_vals = Kvals

    # true_values_1 = fu_xA * Skq(Kvals)  
    # true_values_2 = fubar_xB * Skqbar(pT_k_vals) 

    true_values_1 = fu_xA * Skq(Kvals)
    true_values_2 = fubar_xB * Skqbar(pT_k_vals) 

    true_values_1_rev = fu_xB * Skq(Kvals)
    true_values_2_rev = fubar_xA * Skqbar(pT_k_vals) 

    concatenated_inputs_1 = np.column_stack((x1vals,Kvals,QMvals))
    concatenated_inputs_2 = np.column_stack((x2vals,pT_k_vals,QMvals))


    #concatenated_inputs_for_sigma = np.column_stack((x1vals, x2vals,pTvals,QMvals))

    # predicted_values_1 = modnnu.predict(concatenated_inputs_1)
    # predicted_values_2 = modnnubar.predict(concatenated_inputs_2)

    tempfu = []
    tempfubar = []

    tempfu_rev = []
    tempfubar_rev = []

    sigma = []

    for i in range(num_replicas):
        t = modelsArray[i]
        modnnu = t.get_layer('nnu')
        modnnubar = t.get_layer('nnubar')
        temp_pred_fu = modnnu.predict(concatenated_inputs_1)
        temp_pred_fubar = modnnubar.predict(concatenated_inputs_2)
        tempfu.append(list(temp_pred_fu))
        tempfubar.append(list(temp_pred_fubar))

        temp_pred_fu_rev = modnnu.predict(concatenated_inputs_2)
        temp_pred_fubar_rev = modnnubar.predict(concatenated_inputs_1)
        tempfu_rev.append(list(temp_pred_fu_rev))
        tempfubar_rev.append(list(temp_pred_fubar_rev))

        # temp_sigma = t.predict([x1vals, x2vals,pTvals,QMvals])
        # sigma.append(temp_sigma[0]) # Index 0 because we only need the cross-section
     

    tempfu = np.array(tempfu)
    tempfu_mean = np.array(tempfu.mean(axis=0))
    tempfu_err = np.array(tempfu.std(axis=0))

    tempfubar = np.array(tempfubar)
    tempfubar_mean = np.array(tempfubar.mean(axis=0))
    tempfubar_err = np.array(tempfubar.std(axis=0))


    tempfu_rev = np.array(tempfu_rev)
    tempfu_mean_rev = np.array(tempfu_rev.mean(axis=0))
    tempfu_err_rev = np.array(tempfu_rev.std(axis=0))

    tempfubar_rev = np.array(tempfubar_rev)
    tempfubar_mean_rev = np.array(tempfubar_rev.mean(axis=0))
    tempfubar_err_rev = np.array(tempfubar_rev.std(axis=0))


    # tempA = np.array(sigma)
    # tempA_mean = np.array(tempA.mean(axis=0))
    # tempA_mean = np.array(tempA_mean.flatten())

    # return (tempfu_mean-tempfu_err).flatten()

    # print(len(true_values_1))
    # print(len(tempfu_mean))


    plt.figure(1, figsize=(10, 6))
    plt.plot(x1vals, true_values_1, 'b.', label='True nnu')
    plt.plot(x1vals, tempfu_mean, 'r.', label='Predicted nnu')
    #plt.fill_between(x1vals, (tempfu_mean-tempfu_err).flatten(), (tempfu_mean+tempfu_err).flatten(), facecolor='r', alpha=0.3)
    plt.title('Comparison of True and Predicted nnu TMD Values')
    plt.xlabel('x1')
    plt.ylabel('nnu')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig('Evaluations_TMDs_DataPoints/True_Pred_fx_q_'+str(output_name)+'.pdf')
    plt.close()

    plt.figure(2, figsize=(10, 6))
    plt.plot(x2vals, true_values_2, 'b.', label='True nnubar')
    plt.plot(x2vals, tempfubar_mean, 'r.', label='Predicted nnubar')
    #plt.fill_between(x2vals, (tempfubar_mean-tempfubar_err).flatten(), (tempfubar_mean+tempfubar_err).flatten(), facecolor='r', alpha=0.3)
    plt.title('Comparison of True and Predicted nnubar TMD Values')
    plt.xlabel('x2')
    plt.ylabel('nnubar')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig('Evaluations_TMDs_DataPoints/True_Pred_fx_qbar_'+str(output_name)+'.pdf')
    plt.close()


    plt.figure(3, figsize=(10, 6))
    plt.plot(x2vals, true_values_1_rev, 'b.', label='True nnu')
    plt.plot(x2vals, tempfu_mean_rev, 'r.', label='Predicted nnu')
    #plt.fill_between(x2vals, (tempfu_mean_rev-tempfu_err_rev).flatten(), (tempfu_mean_rev+tempfu_err_rev).flatten(), facecolor='r', alpha=0.3)
    plt.title('Comparison of True and Predicted nnu TMD Values')
    plt.xlabel('x2')
    plt.ylabel('nnu')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig('Evaluations_TMDs_DataPoints/True_Pred_fx_q_rev_'+str(output_name)+'.pdf')
    plt.close()

    plt.figure(4, figsize=(10, 6))
    plt.plot(x1vals, true_values_2_rev, 'b.', label='True nnubar')
    plt.plot(x1vals, tempfubar_mean_rev, 'r.', label='Predicted nnubar')
    #plt.fill_between(x1vals, (tempfubar_mean_rev-tempfubar_err_rev).flatten(), (tempfubar_mean_rev+tempfubar_err_rev).flatten(), facecolor='r', alpha=0.3)
    plt.title('Comparison of True and Predicted nnubar TMD Values')
    plt.xlabel('x1')
    plt.ylabel('nnubar')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig('Evaluations_TMDs_DataPoints/True_Pred_fx_qbar_rev_'+str(output_name)+'.pdf')
    plt.close()


    # # 3D scatter plot
    # fig = plt.figure(5)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x1vals, x2vals, Avals, c='r', marker='o', label='Actual')
    # # Plot the model predictions
    # ax.scatter(x1vals, x2vals, tempA_mean, c='b', marker='^', label='Predicted')
    # ax.set_xlabel('x1')
    # ax.set_ylabel('x2')
    # ax.set_zlabel('A')
    # ax.set_title('Actual vs Predicted')
    # ax.legend()
    # #plt.show()
    # plt.savefig('Evaluations_TMDs_DataPoints/Actual_vs_Predicted_CS_'+str(output_name)+'.pdf')
    # plt.close()

Generate_Comparison_Plots(df,numreplicas,'points')

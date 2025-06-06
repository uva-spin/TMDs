import tensorflow as tf
import pandas as pd
import numpy as np
#import lhapdf
import matplotlib.pyplot as plt
import os




####################################################

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")


create_folders('Evaluations_TMDs_with_kval')


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


## Load the pdf grids
pdfs0 = pd.read_csv('Eval_PDFs/Eval_pdfs_0.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
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



def nnq(model, flavor, kins):
    # For example: flavor = {'nnu', 'nnubar'}
    mod_out = tf.keras.backend.function(model.get_layer(flavor).input,
                                       model.get_layer(flavor).output)
    return mod_out(kins)

def Generate_Comparison_Plots_for_Kvalue(df, num_replicas, kvalue, output_name):
    x1vals = df['xA']
    x2vals = df['xB']
    pTvals = df['PT']
    QMvals = df['QM']

    fu_xA = df['fu_xA']
    fubar_xB = df['fubar_xB']

    fu_xB = df['fu_xB']
    fubar_xA = df['fubar_xA']


    fu_xA = np.array(fu_xA)
    fubar_xB = np.array(fubar_xB)

    fu_xB = np.array(fu_xB)
    fubar_xA = np.array(fubar_xA)

    #Kvals = np.linspace(0.1,2,len(x1vals))
    #Kvals = np.linspace(0,0,len(x1vals))
    Kvals = np.array([i*0 + kvalue for i in range(len(x1vals))])
    #pTvals = np.array([i*0 + 5 for i in range(len(pTvals))])
    #pT_k_vals = pTvals - Kvals
    pT_k_vals = Kvals

    true_values_1 = fu_xA * Skq(Kvals)  
    true_values_2 = fubar_xB * Skqbar(pT_k_vals) 

    # true_values_1 = fu_xA 
    # true_values_2 = fubar_xB

    true_values_1_rev = fu_xB * Skq(Kvals)
    true_values_2_rev = fubar_xA * Skqbar(pT_k_vals) 

    concatenated_inputs_1 = np.column_stack((Kvals,QMvals))
    concatenated_inputs_2 = np.column_stack((pT_k_vals,QMvals))


    tempfu = []
    tempfubar = []

    tempfu_rev = []
    tempfubar_rev = []

    for i in range(num_replicas):
        t = modelsArray[i]
        # modnnu = t.get_layer('nnu')
        # modnnubar = t.get_layer('nnubar')
        # temp_pred_fu = modnnu.predict(concatenated_inputs_1)
        # temp_pred_fubar = modnnubar.predict(concatenated_inputs_2)
        temp_pred_Su = nnq(t, 'nnu', concatenated_inputs_1)
        temp_pred_Subar = nnq(t, 'nnubar', concatenated_inputs_2)
        tempfu.append(temp_pred_Su)
        tempfubar.append(temp_pred_Subar)

        temp_pred_Su_rev = nnq(t, 'nnu', concatenated_inputs_2)
        temp_pred_Subar_rev = nnq(t, 'nnubar', concatenated_inputs_1)
        tempfu_rev.append(temp_pred_Su_rev)
        tempfubar_rev.append(temp_pred_Subar_rev)


        # temp_pred_Subar = (nnq(t, 'nnubar', concatenated_inputs_2)).flatten()
        # #print(np.array(fu_xA)*temp_pred_Su)
        # tempfu.append(np.array(fu_xA)*np.array(temp_pred_Su))
        # tempfubar.append( np.array(fubar_xB)*temp_pred_Subar)

        # temp_pred_Su_rev = nnq(t, 'nnu', concatenated_inputs_2)
        # temp_pred_Subar_rev = nnq(t, 'nnubar', concatenated_inputs_1)
        # tempfu_rev.append((np.array(fu_xB)*temp_pred_Su_rev).flatten())
        # tempfubar_rev.append((np.array(fubar_xA)*temp_pred_Subar_rev).flatten())
    

    # print(len(tempfu_mean))
    # print(tempfu_mean)
    # print(len(fu_xA))
    # print(np.array(fu_xA))
    # print(np.array(fu_xA)*(tempfu_mean.flatten()))
    #print(tempfu_mean)

    tempfu = np.array(tempfu)
    tempfu_mean = np.array(tempfu.mean(axis=0))
    tempfu_mean = fu_xA*(tempfu_mean.flatten())
    tempfu_err = np.array(tempfu.std(axis=0))

    tempfubar = np.array(tempfubar)
    tempfubar_mean = np.array(tempfubar.mean(axis=0))
    tempfubar_mean = fubar_xB*(tempfubar_mean.flatten())
    tempfubar_err = np.array(tempfubar.std(axis=0))

    tempfu_rev = np.array(tempfu_rev)
    tempfu_mean_rev = np.array(tempfu_rev.mean(axis=0))
    tempfu_mean_rev = fu_xB*(tempfu_mean_rev.flatten())
    tempfu_err_rev = np.array(tempfu_rev.std(axis=0))

    tempfubar_rev = np.array(tempfubar_rev)
    tempfubar_mean_rev = np.array(tempfubar_rev.mean(axis=0))
    tempfubar_mean_rev = fubar_xA*(tempfubar_mean_rev.flatten())
    tempfubar_err_rev = np.array(tempfubar_rev.std(axis=0))



    plt.figure(1, figsize=(20, 5))
    #################
    plt.subplot(1,4,1)
    plt.plot(x1vals, true_values_1, label='True nnu', linestyle='--')
    plt.plot(x1vals, tempfu_mean, 'r', label='Predicted nnu')
    #plt.fill_between(x1vals, (tempfu_mean-tempfu_err).flatten(), (tempfu_mean+tempfu_err).flatten(), facecolor='r', alpha=0.3)
    plt.title(f'f(xA, k ={kvalue})')
    plt.xlabel('xA')
    plt.ylabel('nnu')
    plt.legend()
    plt.grid(True)
    #################
    plt.subplot(1,4,2)
    plt.plot(x2vals, true_values_2, label='True nnubar', linestyle='--')
    plt.plot(x2vals, tempfubar_mean, 'r', label='Predicted nnubar')
    #plt.fill_between(x2vals, (tempfubar_mean-tempfubar_err).flatten(), (tempfubar_mean+tempfubar_err).flatten(), facecolor='r', alpha=0.3)
    #plt.title('Comparison of True and Predicted nnubar PDF Values')
    plt.xlabel('xB')
    plt.ylabel('nnubar')
    plt.legend()
    plt.title(f'fbar(xB, k ={kvalue})')
    plt.grid(True)
    ################
    plt.subplot(1,4,3)
    plt.plot(x2vals, true_values_1_rev, label='True nnu', linestyle='--')
    plt.plot(x2vals, tempfu_mean_rev, 'r', label='Predicted nnu')
    #plt.fill_between(x2vals, (tempfu_mean_rev-tempfu_err_rev).flatten(), (tempfu_mean_rev+tempfu_err_rev).flatten(), facecolor='r', alpha=0.3)
    plt.title(f'f(xB, k ={kvalue})')
    plt.xlabel('xB')
    plt.ylabel('nnu')
    plt.legend()
    plt.grid(True)
    ################
    plt.subplot(1,4,4)
    plt.plot(x1vals, true_values_2_rev, label='True nnubar', linestyle='--')
    plt.plot(x1vals, tempfubar_mean_rev, 'r', label='Predicted nnubar')
    #plt.fill_between(x1vals, (tempfubar_mean_rev-tempfubar_err_rev).flatten(), (tempfubar_mean_rev+tempfubar_err_rev).flatten(), facecolor='r', alpha=0.3)
    plt.title(f'fbar(xA, k ={kvalue})')
    plt.xlabel('xA')
    plt.ylabel('nnubar')
    plt.legend()
    plt.grid(True) 
    plt.savefig('Evaluations_TMDs_with_kval/True_vs_Pred_fxk_'+str(output_name)+'.pdf')
    plt.close()


Generate_Comparison_Plots_for_Kvalue(pdfs0,numreplicas, 0, 'set_0_k=0')
Generate_Comparison_Plots_for_Kvalue(pdfs0,numreplicas, 0.5, 'set_0_k=0.5')
Generate_Comparison_Plots_for_Kvalue(pdfs0,numreplicas, 1, 'set_0_k=1')
Generate_Comparison_Plots_for_Kvalue(pdfs0,numreplicas, 1.5, 'set_0_k=1.5')
Generate_Comparison_Plots_for_Kvalue(pdfs0,numreplicas, 2, 'set_0_k=2')

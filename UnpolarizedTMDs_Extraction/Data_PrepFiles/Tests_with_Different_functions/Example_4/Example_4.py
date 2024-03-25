######################################################################
## Here we have A(x1,x2,pT) = f(x1)*Sk(k)*f(x2)*Sk(pT-k)
## k-integration:  was done within the function ##########
## Loss function: three types of losses are defined   ##
## (1) MSE
## (2) f(x,k) --> f(x) when k -> 0
## (3) f(x)S(k) = DNN(x,k)
## Here is an example with simple fucntions for f(x) and S(k)
##  f(x) = fu(x) and fubar(x) from NNPDF40
##  S(k) = k + 1
############################

import tensorflow as tf
import pandas as pd
import numpy as np
#import lhapdf
import matplotlib.pyplot as plt
from scipy.integrate import simps


############ Generating Pseudodata #################



lhapdf_df = pd.read_csv('NNPDF4_nlo.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')



def Sk(k):
    return np.exp(-k**2/4)


fu = np.array(lhapdf_df['fu'])
fubar = np.array(lhapdf_df['fubar'])


def f_map(x,f_array):
    mapping = dict(zip(x,f_array))
    return mapping


#x1vals = np.linspace(0.1, 0.3, 10)
x1vals = np.array(lhapdf_df['x'])
x2vals = np.array(lhapdf_df['x'])
pTvals = np.linspace(0.1,2,len(x1vals))
Kvals = np.linspace(0.1,2,len(x1vals))
pT_k_vals = pTvals - Kvals
kk_values_loss = np.array(np.linspace(0,0,len(x1vals)))


def fu_val(x):
    map = f_map(x1vals,fu)
    return map.get(x,None)

def fubar_val(x):
    map = f_map(x2vals,fubar)
    return map.get(x,None)

#print(fubar_val(x1vals[0]))

def fx1kx2k(x1,x2,pT,k):
    return fu*Sk(k)*fubar*Sk(pT-k)

# def Apseudo(x1,x2,pT):
#     tempx1, tempx2, temppT, tempA = [], [], [], []
#     kk = np.linspace(0.0,2.0,len(x1vals))
#     for i in range(len(x1)):
#         for j in range(len(x2)):
#             tempx1.append(x1[i])
#             tempx2.append(x2[j])
#             temppT.append(pT[j])
#             tempfx1kfx2k = simps(fx1kx2k(x1[i],x2[j],pT[j],kk), dx=(kk[1]-kk[0]))
#             tempA.append(tempfx1kfx2k)
#     return np.array(tempx1), np.array(tempx2), np.array(temppT), np.array(tempA)

def Apseudo(x1,x2,pT):
    tempx1, tempx2, temppT, tempA = [], [], [], []
    kk = np.linspace(0.0,2.0,len(x1vals))
    for i in range(len(x1)):
        tempx1.append(x1[i])
        tempx2.append(x2[i])
        temppT.append(pT[i])
        tempfx1kfx2k = simps(fx1kx2k(x1[i],x2[i],pT[i],kk), dx=(kk[1]-kk[0]))
        tempA.append(tempfx1kfx2k)
    return np.array(tempx1), np.array(tempx2), np.array(temppT), np.array(tempA)

x1Vals, x2Vals, pTVals, Avals = Apseudo(x1vals,x2vals,pTvals)

df = pd.DataFrame({'x1': x1Vals, 'x2': x2Vals, 'pT': pTVals, 'A': Avals})
#df

Hidden_Layers=5
Nodes_per_HL=256
Learning_Rate = 0.0001
L1_reg = 10**(-12)
EPOCHS = 300


def create_nn_model(name, hidden_layers=Hidden_Layers, width=Nodes_per_HL, activation='tanh'):
    inp = tf.keras.Input(shape=(2))
    x = tf.keras.layers.Dense(width, activation=activation)(inp)
    for i in range(hidden_layers-1):
        x = tf.keras.layers.Dense(width, activation=activation)(x)
    nnout = tf.keras.layers.Dense(1, activation=activation)(x)
    mod = tf.keras.Model(inp, nnout, name=name)
    return mod



def createModel_DY():
    x1 = tf.keras.Input(shape=(1), name='x1')
    x2 = tf.keras.Input(shape=(1), name='x2')
    qT = tf.keras.Input(shape=(1), name='qT')

    modnnu = create_nn_model('nnu')
    modnnubar = create_nn_model('nnubar')

    # Implementing f(x,k) --> f(x) when k-->0
    nnu_pdf = modnnu(tf.keras.layers.Concatenate()([x1, x1*0 ]))
    nnubar_pdf = modnnu(tf.keras.layers.Concatenate()([x2, x2*0 ]))

    k_values = tf.linspace(0.1, 2.0, 100)
    dk = k_values[1] - k_values[0]

    tmd1_list, tmd2_list = [], []
    product_list = []  # List to store TMD values for each k value
    for k_val in k_values:      
        nnu_input = tf.keras.layers.Concatenate()([x1, qT*0 + k_val])
        nnubar_input = tf.keras.layers.Concatenate()([x2, qT - k_val])

        nnu_x1 = modnnu(nnu_input)
        nnubar_x2 = modnnubar(nnubar_input)
        tmd1_list.append(nnu_x1)
        tmd2_list.append(nnubar_x2)
        product_list.append(tf.multiply(nnu_x1, nnubar_x2))

    # Summing over all k values using tf.reduce_sum
    tmd_product_sum = tf.reduce_sum(tf.stack(product_list), axis=0) * dk 

    return tf.keras.Model([x1, x2, qT], [tmd_product_sum, nnu_pdf, nnubar_pdf])


TMD_Model_DY = createModel_DY()

# Testing loss
# loss_function = tf.keras.losses.MeanSquaredError()
# optimizer = tf.keras.optimizers.Adam(learning_rate=Learning_Rate)



# Compile the model
model = createModel_DY()



def pdf1_loss_val(y_true, y_pred):
    modnnu = model.get_layer('nnu')
    concatenated_inputs_1 = np.column_stack((x1vals,kk_values_loss))
    fx1_true = fu
    fx1_result = modnnu(concatenated_inputs_1)
    pdf1_loss = tf.reduce_mean(tf.square(fx1_true - fx1_result))
    return pdf1_loss


def pdf2_loss_val(y_true, y_pred):
    modnnubar = model.get_layer('nnubar')
    concatenated_inputs_2 = np.column_stack((x2vals,kk_values_loss))
    fx2_true = fubar
    fx2_result = modnnubar(concatenated_inputs_2)
    pdf2_loss = tf.reduce_mean(tf.square(fx2_true - fx2_result))
    return pdf2_loss


# def pdf1_loss_val(y_true, y_pred):
#     modnnu = model.get_layer('nnu')
#     concatenated_inputs_1 = np.column_stack((x1vals,kk_values_loss))
#     fx1_true = f(x1vals)
#     fx1_result = modnnu(concatenated_inputs_1)
#     pdf1_loss = tf.reduce_mean(tf.square(fx1_true - fx1_result))
#     return pdf1_loss


# def pdf2_loss_val(y_true, y_pred):
#     modnnubar = model.get_layer('nnubar')
#     concatenated_inputs_2 = np.column_stack((x2vals,kk_values_loss))
#     fx2_true = f(x2vals)
#     fx2_result = modnnubar(concatenated_inputs_2)
#     pdf2_loss = tf.reduce_mean(tf.square(fx2_true - fx2_result))
#     return pdf2_loss



# def prod_1_loss_val(y_true, y_pred):
#     modnnu = model.get_layer('nnu')
#     concatenated_inputs_1 = np.column_stack((x1vals,Kvals))
#     fx1_true = f(x1vals) * Sk(Kvals)
#     fx1_result = modnnu(concatenated_inputs_1)
#     pdf1_loss = tf.reduce_mean(tf.square(fx1_true - fx1_result))
#     return pdf1_loss


# def prod_2_loss_val(y_true, y_pred):
#     modnnubar = model.get_layer('nnubar')
#     concatenated_inputs_2 = np.column_stack((x2vals,pT_k_vals))
#     fx2_true = f(x2vals) * Sk(pT_k_vals)
#     fx2_result = modnnubar(concatenated_inputs_2)
#     pdf2_loss = tf.reduce_mean(tf.square(fx2_true - fx2_result))
#     return pdf2_loss



def mse_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return mse_loss

# def custom_loss(y_true, y_pred):
#     return mse_loss(y_true, y_pred) + pdf1_loss_val(y_true, y_pred) + pdf2_loss_val(y_true, y_pred) + prod_1_loss_val(y_true, y_pred) + prod_2_loss_val(y_true, y_pred) 


# def custom_loss(y_true, y_pred):
#     return prod_1_loss_val(y_true, y_pred) + prod_2_loss_val(y_true, y_pred) 


# def custom_loss(y_true, y_pred):
#     return mse_loss(y_true, y_pred) + pdf1_loss_val(y_true, y_pred) + pdf2_loss_val(y_true, y_pred) 

def custom_loss(y_true, y_pred):
    return mse_loss(y_true, y_pred) 



plt.figure(1, figsize=(10, 6))
plt.plot(x1vals, fu, label='$f_u(x)$')
plt.plot(x1vals, fubar, label='$f_{\\bar{u}}(x)$')
plt.title('NNPDF4 fu and fubar')
plt.xlabel('x')
plt.ylabel('f')
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig('PDFs.pdf')


loss_weights = { 'tf.math.multiply_405': 0.5, "nnu": 1.0, "nnu_1": 1.0}


model.compile(optimizer='adam', loss=custom_loss, loss_weights = loss_weights)
# model.compile(optimizer=optimizer, loss=custom_loss, metrics=[mse_loss,pdf1_loss_val,pdf2_loss_val])
# model.compile(optimizer=optimizer, loss=custom_loss, metrics=[mse_loss, prod_1_loss_val, prod_2_loss_val])

# Train the model on the entire dataset
#history = model.fit([df['x1'],df['x2'],df['pT']], df['A'], epochs=EPOCHS, batch_size=32, verbose=2)
history = model.fit([df['x1'],df['x2'],df['pT']], [df['A'], fu, fubar], epochs=EPOCHS, batch_size=32, verbose=2)


modnnu = model.get_layer('nnu')
modnnubar = model.get_layer('nnubar')

# modnnu = model.predict([df['x1'],df['x2'],df['pT']])[1]
# modnnubar = model.predict([df['x1'],df['x2'],df['pT']])[2]


# true_values_1 = f(x1vals) * Sk(Kvals)  
# true_values_2 = f(x2vals) * Sk(pT_k_vals)  

true_values_1 = fu * Sk(Kvals)  
true_values_2 = fubar * Sk(pT_k_vals) 

concatenated_inputs_1 = np.column_stack((x1vals,Kvals))
concatenated_inputs_2 = np.column_stack((x2vals,pT_k_vals))

# predicted_values_1 = modnnu.predict(concatenated_inputs_1)
# predicted_values_2 = modnnubar.predict(concatenated_inputs_2)

predicted_values_1 = model.predict([df['x1'],df['x2'],df['pT']])[1]
predicted_values_2 = model.predict([df['x1'],df['x2'],df['pT']])[2]

plt.figure(2, figsize=(10, 6))
plt.plot(x1vals, true_values_1, label='True nnu', linestyle='--')
plt.plot(x1vals, predicted_values_1, label='Predicted nnu')
plt.title('Comparison of True and Predicted nnu Values')
plt.xlabel('x1')
plt.ylabel('nnu')
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig('True_Pred_fxk_q.pdf')

plt.figure(3, figsize=(10, 6))
plt.plot(x2vals, true_values_2, label='True nnubar', linestyle='--')
plt.plot(x2vals, predicted_values_2, label='Predicted nnubar')
plt.title('Comparison of True and Predicted nnubar Values')
plt.xlabel('x2')
plt.ylabel('nnubar')
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig('True_Pred_fxk_qbar.pdf')

# 3D scatter plot
fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1vals, Kvals, true_values_1, c='r', marker='o', label='Actual')
# Plot the model predictions
ax.scatter(x1vals, Kvals, predicted_values_1, c='b', marker='^', label='Predicted')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('A')
ax.set_title('Actual vs Predicted')
ax.legend()
#plt.show()
plt.savefig('True_Pred_fxk_q_2D.pdf')



predictions = model.predict([df['x1'],df['x2'],df['pT']])[0]

# 3D scatter plot
fig = plt.figure(6)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1Vals, pTVals, df['A'], c='r', marker='o', label='Actual')
ax.scatter(x1Vals, pTVals, predictions, c='b', marker='^', label='Predicted')
ax.set_xlabel('x1')
ax.set_ylabel('pT')
ax.set_zlabel('A')
ax.set_title('Actual vs Predicted')
ax.legend()
#plt.show()
plt.savefig('Actual_vs_Predicted_Integral.pdf')
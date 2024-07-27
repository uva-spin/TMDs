######################################################################
## Here we have A(x1,x2,pT) = f(x1)*Sk(k)*f(x2)*Sk(pT-k)
## k-integration:  was done within the function ##########
## Loss function: mse ####
############################

import tensorflow as tf
import pandas as pd
import numpy as np
#import lhapdf
import matplotlib.pyplot as plt
from scipy.integrate import simps


############ Generating Pseudodata #################

def f(x):
    return (x**0.1)*((1-x)**0.3)

# def Sk(k):
#     return 2*k**2/(k**2 + 4)

def Sk(k):
    return np.exp(-k**2/4)

def fx1kx2k(x1,x2,pT,k):
    return f(x1)*Sk(k)*f(x2)*Sk(pT-k)

x1vals = np.linspace(0.0001, 0.3, 10)
x2vals = np.linspace(0.1, 0.7, 10)
pTvals = np.linspace(0.1,2,10)

def Apseudo(x1,x2,pT):
    tempx1, tempx2, temppT, tempA = [], [], [], []
    kk = np.linspace(0.0001,2,50)
    for i in range(len(x1)):
        for j in range(len(x2)):
            tempx1.append(x1[i])
            tempx2.append(x2[j])
            temppT.append(pT[j])
            tempfx1kfx2k = simps(fx1kx2k(x1[i],x2[j],pT[j],kk), dx=(kk[1]-kk[0]))
            tempA.append(tempfx1kfx2k)
    return np.array(tempx1), np.array(tempx2), np.array(temppT), np.array(tempA)

x1Vals, x2Vals, pTVals, Avals = Apseudo(x1vals,x2vals,pTvals)

df = pd.DataFrame({'x1': x1Vals, 'x2': x2Vals, 'pT': pTVals, 'A': Avals})
#df

Hidden_Layers=5
Nodes_per_HL=100
Learning_Rate = 0.0001
L1_reg = 10**(-12)
EPOCHS = 300


def create_nn_model(name, hidden_layers=Hidden_Layers, width=Nodes_per_HL, activation='relu6'):
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

    k_values = tf.linspace(0.0, 2.0, 50)  # Generate k values
    dk = k_values[1] - k_values[0]

    tmd_list = []  # List to store TMD values for each k value
    for k_val in k_values:      
        nnu_input = tf.keras.layers.Concatenate()([x1, qT*0 + k_val])
        nnubar_input = tf.keras.layers.Concatenate()([x2, qT - k_val])

        nnu_x1 = tf.abs(modnnu(nnu_input))
        nnubar_x2 = tf.abs(modnnubar(nnubar_input))

        tmd_list.append(dk*tf.keras.layers.Multiply()([nnu_x1, nnubar_x2]))

    # Summing over all k values
    tmd_sum = tf.keras.layers.Add()(tmd_list)

    return tf.keras.Model([x1, x2, qT], tmd_sum)

TMD_Model_DY = createModel_DY()

# Define loss function and optimizer
loss_function = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)



# Compile the model
model = createModel_DY()


x1_values = np.array(np.linspace(0.0001, 0.3, 100))
kk_values = np.array(np.linspace(0.0001, 2, 100))
kk_values_loss = np.array(np.linspace(0,0,100))
x2_values = np.array(np.linspace(0.1, 0.7, 100))
pT_values = np.array(np.linspace(0.1,2,100))
pT_k_values = pT_values - kk_values




def custom_loss(y_true, y_pred):
    # modnnu = tf.keras.Model(inputs=model.input, outputs=model.get_layer('nnu').output)
    # modnnubar = tf.keras.Model(inputs=model.input, outputs=model.get_layer('nnubar').output)
    modnnu = model.get_layer('nnu')
    modnnubar = model.get_layer('nnubar')
    concatenated_inputs_1 = np.column_stack((x1_values,kk_values_loss))
    concatenated_inputs_2 = np.column_stack((x2_values,kk_values_loss))
    fx1_true = f(x1_values) * Sk(kk_values_loss)  
    fx2_true = f(x2_values) * Sk(kk_values_loss)  
    fx1_result = modnnu(concatenated_inputs_1)
    fx2_result = modnnubar(concatenated_inputs_2)
    pdf1_loss = tf.reduce_mean(tf.square(fx1_true - fx1_result))
    pdf2_loss = tf.reduce_mean(tf.square(fx2_true - fx2_result))
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    loss = mse_loss + pdf1_loss + pdf2_loss
    test_print_pdf1 = tf.convert_to_tensor(pdf1_loss,np.float32)
    #print(test_print_pdf1)
    #loss = mse_loss
    #print(f'pdf1_loss: {pdf1_loss}')
    return loss

#model.compile(optimizer=optimizer, loss=loss_function)
model.compile(optimizer=optimizer, loss=custom_loss)

# Train the model on the entire dataset
history = model.fit([df['x1'],df['x2'],df['pT']], df['A'], epochs=EPOCHS, batch_size=32, verbose=2)

# Print pdf1_loss at each epoch
# for epoch in range(len(history.history['loss'])):
#     modnnu = model.get_layer('nnu')
#     modnnubar = model.get_layer('nnubar')
#     concatenated_inputs_1 = np.column_stack((x1_values, kk_values_loss))
#     concatenated_inputs_2 = np.column_stack((x2_values, kk_values_loss))
#     fx1_true = f(x1_values) * Sk(kk_values_loss)
#     fx2_true = f(x2_values) * Sk(kk_values_loss)
#     fx1_result = modnnu(concatenated_inputs_1)
#     fx2_result = modnnubar(concatenated_inputs_2)
#     pdf1_loss = tf.reduce_mean(tf.square(fx1_true - fx1_result))
#     pdf2_loss = tf.reduce_mean(tf.square(fx2_true - fx2_result))
#     #predictions = model.predict([x1vals, x2vals, pTvals])
#     #mse_loss = np.mean((Avals - predictions) ** 2)
#     #print(f'Epoch {epoch + 1}: pdf1_loss = {pdf1_loss}, pdf2_loss = {pdf2_loss}, mse_loss = {mse_loss}')
#     #print(f'pdf1_loss at epoch {epoch + 1}: {pdf1_loss}')
#     print(f'Epoch {epoch + 1}: pdf1_loss = {pdf1_loss}, pdf2_loss = {pdf2_loss}')

modnnu = model.get_layer('nnu')
modnnubar = model.get_layer('nnubar')


true_values_1 = f(x1_values) * Sk(kk_values)  
true_values_2 = f(x2_values) * Sk(pT_k_values)  

concatenated_inputs_1 = np.column_stack((x1_values,kk_values))
concatenated_inputs_2 = np.column_stack((x2_values,pT_k_values))

predicted_values_1 = modnnu.predict(concatenated_inputs_1)
predicted_values_2 = modnnubar.predict(concatenated_inputs_2)

plt.figure(3, figsize=(10, 6))
plt.plot(x1_values, true_values_1, label='True nnu', linestyle='--')
plt.plot(x1_values, predicted_values_1, label='Predicted nnu')
plt.title('Comparison of True and Predicted nnu Values')
plt.xlabel('x1')
plt.ylabel('nnu')
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig('True_Pred_fxk_q.pdf')

plt.figure(4, figsize=(10, 6))
plt.plot(x2_values, true_values_2, label='True nnubar', linestyle='--')
plt.plot(x2_values, predicted_values_2, label='Predicted nnubar')
plt.title('Comparison of True and Predicted nnu Values')
plt.xlabel('x2')
plt.ylabel('nnubar')
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig('True_Pred_fxk_qbar.pdf')
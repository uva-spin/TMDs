import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from kerastuner.tuners import BayesianOptimization
from sklearn.model_selection import train_test_split

# Step 1: Generating Synthetic Data
np.random.seed(42)  # Seed for reproducibility
x1 = np.linspace(-1, 1, 400)
x2 = np.linspace(-1, 1, 400)
y = (0.5 * x1**3 - 0.2 * x1**2 + 0.1 * x1 - 0.2 + 
     0.7 * x2**3 - 0.4 * x2**2 + 0.05 * x2 - 0.2 + np.random.normal(scale=0.05, size=x1.shape))

# Step 2: Building the Model and Optimizing it with Keras Tuner
def build_model(name, hp):
    model = keras.Sequential()
    inp = tf.keras.Input(shape=(1,))
    model.add(inp)  # Input layer
    
    # Tuning the number of layers and their configurations
    for i in range(hp.Int('num_layers', 1, 5)):
        model.add(layers.Dense(
            units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
            activation=hp.Choice('activation_' + str(i), ['relu', 'tanh', 'sigmoid'])
        ))

    model.add(layers.Dense(1))  # Output layer
    return model

def createModel_DY(hp):
    x1 = tf.keras.Input(shape=(1), name='x1')
    x2 = tf.keras.Input(shape=(1), name='x2')

    modnnu = build_model('nnu', hp)
    modnnubar = build_model('nnubar', hp)
   
    nnu_pdf_input = tf.keras.layers.Concatenate()([x1])
    nnubar_pdf_input = tf.keras.layers.Concatenate()([x2])

    modnnu_pdf_eval = modnnu(nnu_pdf_input)
    modnnubar_pdf_eval = modnnubar(nnubar_pdf_input)

    product = tf.multiply(modnnu_pdf_eval, modnnubar_pdf_eval)

    model = tf.keras.Model([x1, x2], [product])

    # Compilation of the model with a tunable learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )

    return model

# Initiating the tuner
tuner = BayesianOptimization(
    createModel_DY,
    objective='val_mean_squared_error',
    max_trials=10,  
    executions_per_trial=1,  
    directory='my_dir',
    project_name='keras_tuner_demo'
)

concatenated_inputs = np.column_stack((x1, x2))

# Splitting data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(concatenated_inputs, y, test_size=0.2, random_state=42)

# Start the search for the best hyperparameter configuration
tuner.search(x_train, y_train, epochs=20, validation_data=(x_val, y_val))

# After the search, retrieve the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Training the Best Model Found by Keras Tuner
model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val))

# Optional: plotting the loss curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()




# # Import necessary libraries
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow import keras
# from keras import layers
# from kerastuner.tuners import BayesianOptimization
# from sklearn.model_selection import train_test_split

# # Step 1: Generating Synthetic Data
# # This generates data based on a polynomial function with some noise
# np.random.seed(42)  # Seed for reproducibility
# x1 = np.linspace(-1, 1, 400)
# x2 = np.linspace(-1, 1, 400)
# y = 0.5 * x1**3 - 0.2 * x1**2 + 0.1 * x1 - 0.2+ 0.7 * x2**3 - 0.4 * x2**2 + 0.05 * x2 - 0.2 + np.random.normal(scale=0.05, size=x1.shape)

# # Step 2: Building the Model and Optimizing it with Keras Tuner
# def build_model(name, hp):
#     model = keras.Sequential()
#     inp = tf.keras.Input(shape=(1,))
#     model.add(inp)  # Input layer
    
#     # Tuning the number of layers and their configurations
#     # Keras Tuner will choose an optimal number of layers, units in each layer, and their activation functions
#     for i in range(hp.Int('num_layers', 1, 5)):
#         model.add(layers.Dense(
#             units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
#             activation=hp.Choice('activation_' + str(i), ['relu', 'tanh', 'sigmoid'])
#         ))

#     #model.add(layers.Dense(1))  # Output layer
#     nnout = tf.keras.layers.Dense(1)(model)

#     mod = tf.keras.Model(inp, nnout, name=name)
#     return mod


# def createModel_DY(hp):
#     x1 = tf.keras.Input(shape=(1), name='x1')
#     x2 = tf.keras.Input(shape=(1), name='x2')

#     modnnu = build_model('nnu',hp)
#     modnnubar = build_model('nnubar',hp)
   
#     nnu_pdf_input = tf.keras.layers.Concatenate()([x1])
#     nnubar_pdf_input = tf.keras.layers.Concatenate()([x2])

#     modnnu_pdf_eval = modnnu(nnu_pdf_input)
#     modnnubar_pdf_eval = modnnubar(nnubar_pdf_input)

#     product = tf.multiply(modnnu_pdf_eval, modnnubar_pdf_eval)

#     # Compilation of the model with a tunable learning rate
#     model.compile(
#         optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
#         loss='mean_squared_error',
#         metrics=['mean_squared_error']
#     )

#     # return tf.keras.Model([x1, x2], [product])
#     return model

# # Initiating the tuner
# # Here, BayesianOptimization is used to find the best hyperparameter values by building and evaluating different models
# tuner = BayesianOptimization(
#     createModel_DY,
#     objective='val_mean_squared_error',
#     max_trials=10,  # The total number of trials (model configurations) to test
#     executions_per_trial=1,  # The number of models that should be built and fit for each trial
#     directory='my_dir',
#     project_name='keras_tuner_demo'
# )

# concatenated_inputs = np.column_stack((x1,x2))

# # Splitting data into training and validation sets
# x_train, x_val, y_train, y_val = train_test_split(concatenated_inputs, y, test_size=0.2, random_state=42)

# # Start the search for the best hyperparameter configuration
# # The tuner explores different configurations, evaluating them on the validation set
# tuner.search(x_train, y_train, epochs=20, validation_data=(x_val, y_val))

# # After the search, retrieve the best hyperparameters
# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# # Step 3: Training the Best Model Found by Keras Tuner
# # The model is built using the best hyperparameters and then trained on the full dataset
# model = tuner.hypermodel.build(best_hps)
# history = model.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val))

# # Optional: plotting the loss curve
# plt.figure(figsize=(10, 6))
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss Over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()


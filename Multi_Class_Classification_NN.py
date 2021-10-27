"""
Created on Tue Oct 26 20:23:23 2021

@author: Danial Arab
"""
# In this project the datasets from the 4th week of Machine Learning class, offered by Prof. Andrew Ng from Stanford University, was used

# to build a neural network to recognize the handwritten digits.

# Step_1: Importing the required libraries 

import keras
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation 
from tensorflow.keras.optimizers import SGD
from scipy.io import loadmat
from sklearn.model_selection import train_test_split 
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# Step_2: Data importing and handling: importing the dataset and making them ready for the training 

data = loadmat ('ex3data1.mat')

X = data['X']
y = data['y']

y = to_categorical(y)

for i in range(len(y)):
    if y[i, 10]==1.:
        y[i, 10]=0;
        y[i,0] = 1.; 

y = y[:, 0:10]

# Step_3: Splitting the data into training and testing datasets 

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.5, random_state=10)

# Step_4: Building the model 

model = Sequential()

model.add (Dense (25, activation = 'sigmoid', input_dim = 400))
model.add(Dropout(0.5))

model.add (Dense (25, activation = 'sigmoid'))
model.add(Dropout(0.5))

model.add (Dense (10, activation = 'softmax'))
sgd = SGD (lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov=True)


model.compile (loss = 'categorical_crossentropy', 
               optimizer = sgd, 
               metrics = ['accuracy'])
print(model.summary()) 

# Step_5: Training the model 

model.fit (X_train, y_train, 
           epochs = 1000,
           batch_size = 128)

score = model.evaluate (X_test, y_test, batch_size = 128)

model.metrics_names

score

# Step_6: Making a prediction via the trained model using the testing datasets 

pred = model.predict(X_test)


plt.figure(figsize=(40, 4))
for i in range(10, 20):
    # display original
    ax = plt.subplot(3, 20, i + 1)
    X = np.reshape (X_test[i], (20, 20), order='F')
    plt.imshow(X, cmap="binary")
    index = np.where (pred[i] == max(pred[i]))
    print(index[0])
plt.show()

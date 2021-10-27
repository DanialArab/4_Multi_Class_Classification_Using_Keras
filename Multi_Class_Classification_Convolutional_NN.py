"""
Created on Sun Oct 10 08:57:56 2021

@author: Danial Arab
"""
# In this project the datasets from the 4th week of Machine Learning class, offered by Prof. Andrew Ng from Stanford University, was used

# to build a convolutional neural network to recognize the handwritten digits. 

# Step_1: Importing required libraries 

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from sklearn.model_selection import train_test_split 


# Step_2: Data importing and handling: importing the dataset and making them ready for the training 

data = loadmat('ex3data1.mat')
X = data['X']
y = data['y']

X_img_list = []
for i in range(len(X)):
    X_img = np.reshape(X[i, :], (20,20, 1), order='F')
    X_img_list.append(X_img)
    
X = np.array(X_img_list)

y = to_categorical(y)

for i in range(len(y)):
    if y[i, 10]==1.:
        y[i, 10]=0;
        y[i,0] = 1.; 

y = y[:, 0:10]
    
# Step_3: Displaying some images  
  
plt.imshow(X_img_list[1500], cmap = 'gray')

# Step_4: Splitting the data into training and testing datasets 

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.5, random_state = 10)


# Step_5: Building the model 

activation = 'sigmoid'
model = Sequential()
model.add(Conv2D(16, 3, activation = activation, padding = 'same', input_shape = (20, 20, 1)))
model.add(BatchNormalization())

model.add(Conv2D(16, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(32, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
model.add(BatchNormalization())

model.add(Conv2D(32, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
model.add(BatchNormalization()) 
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(64, activation = activation, kernel_initializer = 'he_uniform'))
model.add(Dense(10, activation = 'softmax'))

model.compile(optimizer = 'rmsprop',loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(model.summary()) 

# Step_6: Training the model 

history = model.fit(X_train, y_train, epochs = 10, validation_data = (X_test, y_test)) 

# Step_7: Plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Step_8: Making a prediction via the trained model using the testing datasets 

pred = model.predict(X_test)


plt.figure(figsize=(40, 4))
for i in range(10):
    # display original
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(X_test[i].reshape(20, 20, 1), cmap="binary")
    index = np.where (pred[i] == max(pred[i]))
    print(index[0])
plt.show()

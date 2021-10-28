"""
Created on Wed Oct 27 10:48:10 2021

@author: Danial Arab
"""
# Step_1: Importing the required libraries 

from scipy.io import loadmat
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.metrics import confusion_matrix

# Step_2: Data importing and handling: importing the dataset and making them ready for the training 
data = loadmat ('ex3data1.mat')

X = data['X']
y = data['y']

for i in range(len(y)):
    if y[i] == 10:
        y[i] = 0;
 
# Step_3: Splitting the data into training and testing datasets 

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.5, random_state=10)

# Step_4: Building the model 

model = LogisticRegression (solver = 'liblinear', multi_class='ovr',)

# Step_5: Training the model 

model.fit(X_train, y_train)

# Step_6: Making a prediction via the trained model using the testing datasets 

pred = model.predict (X_test)
pred = pred.reshape(2500, 1)

score = model.score (X_test, y_test)
print(score)


plt.figure(figsize=(40, 4))
for i in range(50,60):
    # display original
    ax = plt.subplot(3, 20, i + 1)
    img = np.reshape (X_test[i], (20, 20), order='F')
    plt.imshow(img, cmap="binary")
    print(pred[i])
plt.show()

# Step_7: Visualizing the model predictions and model accuracy through Confusion Matrix
conf_matrix = confusion_matrix (y_test, pred)
conf_matrix

plt.figure(figsize = (10,10))
plt.imshow(conf_matrix, cmap = 'Pastel1')
plt.title('Confusion Matrix')

plt.xticks (np.arange(10))
plt.yticks(np.arange(10))

plt.ylabel('Ground truth labels')
plt.xlabel('Predicted labels')

plt.colorbar()
width, height = conf_matrix.shape
for x in range(width):
    for y in range(height):
        plt.annotate ( str(conf_matrix [x][y]), xy = (y, x), 
                      horizontalalignment = 'center', verticalalignment = 'center')
        
        
        
        

# Multi Class Classification Using Keras 

In this project the datasets from the 4th week of Machine Learning class, offered by Prof. Andrew Ng from Stanford University, was used to build a convolutional neural network to recognize the handwritten digits. The dataset is a subset of the MNIST handwritten digit dataset (http://yann.lecun.com/exdb/mnist/).

Although Octave was suggested for this class assignment, here the problem was solved through Python using Keras library. Thuis project can be done through the following approaches:

# Table of content

1. [Convolutional Neural Network -- working with the images](#1)
2. [Neural Network -- working with the unrolled data of images](#2)
3. [Logistic Regression (one-vs-rest classification)-- working with the unrolled data of images](#3)

<a name="1"></a>
# Convolutional Neural Network working with the images

In this case, the convolutional neural network is applied to build the model. The training and validation accuracy is shown in Fig. 1. 

![plot](https://user-images.githubusercontent.com/54812742/138943711-8da751c1-93bf-4402-a824-7a8c0c36aefd.png)

Fig. 1: Training and validation accuracies vs. number of epochs

One slice from the testing dataset, never seen by the model, was shown in Fig. 2.

![testing data](https://user-images.githubusercontent.com/54812742/138944068-815ec3a3-0071-4246-9a02-74e97bec6f3a.png)

Fig. 2: 10 datapoints from the testing dataset

The model predictions for the above handwritten digits are shown in Fig. 3.

![result](https://user-images.githubusercontent.com/54812742/138944219-7e066452-e51f-453e-87d9-bf75c770b6d8.PNG)

Fig. 3: Convolutional Neural Network Model predictions  

<a name="2"></a>
# Neural Network -- working with the unrolled data of images

In this approach, the images of 20 by 20 by 1 are unrolled as 400-dimensional vectors. So, the input layer is of the size 400 units. In this case model is built of different layers of neural networks. The hidden layer, the second layer, has 25 units and there would be 10 output units (corresponding to the 10 digit classes). The model architecture is shown in Fig. 4. 

![1](https://user-images.githubusercontent.com/54812742/139000637-32e3d397-824d-42dd-8fcf-8e310510da11.PNG)

Fig. 4: Neural Network architecture [1]

One slice from the testing dataset, never seen by the model, was shown in Fig. 5. 

![2](https://user-images.githubusercontent.com/54812742/139000905-e89c2aca-0a9c-4710-9d5b-2b842897df80.png)

Fig. 5: 10 datapoints from the testing dataset

The neural network model predictions for the above handwritten digits are shown in Fig. 6.

![3](https://user-images.githubusercontent.com/54812742/139001070-4ac429d9-8e36-4e26-b80c-82d6ad0eed2e.PNG)

Fig. 6: Neural Network model predictions  

In this case, 1000 epochs leads to an accuracy of 0.87. 

<a name="3"></a>
# Logistic Regression (one-vs-rest classification)-- working with the unrolled data of images

Logistic regression model is available in sklearn library (sklearn.linear_model) and can fit binary, One-vs-Rest, or multinomial logistic regression with optional L_1, L_2 or Elastic-Net regularization [3]. The solvers of the logistic regression model are ???liblinear???, ???newton-cg???, ???lbfgs???, ???sag??? and ???saga???, which are compared in Fig. 7. 

![2](https://user-images.githubusercontent.com/54812742/139168098-622ea424-9e30-4172-a3fc-b11944ac49d7.PNG)

Fig. 7:Solvers of logistic regression model [3]

The compatibility of each solver and the regularization is summarized in Fig. 7 (e.g., Elastic-Net regularization, where both L1 and L2 penalty terms are added, is only supported by the ???saga??? solver [3]. 
In this project, because we have 10 different classes, corresponding to 10 digits (0 to 9) the problem is multi class classification. To do this multi class classification, one-vs-rest classification is applied meaning a binary problem is fit for each label. This is achieved through setting the ???multi_class??? parameter of the Logistic regression model to ???ovr???. The confusion matrix is shown in Fig. 8. As shown in Fig. 8, the model predicts the labels very well: for example, digit 0 was labeled correctly for 255 times, while digit 7 is mislabeled 13 times as 9. 

![1](https://user-images.githubusercontent.com/54812742/139168692-64a03f8b-9e26-4c69-bc17-e876cb1b06c5.png)

Fig. 8: Confusion Matrix

The model score is obtained as 0.9 meaning the prediction is correct 90%. 

References:

[1] Machine Learning Course offered by Coursera, https://www.coursera.org/learn/machine-learning

[2] Sreenivas Bhattiprolu's youtube channel (143 - Multiclass classification using Keras), https://www.youtube.com/watch?v=obOjpVdO3gY

[3] https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression 

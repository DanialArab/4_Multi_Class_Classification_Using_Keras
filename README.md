# Multi Class Classification Using Keras 

In this project the datasets from the 4th week of Machine Learning class, offered by Prof. Andrew Ng from Stanford University, was used to build a convolutional neural network to recognize the handwritten digits. The dataset is a subset of the MNIST handwritten digit dataset (http://yann.lecun.com/exdb/mnist/).

Although Octave was suggested for this class assignment, here the problem was solved through Python using Keras library. Thuis project can be done through the following approaches:

# Table of content

1. [Convolutional Neural Network -- working with the images](#1)
2. [Neural Network -- working with the unrolled data of images](#2)

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

Fig. 3: Model predictions  

<a name="2"></a>
# Neural Network -- working with the unrolled data of images




References:

[1] Machine Learning Course offered by Coursera, https://www.coursera.org/learn/machine-learning

[2] Sreenivas Bhattiprolu's youtube channel (143 - Multiclass classification using Keras), https://www.youtube.com/watch?v=obOjpVdO3gY


# Multi_Class_Classification_Using_Keras

In this project the datasets from the 4th week of Machine Learning class, offered by Prof. Andrew Ng from Stanford University, was used to build a convolutional neural network to recognize the handwritten digits. The dataset is a subset of the MNIST handwritten digit dataset (http://yann.lecun.com/exdb/mnist/).

Although Octave was suggested for this class assignment, here the problem was solved through Python using Keras library. The training and validation accuracy was shown in Fig. 1. 

![plot](https://user-images.githubusercontent.com/54812742/138943711-8da751c1-93bf-4402-a824-7a8c0c36aefd.png)

Fig. 1: Training and validation accuracies vs. number of epochs

One slice from the testing dataset, never seen by the model, was shown in Fig. 2.

![testing data](https://user-images.githubusercontent.com/54812742/138944068-815ec3a3-0071-4246-9a02-74e97bec6f3a.png)

Fig. 2: 10 datapoints from the testing dataset

The model prediction for the above handwritten digits are shoen in Fig. 3.

![result](https://user-images.githubusercontent.com/54812742/138944219-7e066452-e51f-453e-87d9-bf75c770b6d8.PNG)

Fig. 3: Model results 

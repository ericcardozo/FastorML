# FastorML
This repository contains a work-in-progress deep learning library implemented in C++ using the Fastor Fastor Tensor Library. The library is designed for learning purposes and provides a set of tools and algorithms for building and training deep neural networks.

This library has been designed taking into account the principles of modern C++ programming, incorporating features like move semantics, smart pointers, and templates to maximize performance, efficiency and safety in the codebase.

The test.cpp file have an already working implementation of a simple neural net trained with the SGD algorithm. To use this files you should download the MNIST dataset from kaggle in the data folder.

https://www.kaggle.com/competitions/digit-recognizer/data

To do list:

* Add SoftMax, Sigmoid, Tanh, ... Activation functions. 
* Add CrossEntropyLoss, MSELoss, ... Loss functions.
* Add momentum to SGD optimizer.
* Add Adam optimizer.
* Optimize the shuffle method in the batches of the dataset class.
* Optimize code making inplace versions of activation functions.
* Creating a base class for layers. 
* Creating a base class for Neural Network.

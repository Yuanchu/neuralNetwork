# neuralNetwork

This is an attempt to implement a neural network model with Julia. Traditional approach of updating model parameters in neural networks consists mainly of stochastic gradient descent (SGD). However, SGD is a first order method with slow convergence rate and suffers from the problem of vanishing gradient. Moreover, it is hard to parallelize calculations using SGD. Here, we treat the training process as an optimization problem and propose a method that resembles the alternating direction method of multiplierstreats (ADMM). Variables are set up in a way that allows for good parallelization of heavy computations.

Right now, we have only coded a functional version for feed-forward neural networks. Below are some sample images when we apply our neural nets to the encoder-decoder problem using the MNIST dataset. 

![alt tag](https://github.com/Yuanchu/neuralNetwork/blob/master/images/87_orig.png) | ![alt tag](https://github.com/Yuanchu/neuralNetwork/blob/master/images/87_comp.png)

![alt tag](https://github.com/Yuanchu/neuralNetwork/blob/master/images/167_orig.png) | ![alt tag](https://github.com/Yuanchu/neuralNetwork/blob/master/images/167_comp.png)

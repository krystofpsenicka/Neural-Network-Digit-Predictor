# Hand-drawn digit predictor

## Function

The hand-drawn digit predictor lets a user draw a digit on a canvas and after the user submits the drawing it uses a pretrained neural network to determine which digit did the user draw.

## Detailed Assignment

The assignment was to build a neural network from scratch without using any high-level machine-learning libraries such as pytorch or
tensorflow, to train the model on the MNIST (digit) dataset and to build an interactive dialog window with a canvas on which a user can draw a digit and get a prediction of the drawn digit.

## Algorithm

The algorithm used for the neural network is gradient descent, an iterative algorithm for training neural networks. -> Forward propagation, error calculation and backpropagation for each sample from the dataset. The neural network used has 1 input layer with 784 nodes (1 for each pixel of 28x28 image), 1 hidden layer and 1 output layer each with 10 nodes.

## Program

### neuralNetwork.py

This file contains the code for training the model. It first imports the images and labels from the MNIST training dataset using the getMNIST("train") function imported from data.py which loads the dataset from an npz file, parses the data, standardizes the pixel values in the images and reshapes the images and labels. Then it initializes the weights as matrices with random values from -0.5 to 0.5 and biases as zero matrices. It then trains the NN (neural network). It loops through all the images applying forward propagation then calculating the error and doing backward propagation (adjusting the weights and biases based on the error) for each image. This is done multiple times (based on the number of epochs). It also calculates the accuracy of each epoch and prints it to the console when the epoch is done. After all the epoch are finished the weights and biases are saved as an npz file.

### showData.py

This file is for visualizing a set of n random images from the training dataset using matplotlib.

### testModel.py

This file is used to test the accuracy of the pretrained model. It is similar to neuralNetwork.py. It first imports the images and labels from the MNIST testing dataset using the getMNIST("test") function imported from data.py. Then it performs forward propagation on each image recording if the prediction was correct and then it prints the accuracy as a percentage in the terminal.

### predict.py

A simple module with one function used for digit prediction in main.py. The function takes the image to be analyzed, weights and biases from a pretrained model and it performs forward propagation. And it returns the prediction.

### main.py

This is the file which handles the user interaction with tkinter. It contains the class DrawingInput which consists of multiple functions:

- init: initializes all tkinter elements (root, window, canvas, buttons...), the image array which is changed as the user draws on the tkinter canvas and imgData which is used for the prediction (it is the image array only formatted after the user submits it for prediction)
- drawOval: function for drawing in the canvas and simultaneously into the image array
- erase: callback for the erase button, it erases the whole canvas and the image array
- predict: callback for the submit button, if the user has drawn something it generates a prediction using the function from predict.py and prints it in the tkinter window, otherwise it just prints a message prompting the user to draw something
- run: function for running the tkinter mainloop, it opens the interactive tkinter window for the user

### data.py

Contains all the functions pertaining to data:

- standardize(dataset): standardizes all images in dataset (increases contrast and simplifies shapes)
- getMNIST(subset): loads the specified subset of the MNIST dataset from npz file
- getWeightsBiases: loads pretrained weights and biases from npz file

## Representation of input and output data

The neural network takes 28x28 grayscale images as 1x784 matrices with standardized values from 0 to 1. The output of the neural network are the adjusted weights and biases as a npz file.
The main program takes the user drawing as input, generates a prediction after the submit button is pressed and prints the prediction onto the tkinter user window as output.

## Process

The neural network took the most amount of time. I wasn't happy with the accuracy so I kept micro adjusting the hyperparameters and applied global standardization on the images. The dataset is great but it is not perfect when the input is a digit drawn with a computer mouse.

## Unfinished work

The program does what it is supposed to do according to the assignment. But it could be more accurate. A convolutional layer might be added for better accuracy. Centering and rotation of the input image could make the predictions better.

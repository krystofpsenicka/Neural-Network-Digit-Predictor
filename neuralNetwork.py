from data import getMNIST
import numpy as np
from scipy.special import expit


# Get training data
images, labels = getMNIST("train")

# Initialize weights with random values and biases as 0
weights1 = np.random.uniform(-0.5, 0.5, (20, 784)) # Weights from input layer to hidden layer
weights2 = np.random.uniform(-0.5, 0.5, (10, 20)) # Weights from hidden layer to output layer
bias1 = np.zeros((20, 1)) # Bias from i to h
bias2 = np.zeros((10, 1)) # Bias from h to o

nrCorrect = 0 # How many predictions did the model get right in each epoch
# Hyperparameters
learnRate = 0.04 # The rate by which the model modifies the weights and biases during back propagation
epochs = 5 # How many times to train the NN on the whole training dataset
# Loop for training the neural network
for epoch in range(epochs):
    for img, label in zip(images, labels): # Loop through all images with their corresponding labels
        # Change vectors to matrices for matrix multiplication
        img.shape += (1,)
        label.shape += (1,)
        # Forward propagation input -> hidden
        hidden0 = bias1 + weights1 @ img
        # Sigmoid activation function to normalize values
        hidden = expit(hidden0)
        # Forward propagation hidden -> output
        output0 = bias2 + weights2 @ hidden
        # Sigmoid activation function
        output = expit(output0)

        # Error calculation: adds 1 to nrCorrect if the network classified the image correctly
        if np.argmax(output) == np.argmax(label):
            nrCorrect += 1

        # Backpropagation output -> hidden
        deltaOutput = output - label # Difference of prediction from label
        weights2 += -learnRate * deltaOutput @ np.transpose(hidden)
        bias2 += -learnRate * deltaOutput * 1 # 1 = output value of "bias neuron" 
        # Backpropagation hidden -> input (activation function derivative: sigmoid * (1 - sigmoid) -> hidden * (1 - hidden))
        deltaHidden = np.transpose(weights2) @ deltaOutput * (hidden * (1 - hidden)) # How much each hidden neuron participated in output error
        weights1 += -learnRate * deltaHidden @ np.transpose(img)
        bias1 += -learnRate * deltaHidden * 1 # 1 = output value of "bias neuron"

    # Print accuracy for epoch
    print(f"Accuracy: {round((nrCorrect / images.shape[0]) * 100, 2)}%")
    nrCorrect = 0

# Save weights and biases into npz file
np.savez("./data/weightsBiases.npz", weights1=weights1, weights2=weights2, bias1=bias1, bias2=bias2)

from data import getMNIST, getWeightsBiases
import matplotlib.pyplot as plt
import numpy as np
from predict import predict

# Get weights and biases from pretrained model
weights1, weights2, bias1, bias2 = getWeightsBiases()

# Get images and labels from test subset
images, labels = getMNIST("test")

def getAccuracy():
    nrCorrect = 0 # How many predictions did the model get right

    for img, l in zip(images, labels): # Loop through all images with their corresponding labels
        # Change input image to matrix
        img.shape += (1,)

        ## Generate prediction from image
        output = predict(img, weights1, weights2, bias1, bias2)

        if output.argmax() == l.argmax(): # Increase counter if the prediction comes out right
            nrCorrect += 1

    return round((nrCorrect / images.shape[0]) * 100, 2)

print(f"The accuracy of the model is: {getAccuracy()}%")

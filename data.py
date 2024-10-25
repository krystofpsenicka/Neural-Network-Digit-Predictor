import numpy as np
import pathlib

def standardize(dataset): # Positive global standardization of each pixel value in all dataset images
    # Calculate mean and standard deviation of pixel values from the whole dataset
    meanPx = dataset.mean()
    stdPx = dataset.std()

    dataset = (dataset - meanPx)/(stdPx) # Standardize pixel values
    dataset = dataset - (dataset.min() - (-1)) # Shift values so that min is -1
    dataset = np.clip(dataset, -1.0, 1.0) # Clip values to [-1, 1]
    dataset = (dataset + 1.0) / 2.0 # Shift values from [-1,1] to [0,1]

    return dataset


def getMNIST(subset): # Load and parse mnist training/testing data from npz file
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz") as f:
        images, labels = f[f"x_{subset}"], f[f"y_{subset}"] # Subset is "train" or "test"
    # images /= 255 # Normalize pixel values
    images = standardize(images) # Standardize images
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2])) # Reshape images from 3d arrays to 2d arrays
    labels = np.eye(10)[labels] # Change labels to vectors with 1 at the index of the label value and 0s everywhere else
    return images, labels

def getWeightsBiases(): # Load and parse weights and biases from npz file
    with np.load("./data/weightsBiases.npz") as f: 
        weights1, weights2, bias1, bias2 = f['weights1'], f['weights2'], f['bias1'], f['bias2']
    # Reshape weights and biases into matrices
    weights1.shape = (20, 784)
    weights2.shape = (10, 20)
    bias1.shape = (20, 1)
    bias2.shape = (10, 1)
    return weights1, weights2, bias1, bias2
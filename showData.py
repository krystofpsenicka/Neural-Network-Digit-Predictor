from data import getMNIST
import matplotlib.pyplot as plt
import numpy as np

# Get training data
images, labels = getMNIST("train")

def showDataAsImages(n):
    for j in range(n):
        pos = np.random.randint(low=0, high=50_000) # Get random number (0->50_000) as starting image for plot
        for i in range(9):
            # Add image to matplotlib plot
            plt.subplot(330 + 1 + i)
            plt.imshow(images[i+pos].reshape(28, 28), cmap="Greys")
        plt.show() 

showDataAsImages(4)
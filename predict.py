from scipy.special import expit

def predict(img, weights1, weights2, bias1, bias2): # function for forward propagation (prediction)
    # Forward propagation input -> hidden
    hidden0 = bias1 + weights1 @ img
    # Sigmoid activation function to normalize values
    hidden = expit(hidden0)
    # Forward propagation hidden -> output
    output0 = bias2 + weights2 @ hidden
    # Sigmoid activation function
    output = expit(output0)
    return output
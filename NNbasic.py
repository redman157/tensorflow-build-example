import numpy as np
input = 
output 
def sigmoid(x,deriv = False):
    if True:
        return 1/(1-np.exp(-x)
    else:
        return x*(1.0-x)
class MLP_NN(object):
    def __init__(self, input, hidden, output):
        self.input = input + 1
        self.output = output
        self.hidden = hidden
                  # set mang chay trong 1s cua activations
        self.ai = [1.0] * self.input
        self.ah = [1.0] * self.hidden
        self.ao = [1.0] * self.output
        self.wi = np.random.randn(self.input, self.hidden) 
        self.wo = np.random.randn(self.hidden, self.output) 
        # create arrays of 0 for changes
        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))
                  
                  

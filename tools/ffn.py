from numpy import *

class FeedForwardNetwork:
    def __init__(self, nIn, nHidden, nOut):
        # learning rate
        self.alpha = 0.1
                                                
        # number of neurons in each layer
        self.nIn = nIn
        self.nHidden = nHidden
        self.nOut = nOut
        
        # initialize weights randomly (+1 for bias)
        self.hWeights = 0.1*random.random((self.nHidden, self.nIn+1)) 
        self.oWeights = 0.1*random.random((self.nOut, self.nHidden+1))
        
        # activations of neurons (sum of inputs)
        self.hActivation = zeros((self.nHidden, 1), dtype=float)
        self.oActivation = zeros((self.nOut, 1), dtype=float)
        
        # outputs of neurons (after sigmoid function)
        self.iOutput = zeros((self.nIn+1, 1), dtype=float)      # +1 for bias
        self.hOutput = zeros((self.nHidden+1, 1), dtype=float)  # +1 for bias
        self.oOutput = zeros((self.nOut), dtype=float)
        
        # deltas for hidden and output layer
        self.hDelta = zeros((self.nHidden), dtype=float)
        self.oDelta = zeros((self.nOut), dtype=float)   
    
    def forward(self, input):
        # set input as output of first layer (bias neuron = 1.0)
        self.iOutput[:-1, 0] = input
        self.iOutput[-1:, 0] = 1.0
        
        # hidden layer
        self.hActivation = dot(self.hWeights, self.iOutput)
        self.hOutput[:-1, :] = tanh(self.hActivation)
        
        # set bias neuron in hidden layer to 1.0
        self.hOutput[-1:, :] = 1.0
        
        # output layer
        self.oActivation = dot(self.oWeights, self.hOutput)
        self.oOutput = tanh(self.oActivation)
    
    def backward(self, teach):
        error = self.oOutput - array(teach, dtype=float)
        
        # deltas of output neurons
        self.oDelta = (1 - tanh(self.oActivation)) * tanh(self.oActivation) * error
                
        # deltas of hidden neurons
        self.hDelta = (1 - tanh(self.hActivation)) * tanh(self.hActivation) * dot(self.oWeights[:,:-1].transpose(), self.oDelta)
                
        # apply weight changes
        self.hWeights = self.hWeights - self.alpha * dot(self.hDelta, self.iOutput.transpose()) 
        self.oWeights = self.oWeights - self.alpha * dot(self.oDelta, self.hOutput.transpose())
        
        return error.item()**2
    
    def getOutput(self):
        return self.oOutput

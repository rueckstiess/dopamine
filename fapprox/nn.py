from dopamine.fapprox.fa import FA
import numpy as np


class NN(FA):
    
    def __init__(self, indim, outdim, nhidden=20):
        FA.__init__(self, indim, outdim)
        
        # learning rate
        self.alpha = 0.1
                                                
        # number of neurons in each layer
        self.indim = indim
        self.nhidden = nhidden
        self.outdim = outdim
        
        # change output activation if task is classification
        self.classification = False
        
        # online training or batch, if batch, then train for that many epochs 
        self.online = False
        self.epochs = 100
        
        # initialize weights randomly (+1 for bias)
        self.hWeights = 0.01 * np.random.random((self.nhidden, self.indim+1)) 
        self.oWeights = 0.01 * np.random.random((self.outdim, self.nhidden+1))
        
        # activations of neurons (sum of inputs)
        self.hActivation = np.zeros((self.nhidden, 1), dtype=float)
        self.oActivation = np.zeros((self.outdim, 1), dtype=float)
        
        # outputs of neurons (after sigmoid function)
        self.iOutput = np.zeros((self.indim+1, 1), dtype=float)      # +1 for bias
        self.hOutput = np.zeros((self.nhidden+1, 1), dtype=float)    # +1 for bias
        self.oOutput = np.zeros((self.outdim), dtype=float)
        
        # deltas for hidden and output layer
        self.hDelta = np.zeros((self.nhidden), dtype=float)
        self.oDelta = np.zeros((self.outdim), dtype=float)   
    
    
    def logistic(self, data):
        return (1. + np.tanh(data/2.)) / 2.
        
    def logistic_prime(self, data):
        return self.logistic(data) * (1. - self.logistic(data))
    
    def softmax(self, x):
        ret = np.exp(x)
        ret /= sum(ret)
        return ret
        
    def predict(self, inp):
        # set input as output of first layer (bias neuron = 1.0)
        self.iOutput[:-1, 0] = np.asarray(inp).flatten()
        self.iOutput[-1:, 0] = 1.0
        
        # hidden layer
        self.hActivation = np.dot(self.hWeights, self.iOutput)
        self.hOutput[:-1, :] = self.logistic(self.hActivation)
        
        # set bias neuron in hidden layer to 1.0
        self.hOutput[-1:, :] = 1.0
        
        # output layer
        self.oActivation = np.dot(self.oWeights, self.hOutput)
        if self.classification:
            self.oOutput = self.softmax(self.oActivation)
        else:
            self.oOutput = self.logistic(self.oActivation)
        return self.oOutput
    
    
    def update(self, inp, tgt):
        if self.online:
            # train only one step online on current sample
            self.predict(inp)
            self.backward(tgt)
        else:
            # don't train, only update dataset
            FA.update(self, inp, tgt)
    
    
    def train(self):
        # only train if online learning is disabled
        if not self.online:
            for e in xrange(self.epochs):
                for (inp, tgt) in self.dataset.randomizedSamples():
                    self.predict(inp)
                    self.backward(tgt)
        
    
    def backward(self, teach):
        error = self.oOutput - np.array(teach, dtype=float) 

        # deltas of output neurons
        if self.classification:
            self.oDelta = error
        else:
            self.oDelta = self.logistic_prime(self.oActivation) * error
                
        # deltas of hidden neurons
        self.hDelta = self.logistic_prime(self.hActivation) * np.dot(self.oWeights[:,:-1].T, self.oDelta)
                
        # apply weight changes
        self.hWeights = self.hWeights - self.alpha * np.dot(self.hDelta, self.iOutput.T) 
        self.oWeights = self.oWeights - self.alpha * np.dot(self.oDelta, self.hOutput.T)
        return error
    
    def getOutput(self):
        return self.oOutput

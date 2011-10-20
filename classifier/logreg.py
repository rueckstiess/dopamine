import numpy as np
from dopamine.classifier.classifier import Classifier


class LogisticRegression(Classifier):
    
    alpha = 0.01
    
    
    def classify(self, inp):
        """ predict the output for the given input. """
        # attach a 1 for the bias
        inp = np.asarray(inp)
        inp = np.r_[inp, np.array([1])]
        
        # calculate outputs
        a = np.exp([np.dot(w, inp) for w in self.weights.T])
        
        # normalize for probabilities
        z = sum(a)
        probs = np.array([a[k] / z for k in range(len(a))])
        
        # store the belief for later use
        self.belief = probs

        return np.argmax(probs)
        
        
    def update(self, inp, tgt):
        """ update the function approximator to return something closer
            to target when queried for input next time. Some function
            approximators only collect the input/target tuples here and
            learn only on a call to learn(). 
            tgt is an integer for the class number. conversion to one-of-k
            coding is done internally.
        """
        # online method doesn't need to keep track of data points
        # Classifier.update(self, inp, tgt)
        
        self.classify(inp)
                
        # attach 1 for the bias
        inp = np.asarray(inp)
        inp = np.r_[inp, np.array([1])]
        
        # convert to 1-of-k
        tgt = self._asOneOfK(tgt)
    
        for j in range(self.weights.shape[0]):
            for k in range(self.weights.shape[1]):
                self.weights[j,k] -= self.alpha*(self.belief[k]-tgt[k])*inp[j]
                    
    
    def reset(self):
        """ this initializes the function approximator to an initial state,
            forgetting everything it has learned before. """
        Classifier.reset(self)
        
        # create belief vector
        self.belief = np.zeros(self.nclasses)
        
        # create weight matrix with indim+1 (for bias) rows and nclasses
        self.weights = np.random.random((self.indim+1, self.nclasses))
        
    def train(self):
        """ some function approximators learn offline after collecting all 
            samples. this function executes one such learning step. """
        pass
        
    def _getParameters(self):
        """ getter method for parameters. """
        return self.weights.flatten()

    def _setParameters(self, parameters):
        """ setter method for parameters. """
        self.weights = parameters.reshape(self.indim+1, self.nclasses)
from dopamine.fapprox.fa import FA
import numpy as np

class RBF(FA):

    parametric = True
    numCenters = 20
    beta = 8.
    
    def _basisfunc(self, c, d):
        return np.exp(-self.beta * np.linalg.norm(c-d)**2)

    def _designMatrix(self, X):
        """ create design matrix with basis functions.
            input n x indim.
            output n x self.centers
        """
        G = np.zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi,ci] = self._basisfunc(c, x)
        return G

    def predict(self, inp):
        G = self._designMatrix(np.atleast_2d(inp))
        Y = np.dot(G, self.W)
        return self._asFlatArray(Y)

    def reset(self):
        FA.reset(self)
        self.centers = [np.random.uniform(-1, 1, self.indim) for i in xrange(self.numCenters)]
        self.W = np.random.random((self.numCenters, self.outdim))
        
    def learn(self):
        if len(self.dataset) == 0:
            return
        
        X = self.dataset.inputs
        Y = self.dataset.targets
        
        # choose random center vectors from training set
        rnd_idx = np.random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i,:] for i in rnd_idx]

        # calculate activations of RBFs
        G = self._designMatrix(X)

        # calculate output weights (pseudoinverse)
        self.W = np.dot(np.linalg.pinv(G), Y)

    def _getParameters(self):
        """ getter method for parameters. """
        return self.W.flatten()

    def _setParameters(self, parameters):
        """ setter method for parameters. """
        self.W = parameters.reshape(self.numCenters, self.outdim)

    parameters = property(_getParameters, _setParameters)


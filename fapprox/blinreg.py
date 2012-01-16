from dopamine.fapprox.fa import FA
import numpy as np

class BLinReg(FA):

    parametric = True
    beta = 8.
    
    def __init__(self, indim, outdim, bayes=True, rbf=False):
        """ initialize function approximator with input and output dimension. """
        self.rbf = rbf
        self.bayes = bayes
        if self.rbf:
            self.numCenters = 20
        else:
            self.numCenters = indim
        FA.__init__(self, indim, outdim)
    
    
    def _basisfunc(self, c, d):
        return np.exp(-self.beta * np.linalg.norm(c-d)**2)

    def _designMatrix(self, X):
        """ create design matrix with basis functions.
            input n x indim.
            output n x self.centers
        """
        if self.rbf:
            G = np.zeros((X.shape[0], self.numCenters), float)
            for ci, c in enumerate(self.centers):
                for xi, x in enumerate(X):
                    G[xi,ci] = self._basisfunc(c, x)
            return G
        else:
            return X

    def predict(self, inp):
        G = self._designMatrix(np.atleast_2d(inp))
        Y = np.dot(G, self.W)
        return self._asFlatArray(Y)

    def reset(self):
        FA.reset(self)
        self.centers = [np.random.uniform(-1, 1, self.indim) for i in xrange(self.numCenters)]
        self.W = np.random.random((self.numCenters, self.outdim))
        
        # parameters for maximum map
        self.alpha = 100.
        self.SN = np.matrix(self.alpha*np.eye(self.numCenters))
        self.mN = np.matrix(np.zeros((self.numCenters, 1), float))
        
        
    def train(self):
        if self.bayes:
            pass
        else:
            self.train_ml()
            
    
    def update(self, inp, tgt):
        if self.bayes:
            self.add_sample_map(inp, tgt)
        else:
            self.dataset.append(inp, tgt)
            
        
    def train_ml(self):
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
        
    
    def train_map(self):
        if len(self.dataset) == 0:
            return
        
        X = self.dataset.inputs
        Y = self.dataset.targets
        
        # choose random center vectors from training set
        rnd_idx = np.random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i,:] for i in rnd_idx]

        # calculate activations of RBFs
        G = np.asmatrix(self._designMatrix(X))
        Y = np.asmatrix(Y)
        M = self.numCenters
        
        # create (reset) prior over weights w
        m0 = np.matrix(np.zeros((M, 1), float))
        S0 = np.matrix(self.alpha*np.eye(M))

        # calculate posterior (p. 153, eqns. 3.50, 3.51)
        self.SN = S0.I + self.beta*G.T*G
        self.mN = np.linalg.inv(self.SN) * (S0.I*m0 + self.beta*G.T*Y)

        self.W = np.asarray(self.mN)
        
        
    def add_sample_map(self, x, y):
        g = np.matrix(self._designMatrix(x))
        
        SN_inv = np.linalg.inv(self.SN)
        SN_new = np.linalg.inv(SN_inv + self.beta*g.T*g)
        
        self.mN = SN_new*(SN_inv*self.mN + self.beta*g.T*y)
        self.SN = SN_new
        
        self.W = np.asarray(self.mN)
    

    def dOutdTheta(self, inp, outp):
        """ return the derivative of the output with respect to the parameters
            for a given input and output. 
        """
        G = self._designMatrix(np.asarray(inp).reshape(1, self.indim))
        return np.dot(G.T, np.asarray(outp).reshape(1, self.outdim)).flatten()

    def _getParameters(self):
        """ getter method for parameters. """
        return self.W.flatten()

    def _setParameters(self, parameters):
        """ setter method for parameters. """
        self.W = parameters.reshape(self.numCenters, self.outdim)

    parameters = property(_getParameters, _setParameters)


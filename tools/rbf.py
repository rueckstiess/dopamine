from numpy.linalg import norm, inv, pinv
from numpy import *

class RBF:
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [random.uniform(-1, 1, indim) for i in xrange(numCenters)]
        self.beta = 8.
        self.W = random.random((self.numCenters, self.outdim))

        # parameters for maximum map
        self.alpha = 1.
        self.SN = matrix(self.alpha*eye(self.numCenters))
        self.mN = matrix(zeros((self.numCenters, 1), float))

    def _basisfunc(self, c, d):
        return exp(-self.beta * norm(c-d)**2)

    def _designMatrix(self, X):
        """ create design matrix with basis functions.
            input n x indim.
            output n x self.centers
        """
        G = zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi,ci] = self._basisfunc(c, x)
        return G

    def train_ml(self, X, Y):
        """ X: matrix of dimensions n x indim 
            y: column vector of dimension n x 1 """

        # choose random center vectors from training set
        rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i,:] for i in rnd_idx]

        # calculate activations of RBFs
        G = self._designMatrix(X)

        # calculate output weights (pseudoinverse)
        self.W = dot(pinv(G), Y)

    def train_map(self, X, Y):
        # choose random center vectors from training set
        rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i,:] for i in rnd_idx]

        # calculate activations of RBFs
        G = asmatrix(self._designMatrix(X))
        Y = asmatrix(Y)
        M = self.numCenters
        
        # create (reset) prior over weights w
        m0 = matrix(zeros((M, 1), float))
        S0 = matrix(self.alpha*eye(M))

        # calculate posterior (p. 153, eqns. 3.50, 3.51)
        self.SN = S0.I + self.beta*G.T*G
        self.mN = inv(self.SN) * (S0.I*m0 + self.beta*G.T*Y)

        self.W = asarray(self.mN)
    
    def _adaptCenters(self, x):
        cl = argmin([norm(x-c) for c in self.centers])
        self.centers[cl] = self.centers[cl] + 0.1*(x-self.centers[cl])
        
    def add_sample_map(self, x, y): 
        self._adaptCenters(x)
         
        # print "input", x
        # print "target", y
        g = self._designMatrix(x)
        # print g
        
        SN_inv = inv(self.SN)
        SN_new = inv(SN_inv + self.beta*g.T*g)
        
        self.mN = SN_new*(SN_inv*self.mN + self.beta*g.T*y)
        self.SN = SN_new
        
        self.W = asarray(self.mN)
        # print self.W.shape
        

    def test(self, X):
        """ X: matrix of dimensions n x indim """

        G = self._designMatrix(X)
        Y = dot(G, self.W)
        return Y


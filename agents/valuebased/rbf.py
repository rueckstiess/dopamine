from numpy import *
from numpy.linalg import norm, inv, pinv

from dopamine.agents.valuebased.estimator import Estimator

class RBFX:

    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [random.uniform(0, 1, indim) for i in xrange(numCenters)]
        self.beta = 10.
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


class RBFEstimator(Estimator):

    conditions = {'discreteStates':False, 'discreteActions':True}
    trainable = True
    
    def __init__(self, stateDim, actionNum):
        """ initialize with the state dimension and number of actions. """
        self.stateDim = stateDim
        self.actionNum = actionNum
        
        # define training and target array
        self.inputs = zeros((0,stateDim))
        self.actions = zeros((0, 1))
        self.targets = zeros((0, 1))
        
        # initialize all RBF models, one for each action
        self.models = [RBFX(stateDim, 20, 1) for i in range(actionNum)]

    def getBestAction(self, state):
        """ returns the action with maximal value in the given state. """
        state = state.flatten()
        action = array([argmax([self.getValue(state, array([a])) for a in range(self.actionNum)])])
        return action

    def getValue(self, state, action):
        """ returns the value of the given (state,action) pair. """
        state = state.flatten()
        action = action.flatten()
        return self.models[int(action.item())].test(state.reshape(1, self.stateDim)).item()

    def updateValue(self, state, action, value):
        self.inputs = r_[self.inputs, state.reshape(1, self.stateDim)]
        self.actions = r_[self.actions, action.reshape(1, 1)]
        self.targets = r_[self.targets, asarray(value).reshape(1, 1)]
   
    def _clear(self):
        """ clear collected training set. """
        self.inputs = zeros((0, self.stateDim))
        self.actions = zeros((0, 1))
        self.targets = zeros((0, 1))
                
    def _train(self):
        """ train individual models for each actions seperately. """
        # avoiding the value drift by substracting the minimum of the training set
        self.targets = (self.targets - min(self.targets))
        
        for a in range(self.actionNum):
            idx = where(self.actions[:,0] == a)[0]
            if len(idx) > 0 and idx.any():
                self.models[a].train_ml(self.inputs[idx,:], self.targets[idx,0])
     


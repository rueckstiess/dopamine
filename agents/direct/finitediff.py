from dopamine.agents.direct.direct import DirectAgent
from dopamine.agents.direct.controllers.linear import LinearController
from numpy import *
from numpy.linalg import pinv
from copy import copy

class FiniteDifferenceAgent(DirectAgent):
 
    def __init__(self, controllerClass=LinearController):
        self.deltas = None
        self.alpha = 0.01
        self.epsilon = 0.1
        self.decay = 0.99
        
        self.deltaList = []
        self._evaluation = False
        
        DirectAgent.__init__(self, controllerClass)
        
    def _setup(self, conditions):
        """ setup the agent and copy the original parameters. """
        DirectAgent._setup(self, conditions)
        self.storedParameters = copy(self.controller.parameters)
        self.deltas = random.uniform(-self.epsilon, self.epsilon, len(self.controller.parameters))

    def newEpisode(self):
        """ modify the original controller parameters and store the deltas """
        DirectAgent.newEpisode(self)
        
        if not self.evaluation:
            self.deltaList.append(self.deltas.copy())  
            self.deltas = random.uniform(-self.epsilon, self.epsilon, len(self.controller.parameters))
            self.controller.parameters = self.storedParameters + self.deltas
            self.epsilon *= self.decay
             
    def learn(self):
        self.controller.parameters = self.storedParameters
        
        # initialize matrix D and vector R
        D = ones((len(self.deltaList), len(self.controller.parameters)))
        R = zeros((len(self.deltaList), 1))
        
        # calculate the gradient with pseudo inverse
        for episode, deltas in enumerate(self.deltaList):
            D[episode, :] = deltas.reshape(1, len(deltas))
            R[episode, :] = sum(self.history[episode].rewards)
                             
        beta = dot(pinv(D), R)        
        gradient = ravel(beta)
        
        # update the parameters
        self.controller.parameters += self.alpha * gradient   
        self.storedParameters = copy(self.controller.parameters)

    
    def forget(self):
        DirectAgent.forget(self)
        self.deltaList = []
    
    def _getEvaluation(self):
        return self._evaluation
    
    def _setEvaluation(self, evaluation):
        self._evaluation = evaluation
        if self.deltas != None:
            self.controller.parameters = self.storedParameters
            if not evaluation:
                self.controller.parameters += self.deltas
        
    evaluation = property(_getEvaluation, _setEvaluation)
    
        
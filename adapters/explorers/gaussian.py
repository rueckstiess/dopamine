from dopamine.adapters.explorers import Explorer
from numpy import random, array, log, exp

class GaussianExplorer(Explorer):
    
    # define the conditions of the environment
    inConditions = {'discreteActions':False}    
    
    # define the conditions of the environment
    outConditions = {}
    
    # use the derivative of the log likelihood to adapt sigma
    sigmaAdaptation = False
    
    def __init__(self, sigma=-2.):
        """ set the variance sigma for the gaussian distribution. """
        Explorer.__init__(self)
        self.sigma = sigma

    def setExperiment(self, experiment):
        Explorer.setExperiment(self, experiment)        
          
    def expln(self, x):
        if x <= 0:
            return exp(x)
        else:
            return log(x+1.)+1.
    
    def explnPrime(self, x):
        if x <= 0:
            return exp(x)
        else:
            return 1./(x+1.)
        
    def _explore(self, action):
        """ add a random number, drawn from N(0, sigma^2), to each dimension
            of the continuous action vector. 
        """
        exploration = random.normal(0, self.expln(self.sigma), size=action.shape)
        action += exploration

        return action

    def getDerivative(self, state, derivs):
        return (derivs**2 - self.expln(self.sigma)**2) / self.expln(self.sigma) * self.explnPrime(self.sigma)
        
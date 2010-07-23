from dopamine.adapters.explorers import Explorer
from numpy import random, array

class GaussianExplorer(Explorer):
    
    # define the conditions of the environment
    inConditions = {'discreteActions':False}    
    
    # define the conditions of the environment
    outConditions = {}
    
    def __init__(self, sigma=0.2):
        """ set the variance sigma for the gaussian distribution. """
        Explorer.__init__(self)
        
        self.sigma = sigma
        
        
    def _explore(self, action):
        """ add a random number, drawn from N(0, sigma^2), to each dimension
            of the continuous action vector. """
        exploration = random.normal(0, sigma, size=action.shape)
        action += exploration

        return action

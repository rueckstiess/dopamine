from dopamine.adapters.explorers import Explorer
from numpy import random, array, where, exp

class BoltzmannExplorer(Explorer):
    
    # define the conditions of the environment
    inConditions = {'discreteActions':True}    
    
    # define the conditions of the environment
    outConditions = {}
    
    def __init__(self, tau, decay=0.999):
        """ set the probability epsilon, with which a random action is chosen. """
        self.tau = tau
        self.decay = decay
        self.state = None
        
    def applyState(self, state):
        self.state = state
        return state
    
    def _explore(self, action):
        """ draw random number r uniformly in [0,1]. if r < epsilon, make random move,
            otherwise return action as is.
        """
        pdf = [exp(self.experiment.agent.estimator.getValue(self.state, array([a]))/self.tau) for a in range(self.experiment.conditions['actionNum'])]
        pdf /= sum(pdf)
        cdf = [sum(pdf[:i+1]).item() for i in range(len(pdf))]
        
        r = random.random()
        self.tau *= self.decay
        return array([min(where(array(cdf) >= r)[0])])
        

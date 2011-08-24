from dopamine.adapters.explorers import Explorer
from numpy import random, array, where, exp

class BoltzmannExplorer(Explorer):
    
    # define the conditions of the environment
    inConditions = {'discreteActions':True}    
    
    # define the conditions of the environment
    outConditions = {}
    
    def __init__(self, tau, decay=0.999):
        """ set the probability tau, with which a random action is chosen. """
        Explorer.__init__(self)
        
        self.tau = tau
        self.decay = decay
        self.state = None
        
    def applyState(self, state):
        self.state = state
        return state
    
    def _explore(self, action):
        """ draw random move with probability proportional to the action's value.
        """
        pdf = [exp(self.experiment.agent.estimator.getValue(self.state, array([a]))/self.tau) for a in range(self.experiment.conditions['actionNum'])]
        pdf /= sum(pdf)
        cdf = [sum(pdf[:i+1]).item() for i in range(len(pdf))]
        
        # disable exploration if tau drops below 0.01
        if self.tau < 0.01:
            self.active = False
            return array([action])
            
        if self.active:
            self.tau *= self.decay
            
        r = random.random()   
        action = sum(array(cdf) < r)

        return array([action])
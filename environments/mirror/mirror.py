import numpy as np
from dopamine.environments.environment import Environment
    

class MirrorEnvironment(Environment):
    """ This is a continuous toy environment with continuous state
        and action. The goal is to return the same value for the action
        that was given in the state.
    """
    
    # define the conditions of the environment
    conditions = {
      'discreteStates':False,
      'stateDim':1,
      'stateNum':np.inf,
      'discreteActions':False,
      'actionDim':1,
      'actionNum':np.inf, 
      'episodic':True
    }

    def __init__(self):
        Environment.__init__(self)
        self.reset()
    
    def episodeFinished(self):
        Environment.episodeFinished(self)
        return self.timestep >= 1
        
    def reset(self):
        Environment.reset(self)
        self.state = np.random.uniform(3, 4)

    def _update(self):
        self.reward = -abs(self.state - self.action)
        # self.state = np.random.uniform(-10, 10, 1)        
        
    
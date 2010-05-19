from numpy import zeros, inf, random, array

from dopamine.environments.environment import Environment
    

class TestEnvironment(Environment):
    """ This is a continuous test environment, which is very easy to learn. 
        This can be used to verify the correctness of learning algorithms.
    """
    
    # define the conditions of the environment
    conditions = {
      'discreteStates':False,
      'stateDim':1,
      'stateNum':inf,
      'discreteActions':True,
      'actionDim':1,
      'actionNum':2, 
      'episodic':True
    }

    actions = [-0.1, 0.1]

    def __init__(self):
        Environment.__init__(self)
        self.reset()
    
    def episodeFinished(self):
        Environment.episodeFinished(self)
        return self.timestep >= 10
        
    def reset(self):
        Environment.reset(self)
        self.state = random.uniform(-1, 1, 1)
        self.target = array([0.4])

    def _update(self):
        self.state += self.actions[self.action]
        diff = -abs(self.state.item() - self.target.item())
        return diff
        
        
    
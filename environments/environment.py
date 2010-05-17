from numpy import zeros, inf
from rllib.tools.utilities import abstractMethod

class EnvironmentException(Exception):
    pass

class Environment(object):
    """ The general interface for whatever we would like to model, learn about, 
        predict, or simply interact in. We can perform actions, and access 
        (partial) observations. 
    """       
    
    # define the conditions of the environment
    conditions = {
      'discreteStates':False,
      'stateDim':0,
      'stateNum':inf,
      'discreteActions':False,
      'actionDim':0,
      'actionNum':inf, 
      'episodic':False
    }                      
    
    def __init__(self):
        # progress counter:
        # 0 = before state has been requested
        # 1 = after state has been requested, before action has been performed
        # 2 = after action has been performed, before reward has been requested
        self.progressCnt = 0
        
        # counts the number of executed interactions (s, a, r) with the environment
        self.timestep = 0
        
        # in case environment has a renderer
        self.renderer = None
        
        # the current state, action, reward. used in _update()
        self.state = None
        self.action = None
        self.reward = None

    
    def getState(self):
        """ the currently visible state of the world (the observation may be 
            stochastic - repeated calls returning different values)
        """
        if self.progressCnt == 0:
            self.progressCnt = 1
            return self.state
        else:
            raise EnvironmentException('state was requested twice before action was given.')
                    
    def performAction(self, action):
        """ perform an action on the world that changes it's internal state (maybe 
            stochastically).
        """
        if self.progressCnt == 1:
            self.action = action
            self._update()
            self.progressCnt = 2
        else:
            if self.progressCnt == 0:
                raise EnvironmentException('action was given before observation was requested.')
            if self.progressCnt > 1:
                raise EnvironmentException('action was given twice, before reward was requested.')

    def getReward(self):
        """ return the reward the agent receives at the current time step. """
        if self.progressCnt == 2:
            self.progressCnt = 0
            self.timestep += 1
            return self.reward
        else:
            raise EnvironmentException('reward was requested before action was performed.')

    def episodeFinished(self):
        """ return whether or not an episode is over. Life-long environments always return False. """
        return False

    def reset(self):
        """ Most environments will implement this optional method that allows for 
            reinitialization. 
        """
        self.progressCnt = 0
        self.timestep = 0
        
    def _update(self):
        """ integrate the action into the environment and set the new state and reward. """
        abstractMethod()

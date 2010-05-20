from dopamine.tools import Episode, History
from numpy import zeros, inf

class AgentException(Exception):
    pass

class Agent(object):
    
    def __init__(self):        
        # current observation, action, reward
        self.state = None
        self.action = None
        self.reward = None
        
        # agent conditions, inherited from environment (plus adapters)
        self.conditions_ = {}
        
        # progress counter:
        # 0 = reward was given, before observation was integrated
        # 1 = integration was integrated, before action was returned
        # 2 = action was returned, before reward was given
        # 0 = reward was given. store experience in history
        self.progressCnt = 0
        
        # enable or disable logging to dataset (for testing)
        self.loggingEnabled = True
        
        
    def _setup(self, conditions):
        """ Tells the agent, if the environment is discrete or continuous and the
            number/dimensionalty of states and actions. This function is called
            just before the first state is integrated.
        """
        self.conditions_ = conditions
        
        # create history to store experiences
        self.history = History(conditions['stateDim'], conditions['actionDim'])
    
    @property
    def conditions(self):
        return self.conditions_
        
    @property
    def episode(self):
        """ returns the last (current) episode. """
        if len(self.history) > 0:
            return self.history[-1]
        else:
            return None
        
    def integrateState(self, state):
        if self.progressCnt == 0:
            self.state = state
            self.progressCnt = 1
        else:
            raise AgentException('observation was given twice before action was requested.')
                
    def getAction(self):
        if self.progressCnt == 1:
            self._calculate()
            self.progressCnt = 2
            return self.action
        else:
            if self.progressCnt == 0:
                raise AgentException('action was requested before observation was integrated.')
            if self.progressCnt > 1:
                raise AgentException('action was requested after reward was given.')
        
    def giveReward(self, reward):
        if self.progressCnt == 2:
            self.reward = reward
            self.progressCnt = 0
            if self.loggingEnabled:
                self.history.append(self.state, self.action, self.reward)
        else:
            raise AgentException('reward was given before action was returned.')
        
    def newEpisode(self):
        self.history.newEpisode()
            
    def learn(self):
        pass
    
    def forget(self):
        """ deletes the entire history. """
        self.history.clear()
        
    def _calculate(self):
        """ this method needs to be overwritten by subclasses to return a non-zero action. """
        self.action = zeros(self.conditions['actionDim'])
    
    
    
from rllib.tools import Episode, History
from numpy import zeros

class AgentException(Exception):
    pass

class Agent(object):
    
    def __init__(self, stateDim, actionDim):
        # store state and action dimensions
        self.stateDim = stateDim
        self.actionDim = actionDim
        
        # create history to store experiences
        self.history = History(stateDim, actionDim)
        
        # current observation, action, reward
        self.obs = None
        self.action = None
        self.reward = None
        
        # progress counter:
        # 0 = reward was given, before observation was integrated
        # 1 = integration was integrated, before action was returned
        # 2 = action was returned, before reward was given
        # 0 = reward was given. store experience in history
        self.progressCnt = 0
        
    @property
    def episode(self):
        """ returns the last (current) episode. """
        return self.history[-1]
        
    def integrateObservation(self, obs):
        if self.progressCnt == 0:
            self.obs = obs
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
            self.history.append(self.obs, self.action, self.reward)
        else:
            raise AgentException('reward was requested before action was returned.')
        
    def newEpisode(self):
        self.history.newEpisode()
    
    def explore(self):
        pass
        
    def learn(self):
        pass
        
    def _calculate(self):
        """ this method needs to be overwritten by subclasses to return a non-zero action. """
        self.action = zeros(self.actionDim)
    
    
    
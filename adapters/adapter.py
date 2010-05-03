class Adapter(object):
    
    # define the conditions of the environment
    inConditions = {}    
    
    # define the conditions of the environment
    outConditions = {}
    
    def applyState(self, state):
        """ apply transformations to state and return it. """
        return state
        
    def applyAction(self, action):
        """ apply transformations to action and return it. """
        return action
    
    def applyReward(self, reward):
        """ apply transformations to reward and return it. """
        return reward
    
    def applyEpisodeFinished(self, episodeFinished):
        """ apply transformations to episodeFinished and return it. """
        return episodeFinished
    
    def reset(self):
        """ reset the adapter. """
        pass
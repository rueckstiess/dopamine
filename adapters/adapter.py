class Adapter(object):
    
    # define the conditions of the environment
    inConditions = {}    
    
    # define the conditions of the environment
    outConditions = {}
    
    # overwrite this value in any subclasses of adapter if your adapter
    # requires some pretraining.
    requirePretraining = 0
    
    def __init__(self):
        self.experiment = None
        
    def setExperiment(self, experiment):
        """ give adapter access to the experiment. """
        self.experiment = experiment
    
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
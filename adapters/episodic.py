from dopamine.adapters import Adapter

class MakeEpisodicAdapter(Adapter):
    """ This adapter makes episodic environments and interrupts 
        the experiment after 'episodeLength' steps.
    """
    
    # can be applied to all conditions
    inConditions = {}
    
    # define the conditions of the environment
    outConditions = {'episodic':True}    
    
    def __init__(self, episodeLength):
        self.episodeLength = episodeLength
        self.counter = 0
    
    def applyReward(self, reward):
        """ use reward function to count interactions. """
        self.counter += 1
        return reward
          
    def applyEpisodeFinished(self, episodeFinished):
        """ stop episode after episodeLength steps or when it is naturally over. """
        return (self.counter >= self.episodeLength) or episodeFinished
    
    def reset(self):
        """ resets the counter. """
        self.counter = 0
    

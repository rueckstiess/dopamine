from rllib.adapters import Adapter

class MakeEpisodicAdapter(Adapter):
    """ This adapter makes episodic environments out of non-episodic ones.
        It interrupts the experiment after 'episodeLength' steps.
    """
    
    # define the conditions of the environment
    inConditions = {'episodic':False}    
    
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
        """ apply transformations to episodeFinished and return it. """
        return self.counter >= self.episodeLength
    
    def reset(self):
        self.counter = 0
    

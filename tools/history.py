from dopamine.tools.episode import Episode

class History(object):

    def __init__(self, stateDim, actionDim):
        self.episodes_ = [Episode(stateDim, actionDim)]
        
    @property
    def stateDim(self):
        return self.episodes_[0].stateDim

    @property
    def actionDim(self):
        return self.episodes_[0].actionDim
    
    def clear(self):
        self.episodes_ = [Episode(self.stateDim, self.actionDim)]
    
    def newEpisode(self):
        episode = self.episodes_[-1]
        if len(episode) > 0:
            self.episodes_.append(Episode(self.stateDim, self.actionDim))
    
    def append(self, state, action, reward):
        self.episodes_[-1].append(state, action, reward)
    
    def appendEpisode(self, episode):
        self.epsiodes_.append(episode)
        
    def extend(self, history):
        self.episodes_.extend(history)
    
    @property
    def episodes(self):
        """ if the last episode is empty, do not consider it. """
        if len(self.episodes_[-1]) == 0:
            return self.episodes_[:-1]
        else:
            return self.episodes_
        
    def __len__(self):
        """ returns the length of episodes (empty episode at the end not considered). """
        return len(self.episodes)
        
    def __iter__(self):
        """ iterates over episodes (empty episode at the end not considered). """        
        for ep in self.episodes:
            yield ep
    
    def __getitem__(self, index):
        """return the episode at the given index. """ 
        return self.episodes[index]
    
    def __str__(self):
        out = []
        for episode in self.episodes:
            out.append(str(episode))
        return "\n\n".join(out)
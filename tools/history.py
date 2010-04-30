from rllib.tools.episode import Episode

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
        
        
    def __len__(self):
        return len(self.episodes_)
        
    def __iter__(self):
        for ep in self.episodes_:
            yield ep
    
    def __getitem__(self, index):
        """Return the sequence at the given index. """ 
        return self.episodes_[index]
    
    def __str__(self):
        out = []
        for episode in self.episodes_:
            out.append(str(episode))
        return "\n\n".join(out)
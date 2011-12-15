from dopamine.tools.episode import Episode
from numpy import zeros, r_

class History(object):

    def __init__(self, stateDim, actionDim):
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.episodes_ = [Episode(stateDim, actionDim)]
        
    # @property
    # def stateDim(self):
    #     return self.episodes_[0].stateDim
    # 
    # @property
    # def actionDim(self):
    #     return self.episodes_[0].actionDim
    
    def clear(self):
        self.episodes_ = [Episode(self.stateDim, self.actionDim)]
    
    def newEpisode(self):
        if len(self.episodes_) == 0 or len(self.episodes_[-1]) > 0:
            self.episodes_.append(Episode(self.stateDim, self.actionDim))
    
    def append(self, state, action, reward):
        self.episodes_[-1].append(state, action, reward)
    
    def appendEpisode(self, episode):
        episode = self.episodes_[-1]
        if len(episode) == 0:
            self.episodes_ = self.episodes_[:-1]
        self.episodes_.append(episode)
        if len(episode) == 0:
            self.newEpisode()
        
    def extend(self, history):
        episode = self.episodes_[-1]
        if len(episode) == 0:
            self.episodes_ = self.episodes_[:-1]
        self.episodes_.extend(history)
        if len(episode) == 0:
            self.newEpisode()
        
    def numTotalSamples(self):
        return sum([len(e) for e in self.episodes])
        
    def pop(self):
        """ returns and removes the last non-empty episode in the history.
            after this call, history will always have a new empty episode
            as it's last element onto which new samples are added.
        """
        if len(self.episodes) == 0:
            raise IndexError('pop from empty history')
            
        episode = self.episodes[-1]
        self.episodes_ = self.episodes[:-1]
        self.newEpisode()
        return episode
    
    def truncate(self, n, newest=True):
        """ truncates the history to leave only n episodes. if newest is set
            to True (default), the n most recent episodes are kept, else the n
            oldest episodes are kept. 
        """
        if newest:
            self.episodes_ = self.episodes_[-n:]
        else:
            self.episodes_ = self.episodes_[:n]
            
    def keepBest(self, n):
        """ selects the best n episodes (with highest sum of rewards) and
            discards the others.
        """        
        returns = [sum(e.rewards) for e in self.episodes]
        decorated = [(r, i) for i,r in enumerate(returns)]
        decorated.sort(reverse=True)
        
        new_episodes = [self.episodes[i] for r,i in decorated[:n]]
        self.episodes_ = new_episodes

        self.newEpisode()
        
    
    @property
    def episodes(self):
        """ if the last episode is empty, do not consider it. """
        if len(self.episodes_[-1]) == 0:
            return self.episodes_[:-1]
        else:
            return self.episodes_
    
    @property
    def states(self):
        """ return array of all states over all episodes in shape n x stateDim """
        st = zeros((0, self.stateDim))
        for e in self.episodes:
            st = r_[st, e.states.reshape(len(e), self.stateDim)]
        return st
    
    @property
    def actions(self):
        """ return array of all actions over all episodes in shape n x actionDim """
        ac = zeros((0, self.actionDim))
        for e in self.episodes:
            ac = r_[ac, e.actions.reshape(len(e), self.actionDim)]
        return ac
    
    @property
    def rewards(self):
        """ return array of all rewards over all episodes in shape n x 1 """
        rew = zeros((0, 1))
        for e in self.episodes:
            rew = r_[rew, e.rewards.reshape(len(e), 1)]
        return rew
        
            
    def __len__(self):
        """ returns the number of episodes (empty episode at the end not considered). """
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
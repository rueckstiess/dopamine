import numpy as np

class Episode(object):
    
    def __init__(self, stateDim, actionDim):
        """ Initialize the episode, create the numpy arrays that store states, actions and rewards.
            stateDim: dimension of state vector
            actionDim: dimension of action vector
            (note: reward is always scalar and therefore 1-dimensional)
        """
        # numpy arrays containing states, actions, rewards
        self.states_ = np.zeros((0, stateDim), dtype=float)
        self.actions_ = np.zeros((0, actionDim), dtype=float)
        self.rewards_ = np.zeros([], dtype=float)
        self.nextstates_ = np.zeros((0, stateDim), dtype=float)
        
        # length of the actual data (the arrays might be larger and padded with zeros for efficiency)
        self.length = 0
        
                
    def clear(self):
        """ clears the episode, resetting all the arrays to zero length. """
        self.states_ = np.zeros((0, self.states_.shape[1]), dtype=float)
        self.actions_ = np.zeros((0, self.actions_.shape[1]), dtype=float)
        self.rewards_ = np.zeros([], dtype=float)
        self.nextstates_ = np.zeros((0, self.nextstates_.shape[1]), dtype=float)
        self.length = 0
            
    @property
    def stateDim(self):
        return self.states_.shape[1]
    
    @property
    def actionDim(self):
        return self.actions_.shape[1]
                
    @property
    def states(self):
        """ return only the states. """
        return self.states_[:self.length, :]
    
    @property
    def actions(self):
        """ return only the actions. """
        return self.actions_[:self.length, :]
            
    @property
    def rewards(self):
        """ return only the rewards. """
        return self.rewards_[:self.length]

    @property
    def nextstates(self):
        return self.nextstates_[:self.length, :]

        
    def append(self, state, action, reward, nextstate=None):
        """ append one state, action, reward and nextstate to the respective arrays. 
            if nextstate is None (default), it will be lazy-evaluated when iterating
            over the episode. The very last nextstate in an episode always remains None.
        """
        # check whether a resize is necessary
        shape = self.states_.shape
        if shape[0] <= self.length:
            self.states_.resize((2*shape[0]+1, self.states_.shape[1]), refcheck=False)
            self.actions_.resize((2*shape[0]+1, self.actions_.shape[1]), refcheck=False)
            self.rewards_.resize((2*shape[0]+1,), refcheck=False)
            self.nextstates_.resize((2*shape[0]+1, self.nextstates_.shape[1]), refcheck=False)
        
        # append state, action, reward to arrays
        self.states_[self.length, :] = np.asarray(state)
        self.actions_[self.length, :] = np.asarray(action)
        self.rewards_[self.length] = np.asarray(reward)
        if nextstate == None:
            self.nextstates_[self.length, :] = np.nan
        else:
            self.nextstates_[self.length, :] = np.asarray(nextstate)

        # check if the last nextstate was None and fill with correct nextstate
        if self.length > 0 and np.any(np.isnan(self.nextstates_[self.length-1, :])):
            self.nextstates[self.length-1, :] = np.asarray(state)
        
        # increase the length counter
        self.length += 1
        
    def extend(self, episode):
        """ appends the given episode to this one. """
        if isinstance(episode, Episode):
            self.states_ = np.r_[self.states, episode.states]
            self.actions_ = np.r_[self.actions, episode.actions]
            self.rewards_ = np.r_[self.rewards, episode.rewards]
            self.nextstates_ = np.r_[self.nextstates, episode.nextstates]

            # if the last nextstate is None, replace with the
            # values from the first nextstate of the extending episode
            if self.length > 0 and np.any(np.isnan(self.nextstates_[self.length-1,:])):
                self.nextstates_[self.length-1,:] = episode.states_[0,:].copy()

            self.length = len(self.rewards_)
        
    
    def setArrays(self, states, actions, rewards, nextstates=None):
        """ set the states, actions, rewards and nextstates arrays manually. they need 
            to be the same length. nextstates are treated the same as in append(). 
        """
        if nextstates == None:
            nextstates = np.r_[states[1:,:], np.zeros((1,self.stateDim)) * np.nan]

        assert(states.shape[0] == actions.shape[0] == len(rewards) == nextstates.shape[0])

        self.states_ = states
        self.actions_ = actions
        self.rewards_ = rewards
        self.nextstates_ = nextstates
        self.length = len(rewards)
        
    def __len__(self):
        """ returns the length (number of interactions) of the episode. """
        return self.length

    def __getitem__(self, index):
        """Return the interaction at the given index. """ 
        return self.states[index,:], self.actions[index,:], self.rewards[index], self.nextstates[index]
    
    def __iter__(self):
        """ iterate over episode and return state, action, reward and next state for each iteration.
            the last iteration returns state as next state. """
        for i in xrange(self.length):
            s = self.states[i,:]
            a = self.actions[i,:]
            r = self.rewards[i]
            ns = self.nextstates[i,:]
            if np.any(np.isnan(ns)):
                ns = None
            yield(s, a, r, ns)
    
    def reversedSamples(self):
        """ iterate over all samples in history (like __iter__) but
            in reversed order. return state, action, reward and next state
            for each iteration. 
        """
        for i in xrange(self.length-1, -1, -1):
            s = self.states[i,:]
            a = self.actions[i,:]
            r = self.rewards[i]
            ns = self.nextstates[i,:]
            if np.any(np.isnan(ns)):
                ns = None
            yield(s, a, r, ns)
    
    def randomizedSamples(self):
        """ iterate over all samples in history (like __iter__) but
            in random order. return state, action, reward and next state for each
            iteration. the (originally) last sample returns state as next state. """
        for i in np.random.permutation(xrange(self.length)):
            s = self.states[i,:]
            a = self.actions[i,:]
            r = self.rewards[i]
            ns = self.nextstates[i,:]
            if np.any(np.isnan(ns)):
                ns = None
            yield(s, a, r, ns)
    
    def __str__(self):
        """ string representation of episode. prints states, actions, rewards. """
        out = []
        for s, a, r, ns in self:
            out.append(str(s) + str(a) + ' ' + str(r) + ' ' + str(ns))
        return "\n".join(out)
        


    
    
    
        
    
        
            
            
    
    
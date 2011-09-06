from numpy import *

class Episode(object):
    
    def __init__(self, stateDim, actionDim):
        """ Initialize the episode, create the numpy arrays that store states, actions and rewards.
            stateDim: dimension of state vector
            actionDim: dimension of action vector
            (note: reward is always scalar and therefore 1-dimensional)
        """
        # numpy arrays containing states, actions, rewards
        self.states_ = zeros((0, stateDim))
        self.actions_ = zeros((0, actionDim))
        self.rewards_ = zeros([])
        
        # length of the actual data (the arrays might be larger and padded with zeros for efficiency)
        self.length = 0
        
                
    def clear(self):
        """ clears the episode, resetting all the arrays to zero length. """
        self.states_ = zeros((0, self.states_.shape[1]))
        self.actions_ = zeros((0, self.actions_.shape[1]))
        self.rewards_ = zeros([])
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
        
    def append(self, state, action, reward):
        """ append one state, action and reward to the respective arrays. """
        
        # check whether a resize is necessary
        shape = self.states_.shape
        if shape[0] <= self.length:
            self.states_.resize((2*shape[0]+1, self.states_.shape[1]), refcheck=False)
            self.actions_.resize((2*shape[0]+1, self.actions_.shape[1]), refcheck=False)
            self.rewards_.resize((2*shape[0]+1,), refcheck=False)
        
        # append state, action, reward to arrays
        self.states_[self.length, :] = asarray(state)
        self.actions_[self.length, :] = asarray(action)
        self.rewards_[self.length] = asarray(reward)
        
        # increase the length counter
        self.length += 1
        
    def extend(self, episode):
        """ appends the given episode to this one. """
        if isinstance(episode, Episode):
            self.states_ = r_[self.states, episode.states]
            self.actions_ = r_[self.actions, episode.actions]
            self.rewards_ = r_[self.rewards, episode.rewards]
            self.length = len(self.rewards_)
        
    
    def setArrays(self, states, actions, rewards):
        """ set the states, actions, rewards arrays manually. they need to be the same length. """
        if states.shape[0] == actions.shape[0] == len(rewards):
            self.states_ = states
            self.actions_ = actions
            self.rewards_ = rewards
            self.length = len(rewards)
        
    def __len__(self):
        """ returns the length (number of interactions) of the episode. """
        return self.length

    def __getitem__(self, index):
        """Return the interaction at the given index. """ 
        return self.states[index,:], self.actions[index,:], self.rewards[index]
    
    def __iter__(self):
        """ iterate over episode and return state, action, reward and next state for each iteration.
            the last iteration returns state as next state. """
        for i in xrange(self.length):
            s = self.states[i,:]
            a = self.actions[i,:]
            r = self.rewards[i]
            if i+1 < self.length:
                ns = self.states[i+1,:]
            else:
                # changed to return None instead
                # ns = self.states[i,:]
                ns = None
            yield(s, a, r, ns)
    
    def randomizedSamples(self):
        """ iterate over all samples in history (like __iter__) but
            in random order. return state, action, reward and next state for each
            iteration. the (originally) last sample returns state as next state. """
        for i in random.permutation(xrange(self.length)):
            s = self.states[i,:]
            a = self.actions[i,:]
            r = self.rewards[i]
            if i+1 < self.length:
                ns = self.states[i+1,:]
            else:
                ns = self.states[i,:]
            yield(s, a, r, ns)
    
    def __str__(self):
        """ string representation of episode. prints states, actions, rewards. """
        out = []
        for s, a, r, _ in self:
            out.append(str(s) + str(a) + ' ' + str(r))
        return "\n".join(out)
        


    
    
    
        
    
        
            
            
    
    
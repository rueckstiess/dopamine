class Adapter(object):
    
    def applyState(self, state):
        """ apply transformations to state and return it. """
        return state
        
    def applyAction(self, action):
        """ apply transformations to action and return it. """
        return action
    
    def applyReward(self, reward):
        """ apply transformations to reward and return it. """
        return reward
    
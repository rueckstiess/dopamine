from dopamine.tools.utilities import abstractMethod

class Actor(object):
    
    conditions = {'discreteStates':True, 'discreteActions':True}
        
    def getAction(self, state):
        """ returns the learned action with the given state. """
        abstractMethod()
    
    def updateAction(self, state, action):
        """ updates the action for a given state. Next call to 
            getAction(state) should return an action that is closer
            to the given action. """
        pass
    
    def reset(self):
        """ for actors that collect samples and require training,
            this function should clear the entire dataset and reset 
            the actor.
        """
        pass
        
    def train(self):
        """ for actors that collect samples and require training,
            this function should run one training step.
        """
        pass
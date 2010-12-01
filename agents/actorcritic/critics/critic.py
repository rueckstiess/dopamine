from dopamine.tools.utilities import abstractMethod

class Critic(object):
    
    conditions = {'discreteStates':True, 'discreteActions':True}
        
    def getValue(self, state):
        """ returns the value for the given state. """
        abstractMethod()
    
    def updateValue(self, state, value):
        """ updates the value for a given state. """
        pass
    
    def reset(self):
        """ for critics that collect samples and require training,
            this function should clear the entire dataset and reset 
            the critic.
        """
        pass
        
    def train(self):
        """ for critics that collect samples and require training,
            this function should run one training step.
        """
        pass
from rllib.tools.utilities import abstractMethod

class Environment(object):
    """ The general interface for whatever we would like to model, learn about, 
        predict, or simply interact in. We can perform actions, and access 
        (partial) observations. 
    """       
    
    def __init__(self):
        # define the state and action dimensionality
        self.actionDim = 0
        self.stateDim = 0
        
        # define if states and/or actions are discrete (rather than continuous)
        self.discreteStates = False
        self.discreteActions = False
    
    def getState(self):
        """ the currently visible state of the world (the observation may be 
            stochastic - repeated calls returning different values)
        """
        abstractMethod()
                    
    def performAction(self, action):
        """ perform an action on the world that changes it's internal state (maybe 
            stochastically).
        """
        abstractMethod()

    def reset(self):
        """ Most environments will implement this optional method that allows for 
            reinitialization. 
        """
        pass
        
    

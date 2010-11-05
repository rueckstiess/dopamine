from dopamine.agents import Agent, AgentException
from dopamine.agents.direct import *

from numpy import zeros, inf

class DirectAgent(Agent):
    
    def __init__(self, controllerClass=LinearController):
        Agent.__init__(self)
        self.controllerClass = controllerClass
             
    def _setup(self, conditions):
        """ Tells the agent, if the environment is discrete or continuous and the
            number/dimensionalty of states and actions. This function is called
            just before the first state is integrated.
        """
        Agent._setup(self, conditions)
        
        # direct learning agents require continuous states/actions
        if self.conditions['discreteStates'] or self.conditions['discreteActions']:
            raise AgentException('DirectAgent expects continuous states and actions. Use adapter or a different environment.')
        
        if not self.conditions['episodic']:
            raise AgentException('DirectAgent expects an episodic environment. Use adapter or different environment.')
            
        self.controller = self.controllerClass(self.conditions['stateDim'], self.conditions['actionDim'])
        
    def learn(self):
        pass
            
    def _calculate(self):
        """ calls the controller's activate and returns the result """
        self.action = self.controller.activate(self.state)
    
    

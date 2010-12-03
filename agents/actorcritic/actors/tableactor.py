from dopamine.agents.actorcritic.actors.actor import Actor
from numpy import ones, ndarray, argmax, where, array
from random import choice
import types

class TableException(Exception):
    pass    

class TableActor(Actor):
    
    conditions = {'discreteStates':True, 'discreteActions':True}
    beta = 0.1
    
    def __init__(self, stateNum, actionNum):
        self.stateNum = stateNum
        self.actionNum = actionNum
        self.initialize(1.)
        
    def getAction(self, state):
        """ returns the best action (highest probability) with the given state. """
        state = self._forceScalar(state)
        bestvalue = max(self.policy[state, :])
        return array([choice(where(self.policy[state, :] == bestvalue)[0])])
    
    def updateAction(self, state, action, delta):
        """ updates the action for a given state. Next call to 
            getAction(state) should return an action that is closer
            to the given action. """
        if delta > 0:
            state = self._forceScalar(state)
            action = self._forceScalar(action)
            self.policy[state, action] += self.beta # * delta
    
    def initialize(self, value=0.0):
        """ Initialize the whole table with the given value. """
        self.policy = ones((self.stateNum, self.actionNum)) * value

    def _forceScalar(self, value):
        """ accepts scalars, lists and arrays. scalars are just passed through, while
            lists and arrays must have one single element. that element is passed back
            as a scalar. lists/arrays with more than one element raise an Exception.
        """
        if type(value) in [ndarray, types.ListType]:
            if len(value) > 1:
                raise TableException('TableActor accepts only scalars or lists/arrays with one element as state or action.')
            if type(value) == ndarray:
                value = value.item()
            else:
                value = value[0]

        return int(value)
        
    def __str__(self):
        return str(self.policy)

from dopamine.agents import Agent, AgentException
from dopamine.agents.actorcritic import TableActor, TableCritic

class ActorCriticAgent(Agent):
    
    alpha = 0.9
    gamma = 0.9
    
    def __init__(self, actorClass=TableActor, criticClass=TableCritic):
        """ initialize the agent with the actor and critic classes. """
        Agent.__init__(self)
        self.actorClass = actorClass
        self.criticClass = criticClass
    
    def _setup(self, conditions):
        """ create actor and critic and check if they are compatible to environment. """
        Agent._setup(self, conditions)

        if self.conditions['discreteStates']:
            ndStates = self.conditions['stateNum']
        else:
            ndStates = self.conditions['stateDim']
        
        if self.conditions['discreteActions']:
            ndActions = self.conditions['actionNum']
        else:
            ndActions = self.conditions['actionDim']
         
        self.critic = self.criticClass(ndStates)
        self.actor = self.actorClass(ndStates, ndActions)

        if (self.conditions['discreteStates'] != self.critic.conditions['discreteStates']) or \
           (self.conditions['discreteStates'] != self.actor.conditions['discreteStates']) or \
           (self.conditions['discreteActions'] != self.critic.conditions['discreteActions']) or \
           (self.conditions['discreteActions'] != self.actor.conditions['discreteActions']):
            raise AgentException('Environment is not compatible to critic/actor. Please check discreteStates/discreteActions.')

        self.conditions['discreteStates'] == self.critic.conditions['discreteStates']
        self.conditions['discreteActions'] == self.actor.conditions['discreteActions']

    def _calculate(self):
        self.action = self.actor.getAction(self.state)

    def learn(self):
        """ go through whole episode and make updates to state-value function and actor. """  

        self.critic.reset()
        self.actor.reset()
        
        for episode in self.history:
            for state, action, reward, nextstate in episode:

                # update critic
                vvalue = self.critic.getValue(state)
                vnext = self.critic.getValue(nextstate)
                delta = reward + self.gamma * vnext - vvalue
                self.critic.updateValue(state, self.alpha * delta)
                
                # update actor
                self.actor.updateAction(state, action, delta)

        self.critic.train()
        self.actor.train()

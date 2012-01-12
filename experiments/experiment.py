from dopamine.adapters.explorers.explorer import Explorer
from dopamine.tools.history import History
from matplotlib import pyplot as plt
from numpy import mean
import sys

class ExperimentException(Exception):
    pass

class Experiment(object):
    
    def __init__(self, environment, agent):
        self.environment = environment
        self.agent = agent
        self.adapters_ = []
        self.visual_x = []
        self.visual_y = []
        self.numEpisodes = 0
        
        # marker that stores if setup on agent has been completed
        self.setupComplete_ = False
            
    def _flattenConditions(self):
        """ flattens the environment and adapters conditions to one conditions dictionary. """
        conditions = dict([(k,v) for k,v in self.environment.conditions.items()])
        conditions['requirePretraining'] = 0
        
        for a in self.adapters_:
            conditions['requirePretraining'] = max(conditions['requirePretraining'], a.requirePretraining)
            for k,v in a.outConditions.items():
                conditions[k] = v
    
        return conditions
    
    @property
    def conditions(self):
        """ returns the conditions after applying all adapters (read-only). """
        return self._flattenConditions()
    
    @property
    def adapters(self):
        """ returns the list of adapters (read-only). """
        return self.adapters_
        
    @property
    def explorers(self):
        """ return list of adapters, that are explorers (read-only). The
            list is returned in reverse order, so that the bottom-most 
            explorer (closest to the agent) is returned first. 
        """
        return [a for a in reversed(self.adapters) if isinstance(a, Explorer)]


    def addAdapter(self, adapter):
        """ add an adapter to the end of the adapter list if it is compatible. """
        # get the flattened conditions dictionary
        conditions = self.conditions

        # check for each inCondition if it is compatible to the adapter stack
        for c,v in adapter.inConditions.items():
            if not c in conditions:
                # condition could not be found in adapter stack or environment
                raise ExperimentException('condition "%s" could not be found in adapter stack or environment.'%c)
            
            if conditions[c] != v:
                # condition could be found but does not match
                raise ExperimentException('%s requires condition "%s" to be %s. Not compatible to environment/adapter stack with value %s.'%(adapter.__class__.__name__, c, v, conditions[c]))
        
        # every condition matches, set experiment and add to adapter list
        adapter.setExperiment(self)
        self.adapters_.append(adapter)
        
        # conditions have changed, new agent setup is necessary
        self.setupComplete_ = False
    

    def _performSetup(self):
        """ provides the agents with the flattened conditions. Also executes
            any necessary pretraining runs beforehand, if needed by any of the
            adapters.
        """
        self.agent._setup(self.conditions)
        self.agent.experiment = self
                
        self.setupComplete_ = True
        
        # some adapters require pretraining. check if this is the case here.
        if self.conditions['requirePretraining'] > 0:
            print "pre-training...",
            sys.stdout.flush()
            self.evaluateEpisodes(self.conditions['requirePretraining'], reset=True, exploration=True, visualize=False)
            print "done."


    def setup(self):
        """ forces the execution of the setup including pretraining, which would 
            normally be executed only when the first interaction is called. 
        """
        if not self.setupComplete_:
            self._performSetup()
    
            
    def interact(self):
        """ run one interaction between agent and environment. The state from
            the environment gets passed through all registered adapters before
            it is presented to the agent. The agent returns an action, which
            goes again through all adapters (in reverse order) and is then 
            executed in the environment. A reward is then passed through the
            adapters to the agent.
        """
        if not self.setupComplete_:
            self._performSetup()
            
        state = self.environment.getState()
        for adapter in self.adapters_:
            state = adapter.applyState(state)
        self.agent.integrateState(state)
        
        action = self.agent.getAction()
        for adapter in reversed(self.adapters_):
            action = adapter.applyAction(action)
        self.environment.performAction(action)
        
        reward = self.environment.getReward()        
        for adapter in self.adapters_:
            reward = adapter.applyReward(reward)
        self.agent.giveReward(reward)
    
    def runEpisode(self, reset=True):
        """ resets the environment if reset is True, then calls interact() until 
            the environment and its adapters signals the end of an episode. 
        """
        if not self.conditions['episodic']:
            raise ExperimentException('Environment is not episodic, or adapters transformed it into non-episodic. Use interact() method.')
        
        # reset adapters and environment
        if reset:
            for adapter in self.adapters_:
                adapter.reset()
            self.environment.reset()
        
        # get environment's episodeFinished and push it through all adapters
        episodeFinished = self.environment.episodeFinished()
        for adapter in self.adapters_:
            episodeFinished = adapter.applyEpisodeFinished(episodeFinished)
        
        # while the resulting episodeFinished is False, loop over interactions
        while not episodeFinished:
            self.interact()
            episodeFinished = self.environment.episodeFinished()
            for adapter in self.adapters_:
                episodeFinished = adapter.applyEpisodeFinished(episodeFinished)
        
        # tell agent that it should start a new episode
        self.agent.newEpisode()
        self.numEpisodes += 1
    
    def runEpisodes(self, count, reset=True):
        for i in range(count):
            self.runEpisode(reset)
    
    def evaluateEpisodes(self, count, reset=True, exploration=False, visualize=True):
        self.setup()
        
        # tell agent that evaluation is taking place
        self.agent.evaluation = True
    
        currNumEpisodes = len(self.agent.history)

        # disable all explorers and store them for later        
        if not exploration:
            explorers = []
            for a in self.adapters_:
                if isinstance(a, Explorer) and a.active:
                    explorers.append(a)
                    a.active = False
            
        # run experiment for evaluation and store history
        self.runEpisodes(count, reset)
        self.numEpisodes -= count
        
        # copy the latest episodes to a new history
        history = History(self.agent.history.stateDim, self.agent.history.actionDim)
        history.episodes_ = self.agent.history.episodes_[currNumEpisodes:-1]

        # remove the evaluation histories from the agent
        self.agent.history.episodes_ = self.agent.history.episodes_[:currNumEpisodes] + [self.agent.history.episodes_[-1]]
        
        # enable exploration again if disabled before
        if not exploration:
            for a in explorers:
                a.active = True
        
        # tell agent that evaluation is over
        self.agent.evaluation = False

        if visualize:
            plt.ion()
            plt.clf()
            self.visual_x.append(self.numEpisodes)
            self.visual_y.append(mean([sum(e.rewards) for e in history]))
            plt.plot(self.visual_x, self.visual_y, 'o-', color='black')
            plt.gcf().canvas.draw()
                
        return history
        
        
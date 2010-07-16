from expsuite import ExperimentSuite

from dopamine.environments import DiscreteCartPoleEnvironment, CartPoleRenderer
from dopamine.agents import FQIAgent, RBFEstimator, LWPREstimator
from dopamine.experiments import Experiment
from dopamine.adapters import EpsilonGreedyExplorer, NormalizingAdapter, IndexingAdapter

from numpy import *
import os, time

class Suite(ExperimentSuite):
    
    def __init__(self):
        ExperimentSuite.__init__(self)
        self.restore_supported = False
    
    def reset(self, params, rep):
        # seed random number generator
        random.seed(int(os.getpid() + time.time()))
        
        # create agent, environment, renderer, experiment
        self.agent = FQIAgent(estimatorClass=eval(params['estimator']))
        self.environment = DiscreteCartPoleEnvironment()
        self.experiment = Experiment(self.environment, self.agent)

        # cut off last two state dimensions
        if params['indexing']:
            indexer = IndexingAdapter([0, 1], None)
            self.experiment.addAdapter(indexer)

        # add normalization adapter
        normalizer = NormalizingAdapter()
        self.experiment.addAdapter(normalizer)

        # add e-greedy exploration
        self.explorer = EpsilonGreedyExplorer(params['epsilon'], 0.99995)
        self.experiment.addAdapter(self.explorer)
        
        # set alpha and gamma
        self.agent.alpha = params['alpha']
        self.agent.gamma = params['gamma']
        
    
    def iterate(self, params, rep, n):
        self.experiment.runEpisodes(params['numepisodes'])
        self.agent.learn()
        
        if params['keepbest'] > 0:
            self.agent.history.keepBest(params['keepbest'])
        if params['truncate'] > 0:
            self.agent.history.truncate(params['truncate'])

        valdata = self.experiment.evaluateEpisodes(20, visualize=False)
        exploration = self.explorer.epsilon
        ret = mean([sum(v.rewards) for v in valdata])
        numsamples = len(self.agent.history.rewards)
        
        ret = {'iteration':n, 'epsilon':exploration, 'return':ret, 'samples':numsamples}
        print ret
        return ret


if __name__ == '__main__':
    suite = Suite()
    suite.start()

from expsuite import ExperimentSuite

from dopamine.environments import CartPoleEnvironment
from dopamine.agents import FQIAgent, BASAgent, NNEstimator, RBFEstimator
from dopamine.experiments import Experiment
from dopamine.adapters import EpsilonGreedyExplorer, NormalizingAdapter, IndexingAdapter

from numpy import *
import os, time

class Suite(ExperimentSuite):
    
    def __init__(self):
        ExperimentSuite.__init__(self)
        self.restore_supported = False
    
    def reset(self, params, rep):
        """ needs to be implemented by subclass """
        # seed random number generator
        random.seed(int(os.getpid() + time.time()))
        
        # create agent, environment, renderer, experiment
        self.agent = BASAgent(estimatorClass=eval(params['estimator']))
        self.agent.nres = params['recursiondepth']
        self.environment = CartPoleEnvironment()
        self.experiment = Experiment(self.environment, self.agent)

        # cut off last two state dimensions
        indexer = IndexingAdapter([0, 1], None)
        self.experiment.addAdapter(indexer)

        # add normalization adapter
        # normalizer = NormalizingAdapter()
        # self.experiment.addAdapter(normalizer)

        # add e-greedy exploration
        # self.explorer = EpsilonGreedyExplorer(params['epsilon'], 1.0)
        # self.experiment.addAdapter(self.explorer)
        # self.explorer.decay = 0.999
        
    
    def iterate(self, params, rep, n):
        """ needs to be implemented by subclass """
        
        self.experiment.runEpisodes(params['numepisodes'])
        self.agent.learn()

        valdata = self.experiment.evaluateEpisodes(20, visualize=False)
        ret = mean([sum(v.rewards) for v in valdata])
        numsamples = len(self.agent.history.rewards)
        
        ret = {'iteration':n, 'return':ret, 'samples':numsamples}
        print ret
        
        return ret


if __name__ == '__main__':
    suite = Suite()
    suite.start()

from dopamine.environments import DiscreteCartPoleEnvironment, CartPoleRenderer
from dopamine.agents import FQIAgent, RBFEstimator, NNEstimator, RBFOnlineEstimator, LWPREstimator
from dopamine.experiments import Experiment
from dopamine.adapters import EpsilonGreedyExplorer, NormalizingAdapter, IndexingAdapter

from matplotlib import pyplot as plt
from numpy import *

# create agent, environment, renderer, experiment
agent = FQIAgent(estimatorClass=RBFEstimator)
agent.iterations = 1
environment = DiscreteCartPoleEnvironment()
environment.conditions['actionNum'] = 2
environment.centerCart = False
experiment = Experiment(environment, agent)

# cut off last two state dimensions
indexer = IndexingAdapter([0, 1], None)
experiment.addAdapter(indexer)

# add normalization adapter
normalizer = NormalizingAdapter()
experiment.addAdapter(normalizer)

# add e-greedy exploration
explorer = EpsilonGreedyExplorer(0.3, 0.99995)
experiment.addAdapter(explorer)

# renderer = CartPoleRenderer()
# environment.renderer = renderer
# renderer.start()
    
# run experiment
for i in range(1000):
    experiment.runEpisodes(1)
    agent.learn()
    # agent.history.truncate(20)
    # agent.forget()
    
    valdata = experiment.evaluateEpisodes(20, visualize=True)
    mean_return = mean([sum(v.rewards) for v in valdata])
    print "exploration", explorer.epsilon
    print "mean return", mean_return
    print "num episodes", len(agent.history)
    print "num total samples", agent.history.numTotalSamples()

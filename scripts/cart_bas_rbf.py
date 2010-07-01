from dopamine.environments import CartPoleEnvironment, CartPoleRenderer
from dopamine.agents import BASAgent, RBFEstimator, NetworkEstimator
from dopamine.experiments import Experiment
from dopamine.adapters import EpsilonGreedyExplorer, NormalizingAdapter, IndexingAdapter

from matplotlib import pyplot as plt
from numpy import *


# create agent, environment, renderer, experiment
agent = BASAgent(estimatorClass=NetworkEstimator)
environment = CartPoleEnvironment()
experiment = Experiment(environment, agent)

# cut off last two state dimensions
indexer = IndexingAdapter([0, 1], None)
experiment.addAdapter(indexer)

# add normalization adapter
normalizer = NormalizingAdapter(scaleActions=[(-50, 50)])
experiment.addAdapter(normalizer)

# # add e-greedy exploration
# explorer = EpsilonGreedyExplorer(0.4, 1.0)
# experiment.addAdapter(explorer)

experiment.runEpisodes(10)
agent.forget()

# explorer.decay = 0.999
# renderer = CartPoleRenderer()
# environment.renderer = renderer
# renderer.start()

# run experiment
for i in range(100):
    experiment.runEpisodes(5)
    agent.learn()

    # agent.forget()
    
    valdata = experiment.evaluateEpisodes(10, visualize=True)
    # print "exploration", explorer.epsilon
    print "mean return", mean([sum(v.rewards) for v in valdata])
    print "num episodes", len(agent.history)
    # print "num total samples", agent.history.numTotalSamples()


from dopamine.environments import DiscreteCartPoleEnvironment, CartPoleRenderer
from dopamine.agents import FQIAgent, RBFEstimator
from dopamine.agents.valuebased.lwprestimator import LWPREstimator
from dopamine.experiments import Experiment
from dopamine.adapters import EpsilonGreedyExplorer, NormalizingAdapter, IndexingAdapter

from matplotlib import pyplot as plt
from numpy import *


# create agent, environment, renderer, experiment
agent = FQIAgent(estimatorClass=LWPREstimator)
environment = DiscreteCartPoleEnvironment()
environment.centerCart = False
experiment = Experiment(environment, agent)

# cut off last two state dimensions
indexer = IndexingAdapter([0, 1], None)
experiment.addAdapter(indexer)

# add normalization adapter
normalizer = NormalizingAdapter()
experiment.addAdapter(normalizer)

# add e-greedy exploration
explorer = EpsilonGreedyExplorer(0.3, 1.0)
experiment.addAdapter(explorer)

explorer.decay = 0.99999
# renderer = CartPoleRenderer()
# environment.renderer = renderer
# renderer.start()

# run experiment
for i in range(100):
    experiment.runEpisodes(100)
    agent.learn()
    # agent.forget()

    valdata = experiment.evaluateEpisodes(20, visualize=True)
    print "exploration", explorer.epsilon
    print "mean return", mean([sum(v.rewards) for v in valdata])
    print "num episodes", len(agent.history)
    # print "num total samples", agent.history.numTotalSamples()


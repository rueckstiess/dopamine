from dopamine.environments import DiscreteCartPoleEnvironment, CartPoleRenderer
from dopamine.agents import FQIAgent
from dopamine.experiments import Experiment
from dopamine.adapters import EpsilonGreedyExplorer, NormalizingAdapter, IndexingAdapter

from matplotlib import pyplot as plt
from numpy import *


# create agent, environment, renderer, experiment
agent = FQIAgent()
environment = DiscreteCartPoleEnvironment()
experiment = Experiment(environment, agent)

# cut off last two state dimensions
indexer = IndexingAdapter([0, 1], None)
experiment.addAdapter(indexer)

# add normalization adapter
normalizer = NormalizingAdapter()
experiment.addAdapter(normalizer)

# add e-greedy exploration
explorer = EpsilonGreedyExplorer(0.2, episodeCount=100)
experiment.addAdapter(explorer)

experiment.runEpisodes(10)
agent.forget()

explorer.decay = 0.999
renderer = CartPoleRenderer()
environment.renderer = renderer
renderer.start()

# run experiment
for i in range(100):
    experiment.runEpisodes(1)
    agent.learn()

    valdata = experiment.evaluateEpisodes(5)
    print "exploration", explorer.epsilon
    print "mean return", mean([sum(v.rewards) for v in valdata])
    print "num episodes", len(agent.history)


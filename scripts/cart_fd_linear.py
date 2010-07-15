from dopamine.environments import CartPoleEnvironment, CartPoleRenderer
from dopamine.agents import FiniteDifferenceAgent, LinearController
from dopamine.adapters import IndexingAdapter, NormalizingAdapter
from dopamine.experiments import Experiment
from numpy import *

environment = CartPoleEnvironment()
agent = FiniteDifferenceAgent()
experiment = Experiment(environment, agent)

# cut off last two state dimensions
indexer = IndexingAdapter([0, 1], None)
experiment.addAdapter(indexer)

# add normalization adapter
normalizer = NormalizingAdapter(scaleActions=[(-50, 50)])
experiment.addAdapter(normalizer)

# run experiment
for i in range(1000):
    experiment.runEpisodes(10)
    agent.learn()
    agent.forget()

    valdata = experiment.evaluateEpisodes(10)
    print "mean return", mean([sum(v.rewards) for v in valdata])


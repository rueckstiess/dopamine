from dopamine.environments import DiscreteCartPoleEnvironment, CartPoleRenderer
from dopamine.agents import FQIAgent
from dopamine.fapprox import *
from dopamine.experiments import Experiment
from dopamine.adapters import EpsilonGreedyExplorer, NormalizingAdapter, IndexingAdapter

from matplotlib import pyplot as plt
from numpy import *

# create agent, environment, renderer, experiment
agent = FQIAgent(faClass=RBF)
agent.iterations = 1
agent.presentations = 1
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
explorer = EpsilonGreedyExplorer(0.3, episodeCount=1000)
experiment.addAdapter(explorer)

experiment.setup()

# renderer = CartPoleRenderer()
# environment.renderer = renderer
# renderer.start()
    
# run experiment
for i in range(1000):
    valdata = experiment.evaluateEpisodes(10, visualize=True)
    mean_return = mean([sum(v.rewards) for v in valdata])
    print "exploration", explorer.epsilon
    print "mean return", mean_return
    print "mean ep. length", mean([len(e) for e in valdata])
    print "num episodes", len(agent.history)
    print "num total samples", agent.history.numTotalSamples()
    print

    experiment.runEpisodes(1)
    agent.learn()
    agent.history.truncate(50)

    

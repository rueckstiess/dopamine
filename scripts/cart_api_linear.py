from dopamine.environments import DiscreteCartPoleEnvironment, CartPoleRenderer
from dopamine.agents import APIAgent
from dopamine.fapprox import *
from dopamine.experiments import APIExperiment
from dopamine.adapters import EpsilonGreedyExplorer, NormalizingAdapter, IndexingAdapter

from matplotlib import pyplot as plt
from numpy import *

# create agent, environment, renderer, experiment
agent = APIAgent()
agent.iterations = 5

environment = DiscreteCartPoleEnvironment(maxSteps=200)
environment.conditions['actionNum'] = 2
environment.centerCart = False
experiment = APIExperiment(environment, agent)

# cut off last two state dimensions
indexer = IndexingAdapter([0, 1], None)
experiment.addAdapter(indexer)

# add normalization adapter
normalizer = NormalizingAdapter()
experiment.addAdapter(normalizer)

# add e-greedy exploration
explorer = EpsilonGreedyExplorer(0.3)
experiment.addAdapter(explorer)

experiment.setup()
explorer.epsilon = 0.3
explorer.decay = 0.9999

renderer = CartPoleRenderer()
# environment.renderer = renderer
# renderer.start()
        
# run experiment
for i in range(100):
    # environment.renderer = renderer
    valdata = experiment.evaluateEpisodes(50, visualize=True)
    # environment.renderer = None

    mean_return = mean([sum(v.rewards) for v in valdata])
    print "exploration", explorer.epsilon
    print "mean return", mean_return
    print "mean ep. length", mean([len(e) for e in valdata])
    print "num episodes", len(agent.history)
    print "num total samples", agent.history.numTotalSamples()
    print

    experiment.runEpisodes(50)
    agent.learn()

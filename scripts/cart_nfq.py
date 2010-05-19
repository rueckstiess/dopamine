from dopamine.environments import DiscreteCartPoleEnvironment, CartPoleRenderer
from dopamine.agents import NFQAgent
from dopamine.experiments import Experiment
from dopamine.adapters import EpsilonGreedyExplorer, NormalizingAdapter, IndexingAdapter

from matplotlib import pyplot as plt

# create agent, environment, renderer, experiment
agent = NFQAgent()
environment = DiscreteCartPoleEnvironment()
experiment = Experiment(environment, agent)

# cut off last two state dimensions
indexer = IndexingAdapter([0, 1], None)
experiment.addAdapter(indexer)

# add normalization adapter
normalizer = NormalizingAdapter()
experiment.addAdapter(normalizer)

# add e-greedy exploration
explorer = EpsilonGreedyExplorer(0.2, 1.0)
experiment.addAdapter(explorer)

for i in range(10):
    experiment.runEpisode(reset=True)

explorer.decay = 0.999
renderer = CartPoleRenderer()
environment.renderer = renderer
renderer.start()

# run experiment
for i in range(100):
    experiment.runEpisode(reset=True)
    agent.learn()
    print explorer.epsilon
    
    # print agent.history[0].rewards
    # agent.forget()

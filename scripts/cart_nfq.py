from rllib.environments import DiscreteCartPoleEnvironment, CartPoleRenderer
from rllib.agents import NFQAgent
from rllib.experiments import Experiment
from rllib.adapters import EpsilonGreedyExplorer, NormalizingAdapter

from matplotlib import pyplot as plt

# create agent, environment, renderer, experiment
agent = NFQAgent()
environment = DiscreteCartPoleEnvironment()
experiment = Experiment(environment, agent)

# add normalization adapter
normalizer = NormalizingAdapter()
experiment.addAdapter(normalizer)

# add e-greedy exploration
explorer = EpsilonGreedyExplorer(0.2, 0.9999)
experiment.addAdapter(explorer)


for i in range(20):
    experiment.runEpisode(reset=True)
agent.forget()

renderer = CartPoleRenderer()
environment.renderer = renderer
renderer.start()

# run experiment
for i in range(500):
    experiment.runEpisode(reset=True)
    for j in range(150):
        agent.learn()
    print agent.history[0].rewards
    agent.forget()

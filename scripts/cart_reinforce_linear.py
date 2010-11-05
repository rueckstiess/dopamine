from dopamine.environments import CartPoleEnvironment, CartPoleRenderer
from dopamine.agents import ReinforceAgent, LinearController
from dopamine.adapters import IndexingAdapter, NormalizingAdapter, GaussianExplorer, StateDependentExplorer
from dopamine.experiments import Experiment
from numpy import *

from dopamine.tools import Episode

environment = CartPoleEnvironment(maxSteps=400)
environment.centerCart = False

renderer = CartPoleRenderer()

agent = ReinforceAgent()
experiment = Experiment(environment, agent)

# cut off last two state dimensions
# indexer = IndexingAdapter([0, 1], None)
# experiment.addAdapter(indexer)

# add normalization adapter
normalizer = NormalizingAdapter(scaleActions=[(-50, 50)])
experiment.addAdapter(normalizer)

# add gaussian explorer
explorer = StateDependentExplorer(sigma=0.2)
experiment.addAdapter(explorer)
# explorer = GaussianExplorer(sigma=0.2)
# experiment.addAdapter(explorer)

# force setup here already to initiate pretraining
experiment.setup()
# environment.renderer = renderer
# renderer.start()

# run experiment
for i in range(5000):
    experiment.runEpisodes(20)
    
    # # split episodes into pieces
    # split = 40
    # for episode in agent.history:
    #     l = len(episode) // split
    #     if l < 1:
    #         continue
    #     for i in range(split):
    #         start = i*l
    #         end = min((i+1)*l, len(episode))
    #         states = episode.states[start:end,:]
    #         actions = episode.actions[start:end,:]
    #         rewards = episode.rewards[start:end]
    #         
    #         piece = Episode(experiment.conditions['stateDim'], experiment.conditions['actionDim'])
    #         piece.setArrays(states, actions, rewards)
    #         agent.history.appendEpisode(piece)
    
    agent.learn()
    agent.forget()

    valdata = experiment.evaluateEpisodes(10, visualize=True)
    environment.renderer = renderer
    experiment.evaluateEpisodes(1, visualize=False)
    environment.renderer = None
    
    print "mean return", mean([sum(v.rewards) for v in valdata])
    if mean([sum(v.rewards) for v in valdata]) > -300:
        if not renderer.isAlive():
            pass
            # renderer.start()
    print "avg. episode length", mean([len(v) for v in valdata])
    print "exploration variance", explorer.sigma
    
plt.show()
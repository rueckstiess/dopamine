from dopamine.environments import CartPoleEnvironment, CartPoleRenderer
from dopamine.agents import ReinforceAgent, ENACAgent
from dopamine.adapters import IndexingAdapter, NormalizingAdapter, GaussianExplorer, LinearSDExplorer, StateEnhancingAdapter
from dopamine.experiments import Experiment
from dopamine.fapprox import *
from numpy import *

from dopamine.tools import Episode

maxSteps = 400
environment = CartPoleEnvironment(maxSteps=maxSteps)
environment.centerCart = True

renderer = CartPoleRenderer()

agent = ENACAgent(faClass=Linear)
experiment = Experiment(environment, agent)

# cut off last two state dimensions
# indexer = IndexingAdapter([0, 1], None)
# experiment.addAdapter(indexer)

# enhance state
# def enhancef(state):
#     return r_[state, state[0]**2, abs(state[2]), sin(state[1]), cos(state[1]), 1]
# enhancer = StateEnhancingAdapter(enhancef)
# experiment.addAdapter(enhancer)

# add normalization adapter
normalizer = NormalizingAdapter(scaleActions=[(-50, 50)])
experiment.addAdapter(normalizer)

# add gaussian explorer
explorer = LinearSDExplorer(sigma=1.)
explorer.sigmaAdaptation = False
experiment.addAdapter(explorer)
explorer = GaussianExplorer(sigma=-1.)
explorer.sigmaAdaptation = False
experiment.addAdapter(explorer)

# force setup here already to initiate pretraining
experiment.setup()

# environment.renderer = renderer
# renderer.start()

# experiment.runEpisodes(4)

# run experiment
for i in range(5000):
    experiment.runEpisodes(10)    
    agent.learn()
    agent.history.keepBest(50)
    # agent.forget()

    valdata = experiment.evaluateEpisodes(10, visualize=False)

    if renderer.isAlive():
        environment.renderer = renderer
        experiment.evaluateEpisodes(1, visualize=False)
        environment.renderer = None
    
    print i
    print "mean return", mean([sum(v.rewards) for v in valdata])
    if mean([sum(v.rewards) for v in valdata]) > 1.75*maxSteps:
        if not renderer.isAlive():
            renderer.start()
            pass
            
    print "avg. episode length", mean([len(v) for v in valdata])
    print "exploration variance", explorer.sigma
    print
    
plt.show()
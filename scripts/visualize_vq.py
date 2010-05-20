from numpy import *
from dopamine.adapters import VQDiscretizationAdapter
from matplotlib import pyplot as plt

plt.ion()

vq = VQDiscretizationAdapter(50)

states = random.normal([0, 3], [3, 1], (500, 2))
states = r_[states, random.normal([-1, -2], [0.5, 2], (200, 2))]

plt.plot(states[:,0], states[:,1], '.')

for s in states:
    vq.applyState(s)

vq.sampleClusters()
vq.adaptClusters()

for i in range(len(vq.stateVectors)):
    plt.text(vq.stateVectors[i,0], vq.stateVectors[i,1], "%i"%i, bbox=dict(facecolor='green', alpha=0.5))

plt.ylim(-10, 10)
plt.xlim(-10, 10)
    
plt.show()
# demonstrate the logistic regression classifier on a dataset
# that has 8 classes, represented by point clouds at each corner
# of a 3d cube (input dim = 3). after starting, the script shows
# convergence to the correct classes after each round of training.

from dopamine.classifier import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def cubeData(nPointsPerClass=100, nRandomFeatures=3):
    points = np.random.random(size=(nPointsPerClass * 8, nRandomFeatures + 4))
 
    for x in range(2):
        for y in range(2):
            for z in range(2):
                cls = int(str(x)+str(y)+str(z), 2)
                # normally distributed point cloud around the corner
                points[cls*nPointsPerClass:(cls+1)*nPointsPerClass, :3] = np.random.normal([x, y, z], 0.1, [nPointsPerClass,3])
                # class value
                points[cls*nPointsPerClass:(cls+1)*nPointsPerClass, -1] = cls

    points = np.random.permutation(points)
    return points


data = cubeData(20, 0)
lr = LogisticRegression(3, 8)

colors = ['red', 'green', 'blue', 'yellow', 'black', 'cyan', 'magenta', 'gray']

plt.ion()

fig = plt.figure()

for i in range(25):
    print i
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')

    # visualize
    for d in data:
         pcls = lr.classify(d[:-1])
         ax.scatter([d[0]], [d[1]], [d[2]], color=colors[pcls])

    plt.gcf().canvas.draw()

    time.sleep(1)

    # learn
    for d in np.random.permutation(data):
        lr.update(d[:-1], d[-1])

plt.ioff()
plt.show()
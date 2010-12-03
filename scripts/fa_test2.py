import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from random import shuffle

from dopamine.fa import *

X_data, Y_data = np.meshgrid(np.arange(-1, 1, 0.3), np.arange(-1, 1, 0.3))
Z_data = np.sin(5*X_data) * np.cos(Y_data) + np.random.normal(0, 0.2, X_data.shape)

plt.ion()
# ax.plot_wireframe(X_data, Y_data, Z_data, cmap=plt.cm.jet, antialiased=True)

models = ['Linear', 'RBF', 'KNN', 'PyBrainNN', 'LWPRFA']

X_model, Y_model = np.meshgrid(np.arange(-1.1, 1.1, 0.05), np.arange(-1.1, 1.1, 0.05))

for mClass in models:
    reps = 1
    if mClass == 'LWPRFA':
        reps = 20
        
    # plot data points
    fig = plt.figure()
    ax = axes3d.Axes3D(fig)
    ax.scatter(X_data.flatten(), Y_data.flatten(), Z_data.flatten(), 'o')

    model = eval(mClass + '(2, 1)')
    
    # train model on data
    for _ in range(reps):
        data3 = zip(X_data.flatten(), Y_data.flatten(), Z_data.flatten())
        shuffle(data3)
        for x, y, z in data3:
            model.update(np.array([x, y]), np.array([z]))
    model.train()
    
    # plot results
    Z_model = np.array([model.predict(np.array([x,y]))[0] for x,y in zip(X_model.flatten(), Y_model.flatten())])
    Z_model = Z_model.reshape(X_model.shape)
    ax.plot_wireframe(X_model, Y_model, Z_model, cmap=plt.cm.jet, antialiased=True)
    plt.title(mClass)
    plt.gcf().canvas.draw()

plt.legend(loc='lower right')
plt.show()
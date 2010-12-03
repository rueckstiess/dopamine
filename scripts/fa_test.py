import numpy as np
import matplotlib.pyplot as plt

from dopamine.fapprox import *

X_data = np.arange(-1, 1, 0.05)
Y_data = [(x+1) * 3 * np.sin(8*x) + 4*x**2 + 10*x - 2 + np.random.normal(0, 1) for x in X_data]

# plot dataset set
plt.plot(X_data, Y_data, 'o')


models = ['Linear', 'RBF', 'KNN', 'LWPRFA', 'PyBrainNN']

X_model = np.arange(-1.1, 1.1, 0.01)

for mClass in models:
    model = eval(mClass + '(1, 1)')
    
    # train model on data
    for x, y in zip(X_data, Y_data):
        model.update(x, y)
    model.train()
    
    # plot results
    Y_model = [model.predict(x)[0] for x in X_model]
    plt.plot(X_model, Y_model, label=mClass)

plt.legend(loc='lower right')
plt.show()
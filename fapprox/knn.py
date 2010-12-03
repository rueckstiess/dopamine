from dopamine.fapprox.fa import FA
import numpy as np
from operator import itemgetter
from bisect import insort
import matplotlib.pyplot as plt

class KNN(FA):
    
    tau = 20.
    neighborhood = 3
    parametric = False
    
    def train(self):
        pass
            
    def _getNeighbors(self, point, n):
        order = []
        point = np.asarray(point)
        for i,p in enumerate(self.dataset.inputs):
            insort(order, (np.linalg.norm(p-point),i) )
        return map(itemgetter(1), order[:n]), map(itemgetter(0), order[:n])            
    
    def predict(self, inp):
        if len(self.dataset) < self.neighborhood:
            return np.array([0.])
        n = min(self.neighborhood, len(self.dataset))
        indices, distances = self._getNeighbors(inp, self.neighborhood)
        value = np.sum([self.dataset.targets[i] * np.exp(-self.tau*d) for i,d in zip(indices,distances)]) / sum([np.exp(-self.tau*d) for d in distances])
        return np.array([value])
    
        

if __name__ == '__main__':
    k = KNN(1, 1)
    for x in np.arange(-1., 1., 0.1):
        k.update([x], np.random.uniform(-10, 10))
    
    for i in range(len(k.dataset)):
        plt.plot(k.dataset.inputs[i], k.dataset.targets[i], 'ks')
    
    x = np.arange(-1., 1., 0.01)
    y = [k.predict([x_]) for x_ in x]
    
    plt.plot(x, y, 'r.')
    plt.show()
    

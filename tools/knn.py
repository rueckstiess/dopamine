import numpy as np
from operator import itemgetter
from bisect import insort

import matplotlib.pyplot as plt

class KNN(object):
    
    def __init__(self):
        self.reset()
        self.tau = 10.
        
    def reset(self):
        self.points = []
        self.values = []
        
    def addPoint(self, point, value):
        self.points.append(np.asarray(point))
        self.values.append(value)
    
    def getNeighbors(self, point, n):
        order = []
        point = np.asarray(point)
        for i,p in enumerate(self.points):
            insort(order, (np.linalg.norm(p-point),i) )
        return map(itemgetter(1), order[:n]), map(itemgetter(0), order[:n])            
    
    def predict(self, point, n=4):
        if len(self.points) < n:
            return 0.
        n = min(n, len(self.points))
        indices, distances = self.getNeighbors(point, n)
        value = np.sum([self.values[i] * np.exp(-self.tau*d) for i,d in zip(indices,distances)]) / sum([np.exp(-self.tau*d) for d in distances])
        return value

if __name__ == '__main__':
    k = KNN()
    for i in range(20):
        k.addPoint([i], np.random.random())
    
    for i,p in enumerate(k.points):
        plt.plot(i, k.values[i], 'ks')
    
    
    x = np.arange(0, 20, 0.01)
    y = [k.predict([x_]) for x_ in x]
    
    plt.plot(x, y, 'r.')
    
    
    # plt.plot([4], [4], 'rs')
    # 
    # indices, dists = k.getNeighbors([4, 4], 5)
    # for n in k.getNeighbors([4, 4], 5):
    #     plt.plot(k.points[n][0], k.points[n][1], 'rx')
    
    plt.show()
    

from numpy import zeros, where
from dopamine.tools.history import History

def abstractMethod():
    """ This should be called when an abstract method is called that should have been 
    implemented by a subclass. It should not be called in situations where no implementation
    (i.e. a 'pass' behavior) is acceptable. """
    raise NotImplementedError('Method not implemented!')


def one_to_n(val, maxval):
    """ Returns a 1-in-n binary encoding of a non-negative integer. """
    a = zeros(int(maxval), float)
    a[int(val)] = 1.
    return a


def n_to_one(arr):
    """ Returns the reverse of a 1-in-n binary encoding. """
    return where(arr == 1)[0][0]
    

def symmetryData(data):
    symmetryData = History(data.stateDim, data.actionDim)
    
    for e in data:
        symmetryData.appendEpisode(e)
    
    # for e in data:
    #     for s, a, r, _ in e:
    #         symmetryData.append(-s, -a, r)
    #     symmetryData.newEpisode()
        
    return symmetryData
  
def reduceData(data, buffer=4):
    """ goes through dataset and only stores the "interesting"
        samples, the ones where a change in reward has happened.
    """
    reducedHistory = History(data.stateDim, data.actionDim)
    transitions = []
    
    for episode in data:
        rdiff = episode.rewards[1:] - episode.rewards[:-1]
        transitions = set([0, len(episode.rewards)-1])
        transitions.update(where(rdiff != 0)[0])
        buf = set()
        for t in transitions:
            for i in range(max(0, t-buffer), min(t+2+buffer, len(episode.rewards))):
                buf.add(i)

        for b in buf:
            reducedHistory.append(episode.states[b,:], episode.actions[b,:], episode.rewards[b])
        
        reducedHistory.newEpisode()
    
    return reducedHistory
    
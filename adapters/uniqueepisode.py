from dopamine.adapters import Adapter
from dopamine.tools import Episode
import numpy as np
from collections import OrderedDict

class UniqueEpisodeAdapter(Adapter):
    """ This adapter ensures that at the end of each episode, the agent's
        history is altered in a way that only unique (s, a, ns) transitions 
        exist, removing all duplicates of such tuples. The reward for the
        remaining tuples becomes the mean of all rewards for that unique 
        tuple, including the deleted ones.

        To achieve this, the adapter hooks into applyEpisodeFinished and
        modifies the agent's history by deleting the duplicates for the 
        last episode.

        This adapter assumes discrete states and actions.
        
        Example:
            assuming the agent's last episode would consist of the following
            (s, a, r, ns) tuples (note the last ns equals None as the episode
            reached the end, as implemented in Episode's __iter__ method):

            ([0], [1], -0.1, [0])
            ([0], [1], -0.5, [0])
            ([0], [1],  0.2, [2])
            ([2], [0],  1.0, [3])
            ([3], [0],  0.0, [3])
            ([3], [0], -0.4, [3])
            ([3], [1], -0.1, [3])
            ([3], [1], -0.1, [3])
            ([3], [1], -1.0, None)

            Then the episode would be altered after episodeFinished was emitted 
            to the following reduced version:

            ([0], [1], -0.3, [0])
            ([0], [1],  0.2, [2])
            ([2], [0],  1.0, [3])
            ([3], [0], -0.2, [3])
            ([3], [1], -0.1, [3])
            ([3], [1], -1.0, None)
    """
    
    inConditions = {
      'discreteStates':True,
      'stateDim':1,
      'discreteActions':True,
      'actionDim':1,
      'episodic':True
    }
        
    def stripEpisode(self, episode):

        sansDict = OrderedDict()

        for s, a, r, ns in episode:
            s = s.item()
            a = a.item()
            if ns != None:
                ns = ns.item()

            if (s, a, ns) not in sansDict:
                sansDict.setdefault((s, a, ns), []).append(r)
            
        strippedEpisode = Episode(episode.stateDim, episode.actionDim)
        for (s, a, ns) in sansDict:
            r = np.mean(sansDict[(s, a, ns)])
            strippedEpisode.append(s, a, r, ns)
        
        return strippedEpisode


    def applyEpisodeFinished(self, episodeFinished):
        if not episodeFinished:
            return episodeFinished

        episode = self.stripEpisode(self.experiment.agent.history.pop())
        self.experiment.agent.history.appendEpisode(episode)

        return episodeFinished
    
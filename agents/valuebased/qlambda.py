from dopamine.agents.valuebased.q import QAgent

class QLambdaAgent(QAgent):
    """ QLambdaAgent is a variation of QAgent that uses an eligibility trace
        to speed up learning. """
    
    alpha = 0.5
    gamma = 0.9
    qlambda = 0.5

    # only back up as much as trace_limit steps (max length of eligibility trace)
    trace_limit = 10

    def learn(self):
        """ go through whole episode and make Q-value updates. """
        for episode in self.history:
            for t, (state, action, reward, nextstate) in enumerate(episode.reversedSamples()):
                t = len(episode)-1-t

                state = int(state)
                action = int(action)
     
                if nextstate != None:
                    nextstate = int(nextstate)
                    maxnext = self.estimator.getValue(nextstate, self.estimator.getBestAction(nextstate))
                else:
                    maxnext = 0.

                qvalue = self.estimator.getValue(state, action)
                delta = self.alpha * (reward + self.gamma * maxnext - qvalue)
                lbda = 1.

                i = 0
                for t_ in xrange(t, -1, -1):
                    if i >= self.trace_limit and not self.trace_limit == None: 
                        break
                    state_ = episode.states[t_]
                    action_ = episode.actions[t_]
                    qvalue_ = self.estimator.getValue(state_, action_)
                    self.estimator.updateValue(state_, action_, qvalue_ + delta * lbda)
                    lbda *= self.qlambda
                    i += 1


    
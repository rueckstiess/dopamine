from matplotlib.mlab import rk4 
from numpy import sin, cos
import time
from numpy import eye, matrix, random, asarray, clip

from rllib.environments.environment import Environment


class CartPoleEnvironment(Environment):
    """ This environment implements the cart pole balancing benchmark, as stated in:
        Riedmiller, Peters, Schaal: "Evaluation of Policy Gradient Methods and
        Variants on the Cart-Pole Benchmark". ADPRL 2007.
        It implements a set of differential equations, solved with a 4th order
        Runge-Kutta method.
    """       
    
    # define state and action dimensions
    stateDim = 4
    actionDim = 1 
        
    # define if states and/or actions are discrete (rather than continuous)
    discreteStates = False
    discreteActions = False
    
    # define if environment has episodes (True) or is life-long (False)
    episodic = True
    
    # some physical constants
    g = 9.81
    l = 0.5
    mp = 0.1
    mc = 1.0
    dt = 0.02   
    
    def __init__(self, maxSteps=500):
        
        # initialize the environment (randomly)
        self.reset()
        self.action = 0.0
        self.maxSteps = maxSteps

    def getState(self):
        """ returns the state one step (dt) ahead in the future. stores the state in
            self.sensors because it is needed for the next calculation. The sensor return 
            vector has 4 elements: theta, theta', s, s' (s being the distance from the
            origin).
        """
        Environment.getState(self)
        return asarray(self.sensors).flatten()
                            
    def performAction(self, action):
        """ stores the desired action for the next runge-kutta step. Actions are expected
            to be between -50 and 50.
        """
        action = clip(asarray(action), -50., 50.)
        Environment.performAction(self, action)
    
    def _update(self):
        self.sensors = rk4(self._derivs, self.sensors, [0, self.dt])
        self.sensors = self.sensors[-1]
        if self.renderer:
            self.renderer.updateData(self.sensors)
        
                        
    def reset(self):
        """ re-initializes the environment, setting the cart back in a random position.
        """
        Environment.reset(self)
        angle = random.uniform(-0.2, 0.2)
        pos = random.uniform(-0.5, 0.5)
        self.sensors = (angle, 0.0, pos, 0.0)
        
    def episodeFinished(self):
        return self.timestep >= self.maxSteps    
        
    def getReward(self):
        Environment.getReward(self)

        angle = abs(self.sensors[0])
        s = abs(self.sensors[2])
        reward = 0
        if angle < 0.05 and s < 0.05:
            reward = 0.
        elif angle > 0.7 or s > 2.4:
            reward = -2.
        else: 
            reward = -1.
        return reward
        
    def _derivs(self, x, t): 
        """ This function is needed for the Runge-Kutta integration approximation method. It calculates the 
            derivatives of the state variables given in x. for each variable in x, it returns the first order
            derivative at time t.
        """
        F = self.action
        (theta, theta_, _s, s_) = x
        u = theta_
        sin_theta = sin(theta)
        cos_theta = cos(theta)
        mp = self.mp
        mc = self.mc
        l = self.l
        u_ = (self.g * sin_theta * (mc + mp) - (F + mp * l * theta ** 2 * sin_theta) * cos_theta) / (4 / 3 * l * (mc + mp) - mp * l * cos_theta ** 2)
        v = s_
        v_ = (F - mp * l * (u_ * cos_theta - (s_ ** 2 * sin_theta))) / (mc + mp)     
        return (u, u_, v, v_)   
    

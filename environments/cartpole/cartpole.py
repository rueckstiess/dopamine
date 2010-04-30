from matplotlib.mlab import rk4 
from math import sin, cos
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
    
    # some physical constants
    g = 9.81
    l = 0.5
    mp = 0.1
    mc = 1.0
    dt = 0.02    
    
    def __init__(self):
        self.stateDim = 4
        self.actionDim = 1
        
        # initialize the environment (randomly)
        self.reset()
        self.action = 0.0
        self.delay = False

    def getState(self):
        """ returns the state one step (dt) ahead in the future. stores the state in
            self.sensors because it is needed for the next calculation. The sensor return 
            vector has 4 elements: theta, theta', s, s' (s being the distance from the
            origin).
        """
        Environment.getState(self)
        return asarray(self.sensors)
                            
    def performAction(self, action):
        """ stores the desired action for the next runge-kutta step. Actions are expected
            to be between -50 and 50.
        """
        action = clip(asarray(action), -50., 50.)
        Environment.performAction(self, action)
    
    def _update(self):
        self.sensors = rk4(self._derivs, self.sensors, [0, self.dt])
        self.sensors = self.sensors[-1]
                        
    def reset(self):
        """ re-initializes the environment, setting the cart back in a random position.
        """
        Environment.reset(self)
        angle = random.uniform(-0.2, 0.2)
        pos = random.uniform(-0.5, 0.5)
        self.sensors = (angle, 0.0, pos, 0.0)
        
        
    def getReward(self):
        Environment.getReward(self)
        # TODO integrate reward from task in environment
        
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
    

import numpy as np

class EpsPolicy_exp:
    '''
    low - minimal epsilon (t -> \infty)
    high - maximal epsilon (t -> 0)
    gamma - decay rate
    '''
    
    def __init__(self, high, gamma, low = 0):
        self.high = high
        self.gamma = gamma
        self.low = low
        
    def __call__(self, n):
        return self.low + (self.high - self.low) * np.exp(-self.gamma * n)
    
class EpsPolicy_pow:
    '''
    low - minimal epsilon (t -> \infty)
    high - maximal epsilon (t -> 0)
    p - power
    '''
    
    def __init__(self, high, p, low = 0):
        self.high = high
        self.p = p
        self.low = low
        
    def __call__(self, n):
        return self.low + (self.high - self.low) / (n + 1) ** self.p
        
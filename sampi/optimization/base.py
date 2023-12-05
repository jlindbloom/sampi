import numpy as np



class ObjectiveTerm:
    """Represents a term in an objective function.
    n: length of expected input vector.
    """
    def __init__(self, n, device="cpu"):
        
        self.device = device
        self.n = n
        
    def evaluate(self, x):
        raise NotImplementedError

        
        
class SmoothObjectiveTerm(ObjectiveTerm):
    """Represents a smooth term in an objective function.
    """
    def __init__(self, n, device="cpu"):
            
        super().__init__(n, device)
        

        
class ConvexNonsmoothObjectiveTerm(ObjectiveTerm):
    """Represents a convex, nonsmooth term in an objective function.
    """
    
    def __init__(self, n, device="cpu"):
            
        super().__init__(n, device)
    
    def evaluate_prox(self, x, lam=None, rho=None):
        raise NotImplementedError
    
        
        
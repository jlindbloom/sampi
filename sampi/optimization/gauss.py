import numpy as np

import jlinops

from .base import SmoothObjectiveTerm


        
class GaussianDataLikelihood(SmoothObjectiveTerm):
    """Represents a Gaussian data fidelity term in an objective function. 
    Looks like: 
    
    E(x) = (1/(2*v))*|| A x - b ||_2^2
    
    A: a LinearOperator.
    b: a shift vector.
    """
    
    def __init__(self, A, b, v=1.0, est_smoothness=True):
        
        n = A.shape[1]
        device = A.device
        super().__init__(n, device)
        
        self.A = A
        self.b = b
        self.v = 1.0
        
        # Estimate L s.t. this factor is L-smooth.
        if est_smoothness:
            # L = 2-norm of the matrix (1/v)*self.A.T @ self.A
            eigvals, _ = jlinops.eigsh( (1.0/v)*(self.A.T @ self.A), k=1, which="LM")
            max_eigval = eigvals[0]
            self.L = 1.05*max_eigval # increase by small safe-guard factor
        else:
            self.L = None
        
    def evaluate(self, x):
        xp = jlinops.get_module(x)
        fac = (1.0/(2*self.v))
        result = fac*xp.linalg.norm(self.A.matvec(x) - self.b)**2
        return result
        
    def evaluate_grad(self, x):
        return (1/self.v)*self.A.rmatvec(self.A.matvec(x) - self.b)
        
        
        
        
        
        
        
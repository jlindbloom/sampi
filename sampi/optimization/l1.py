
import numpy
import jlinops


from .base import ConvexNonsmoothObjectiveTerm




class L1NormPenalty(ConvexNonsmoothObjectiveTerm):
    """Represents a L1-norm penalty. Looks like
    E(x) = reg_lam*|| x ||_1
    """
    def __init__(self, n, reg_lam=1.0, device="cpu"):
        
        self.reg_lam = reg_lam
        super().__init__(n, device)
        
    def evaluate(self, x):
        xp = jlinops.get_module(x)
        return self.reg_lam*xp.abs( x ).sum()
        
    def evaluate_prox(self, x, lam=None, rho=None, *args, **kwargs):
                           
        return jlinops.prox_l1_norm(x, lam=self.reg_lam*lam)
        
        

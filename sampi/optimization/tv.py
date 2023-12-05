import numpy
import jlinops

from .base import ConvexNonsmoothObjectiveTerm


    
class TV1DPenalty(ConvexNonsmoothObjectiveTerm):
    """Represents a 1-norm 1D-TV penalty where the TV operator is a 1D
    discrete gradient operator. Looks like
    E(x) = reg_lam* || R x ||_1.
    """

    def __init__(self, n, reg_lam=1.0, boundary="none", device="cpu"):

        self.boundary = boundary
        self.reg_lam = reg_lam

        # Build proximal operator object
        self._prox_solver = jlinops.ProxTV1DNormOperator(n, boundary=boundary, device=device, D=None)
        self.D = self._prox_solver.D

        super().__init__(n, device)
    
    def evaluate(self, x):
        xp = jlinops.get_module(x)
        return self.reg_lam*xp.abs( self.D.matvec(x) ).sum()
    
    def evaluate_prox(self, x, lam=None, rho=None, *args, **kwargs):
        return self._prox_solver.apply(x, lam=self.reg_lam*lam, rho=rho, *args, **kwargs)
    

class TVNeumann2DPenalty(ConvexNonsmoothObjectiveTerm):
    """Represents a 1-norm 2D-TV penalty where the TV operator is a 2D
    discrete gradient operator with Neumann boundary conditions. Looks like
    E(x) = reg_lam*|| R x ||_1
    """
    def __init__(self, grid_shape, reg_lam=1.0, device="cpu"):
        
        self.grid_shape = grid_shape
        self.reg_lam = reg_lam
        
        # Build the proximal operator object
        self._prox_solver = jlinops.ProxTVNeumann2DNormOperator( grid_shape )
        n = self._prox_solver.D.shape[1]
        self.D = self._prox_solver.D
        
        super().__init__(n, device)
        
    def evaluate(self, x):
        xp = jlinops.get_module(x)
        return self.reg_lam*xp.abs( self.D.matvec(x) ).sum()
        
    def evaluate_prox(self, x, lam=None, rho=None, *args, **kwargs):
        
        return self._prox_solver.apply(x, lam=self.reg_lam*lam, rho=rho, *args, **kwargs)
        













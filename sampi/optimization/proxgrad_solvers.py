import numpy as np

from fastprogress import progress_bar

from .objectives import CompositeModelObjectiveFunction


class ProximalGradientSolver:
    """Represents a proximal gradient solver.
    
    objective: must be a CompositeModelObjectiveFunction
    """
    
    def __init__(self, objective):
        
        # Check
        assert isinstance(objective, CompositeModelObjectiveFunction)
        self.objective = objective
        self.f, self.g = self.objective.terms   
        
        
    def solve(self, iterations=20, L=None, initialization=None):
        
        if initialization is None:
            x = np.zeros(self.objective.n)
        else:
            x = initialization
            
        if L is None:
            assert self.f.L is not None, "must provide stepsize, can't pick default."
            L = self.f.L
        
        obj_vals = []
        for i in progress_bar(range(iterations)):
            
            grad_step = x - (1.0/L)*self.f.evaluate_grad(x)
            x = self.g.evaluate_prox(grad_step, lam=1.0/L)
            
            # Track value
            obj_vals.append(self.objective.evaluate(x))
            
        data = {
            "result": x,
            "obj_vals": np.asarray(obj_vals),
        }
              
        return data
        
    
    def line_search(self):
        """Line search method.
        """
        raise NotImplementedError
        

    
class FISTASolver:
    """Represents a FISTA solver (Fast proximal gradient method).
    
    objective: must be a CompositeModelObjectiveFunction
    """
    
    def __init__(self, objective):
        
        # Check
        assert isinstance(objective, CompositeModelObjectiveFunction)
        self.objective = objective
        self.f, self.g = self.objective.terms   
        
        
    def solve(self, iterations=20, L=None, initialization=None):
        
        if initialization is None:
            x_prev = np.zeros(self.objective.n)
        else:
            x_prev = initialization
            
        if L is None:
            assert self.f.L is not None, "must provide stepsize, can't pick default."
            L = self.f.L
            
        y = x_prev.copy()
        t_prev = 1.0
        
        obj_vals = []
        for i in progress_bar(range(iterations)):
            
            grad_step = y - (1.0/L)*self.f.evaluate_grad(y)
            x_curr = self.g.evaluate_prox(grad_step, lam=1.0/L)
            t_curr = 0.5*(1 + np.sqrt(1 + 4*(t_prev**2)))
            y = x_curr + ((t_prev - 1.0)/t_curr)*( x_curr - x_prev )
            
            # Advance
            t_prev = t_curr
            x_prev = x_curr
            
            # Track value
            obj_vals.append(self.objective.evaluate(x_curr))
            
        data = {
            "result": x_curr,
            "obj_vals": np.asarray(obj_vals),
        }
              
        return data
        
    
    def line_search(self):
        """Line search method.
        """
        raise NotImplementedError
        
    

class MFISTASolver:
    """Represents a MFISTA solver (Monotone FISTA).
    
    objective: must be a CompositeModelObjectiveFunction
    """
    
    def __init__(self, objective):
        
        # Check
        assert isinstance(objective, CompositeModelObjectiveFunction)
        self.objective = objective
        self.f, self.g = self.objective.terms   
        
        
    def solve(self, iterations=20, L=None, initialization=None):
        
        if initialization is None:
            x_prev = np.zeros(self.objective.n)
        else:
            x_prev = initialization
            
        if L is None:
            assert self.f.L is not None, "must provide stepsize, can't pick default."
            L = self.f.L
            
        y = x_prev.copy()
        t_prev = 1.0
        
        obj_vals = []
        for i in progress_bar(range(iterations)):
            
            grad_step = y - (1.0/L)*self.f.evaluate_grad(y)
            z = self.g.evaluate_prox(grad_step, lam=1.0/L)
            
            F_z = self.objective.evaluate(z)
            F_x = self.objective.evaluate(x_prev)
            if F_z < F_x:
                x_curr = z
            else:
                x_curr = x_prev
       
            t_curr = 0.5*(1 + np.sqrt(1 + 4*(t_prev**2)))
            y = x_curr + (t_prev/t_curr)*(z - x_curr) + ((t_prev - 1.0)/t_curr)*( x_curr - x_prev )
            
            # Advance
            t_prev = t_curr
            x_prev = x_curr
            
            # Track value
            obj_vals.append(self.objective.evaluate(x_curr))
            
        data = {
            "result": x_curr,
            "obj_vals": np.asarray(obj_vals),
        }
              
        return data
        
    
    def line_search(self):
        """Line search method.
        """
        raise NotImplementedError
        
    


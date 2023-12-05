import numpy as np

from .base import SmoothObjectiveTerm, ConvexNonsmoothObjectiveTerm


class ObjectiveFunction:
    """Base class for representing an objective function.
    
    terms: a list of 
    """
    
    def __init__(self, terms):
        
        self.terms = terms
        device = self.terms[0].device
        n = self.terms[0].n
        for term in self.terms[1:]:
            assert term.device == device, "devices not compatible."
            assert term.n == n, "expected inputs not compatible."
            
        self.device = device
        self.n = n
            
    def evaluate(self, x):
        result = 0.0
        for term in self.terms:
            result += term.evaluate(x)
        return result
    
    
    
class CompositeModelObjectiveFunction(ObjectiveFunction):
    """Represents an objective function conforming to the composite model, i.e.,
    F(x) = f(x) + g(x) where
    f(x) is proper, closed, dom(f) is convex, dom(g) \subseteq int(dom(f)), and L_f smooth, and
    g(x) is proper closed and convex.
    
    The gradient of f must be defined.
    The proximal operator of g must be defined.
    """
    def __init__(self, f, g):
        
        terms = [f, g]
        super().__init__(terms)
        
        # Check valid
        assert isinstance(f, SmoothObjectiveTerm), "f must be a SmoothObjectiveTerm."
        assert isinstance(g, ConvexNonsmoothObjectiveTerm), "g must be a ConvexNonsmoothObjectiveTerm."
        
        
    
    
    
    
    
    
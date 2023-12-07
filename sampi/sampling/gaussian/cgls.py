import numpy as np
import matplotlib.pyplot as plt

from fastprogress import progress_bar

from runningstatistics import StatsTracker
import jlinops 
import sampi



class CGLSGaussianSampler:
    """Represents a Gaussian sampler for Gaussians of the form
    -log \pi(x) = \sum_i^K || L_i x - s_i ||_2^2 + C.
    
    """
    
    def __init__(self, factors, shifts):
        
        self.factors = factors
        self.shifts = shifts
            
        # Checks
        self.n = self.factors[0].shape[1]
        for factor in self.factors[1:]:
            assert factor.shape[1] == self.n, "incompatible shapes for factors."
        
        # Set None shifts to zeros
        for j in range(len(self.shifts)):
            if self.shifts[j] is None:
                self.shifts[j] = np.zeros(self.factors[j].shape[0])
                
        # Assemble matrix
        self.A = jlinops.StackedOperator(self.factors)
        self.m = self.A.shape[0]
        
        # Assemble deterministic part of rhs
        self.rhs_det = np.hstack(self.shifts)
        
        
    def sample(self, n_samples=100, warmstarting=False, observables=None, *args, **kwargs):
        
        # Instantiate tracker
        tracker = StatsTracker((self.n,))
        
        # Setup
        init = None
        n_cg_its_per_sample = []
        if observables is None:
            pass
        else:
            n_observables = len(observables)
            obs_trackers = []
            for i, observable in enumerate(observables):
                # Instantiate tracker for each observable
                tmp = observable(np.ones(self.n)) # figure out array shape of the output
                if np.isscalar(tmp):
                    obs_tracker = StatsTracker((1,))
                    obs_trackers.append(obs_tracker)
                else:
                    obs_tracker = StatsTracker(tmp.shape)
                    obs_trackers.append(obs_tracker)
        
        # Generate samples
        for j in progress_bar(range(n_samples)):
            
            # Generate random part of rhs
            rhs_rand = np.random.normal(size=self.m)
            
            # Sum together
            rhs = self.rhs_det + rhs_rand
            
            # Solve the random least-squares problem
            cgls_solve = jlinops.cgls(self.A, rhs, x0=init, *args, **kwargs)
            sample = cgls_solve["x"]
            n_cg_its_per_sample.append( cgls_solve["n_iters"] )
            if warmstarting: 
                init = sample
            tracker.push(sample)
            
            # Handle any observables
            if observables is not None:
                for i in range(n_observables):
                    obs_trackers[i].push(sample)
            
        data = {
            "mean": tracker.mean(),
            "stdev": tracker.stdev(),
            "var": tracker.variance(),
            "n_cg_its_per_sample": np.asarray(n_cg_its_per_sample),
            "tracker": tracker,
            "obs_trackers": None,
        }
                                        
        if observables is not None:
            data["obs_trackers"] = obs_trackers
        
        return data

import numpy as np
import matplotlib.pyplot as plt

from fastprogress import progress_bar

from runningstatistics import StatsTracker
import jlinops 
import sampi



class CGLSGaussianSampler:
    """Represents a Gaussian sampler for Gaussians of the form
    -log \pi(x) = \sum_i^K (1/2)|| L_i x - s_i ||_2^2 + C.
    
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


    

class PriorconditionedIASGaussianSampler:
    """Represents a Gaussian sampler for Gaussians of the form
    -log \pi(x) = (1 / 2 v) || F x - y ||_2^2 + (1/2) || R x ||_2^2.
    Compatible with the format for the IAS solver.
    """
    
    def __init__(self, F, R, y, theta, pdata={}, noise_var=1.0):
        
        self.F = F
        self.R = R
        self.noise_var = noise_var
        self.theta = theta
        self.y = y
           
        # Checks
        self.n = self.F.shape[1]
        assert self.R.shape[1] == self.n, "incompatible shapes for factors."
        
        # Assemble deterministic part of rhs
        zeros = np.zeros(self.R.shape[0])
        self.rhs_det = np.hstack([y, zeros])
        
        # Handle pdata
        self._pdata = pdata
        self.pdata = {
            "W": None,
            "Rpinv_factory": None,
            "Rinv": None,
            "FWpinv": None,
        }
        for key in self._pdata.keys():
            if key in ["W", "Rpinv_factory", "Rinv", "FWpinv"]:
                self.pdata[key] = self._pdata[key]
            else:
                raise ValueError
                
        if self.pdata["W"] is not None:
            self.pdata["FW_pinv"] = jlinops.QRPinvOperator( jlinops.MatrixLinearOperator(self.F.matmat(self.pdata["W"].A)) )
        
        
    def sample(self, n_samples=100, priorconditioning=True, warmstarting=False, observables=None, *args, **kwargs):
        
        # Whiten forward operator and data
        noise_stdev = np.sqrt(self.noise_var)
        Ftilde = (1.0/noise_stdev)*self.F.T
        ytilde = (1.0/noise_stdev)*self.y
        
        # Build Rtilde
        Rtilde = jlinops.DiagonalOperator(1.0/np.sqrt(self.theta)) @ self.R
        
        if priorconditioning:
            # Build current Rpinv operator
            Rpinv = self.pdata["Rpinv_factory"](self.theta)
        
        # Instantiate tracker
        tracker = StatsTracker((self.n,))
        
        # Setup
        x_prev = None
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
                
            ytilde_rand = ytilde + np.random.normal(size=ytilde.shape)
            shift = np.random.normal(size=Rtilde.shape[0])
            
            # Solve the random least-squares problem
            if not priorconditioning:
                ytilde_rand = ytilde + np.random.normal(size=ytilde.shape)
                shift = np.random.normal(size=Rtilde.shape[0])
                cgls_solve = jlinops.trlstsq(Ftilde, Rtilde, ytilde_rand, lam=1.0, shift=shift, initialization=x_prev, *args, **kwargs)
            else:
                # Solve using transformed CGLS
                cgls_solve = jlinops.trlstsq_standard_form(Ftilde, ytilde, Rpinv=Rpinv, R=Rtilde,
                                                           AWpinv=self.pdata["FWpinv"], lam=1.0, shift=shift, W=self.pdata["W"], initialization=x_prev, *args, **kwargs)
                
            sample = cgls_solve["x"]
            if warmstarting:
                x_prev = sample.copy()
            n_cg_its_per_sample.append( cgls_solve["n_iters"] )
            if warmstarting: 
                init = -sample
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

    
    
    
    
    
    
    
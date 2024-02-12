import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import recipinvgauss
import scipy.sparse as sps
from fastprogress import progress_bar

from runningstatistics import StatsTracker
import jlinops



class BayesianLASSOGibbsSampler:
    """Implements the Bayesian LASSO hierarchical sampler for the L1 problem.
    """

    def __init__(self, F, R, y, noise_var=1.0, reg_lambda=1.0):

        self.F = F
        self.R = R
        self.y = y
        self.noise_var = noise_var
        self.reg_lambda = reg_lambda


    def sample(self, n_samples, x0=None, n_burn=0, theta_tol=1e-2):
        """Runs the Gibbs sampler.
        """

        # Initialize
        if x0 is None:
            x = np.zeros(self.F.shape[1])
        else:
            x = x0

        # Create trackers
        x_tracker = StatsTracker(self.F.shape[1])
        theta_tracker = StatsTracker(self.R.shape[0])

        # Run the sampler
        for j in progress_bar(range(n_samples+n_burn)):

            # Update theta
            theta = self.sample_theta(x, tol=theta_tol)

            # Update x
            x = self.sample_x(theta)

            # Push to tracker
            if j >= n_burn:
                x_tracker.push(x)
                theta_tracker.push(theta)


        results = {
            "x_tracker": x_tracker,
            "theta_tracker": theta_tracker,
        }

        return results


    def sample_x(self, theta):
        """Given local variances theta, draws a sample for x.
        """

        Q = (1.0/self.noise_var)*(self.F.A.T @ self.F.A) + (1/2)*(self.R.A.T @ ( sps.diags(1.0/theta) @ self.R.A ) )

        # # Bad way
        # Qinv = np.linalg.inv(Q.toarray())
        # mean = Qinv @ ((1.0/self.noise_var)*self.F.T @ self.y )
        # sample = np.random.multivariate_normal(mean, Qinv)

        # Good way
        Q = sps.csc_matrix(Q)
        Q = jlinops.MatrixLinearOperator(Q)
        Linv = jlinops.BandedCholeskyFactorInvOperator(Q)
        mean = Linv.T @ (Linv @ ((1.0/self.noise_var)*self.F.T @ self.y ) )
        sample = mean + ( Linv.T @ np.random.normal(size=Q.shape[0]) )

        return sample
    

    def sample_theta(self, x, tol=1e-2):
        """Given x, draws a sample for the thetas.
        """

        # Get Rx
        Rx = self.R @ x

        # Make output array
        sample = np.zeros(self.R.shape[0])


        # Need to check where Rx is close to zero, so we can sample from exponential there instead
        idx_too_small = np.where(np.abs(Rx) < tol)
        idx_fine = np.where(np.abs(Rx) >= tol)

        # Break into two parts
        Rx_too_small = Rx[idx_too_small]
        Rx_fine = Rx[idx_fine]

        # For the components near zero, sample from the exponential
        theta_from_too_small = np.random.exponential(scale=1.0/self.reg_lambda, size=len(Rx_too_small))

        # For the components not near zero, sample from the inverse Gaussian
        theta_from_fine = recipinvgauss.rvs(mu=1.0/(self.reg_lambda*np.abs(Rx_fine)), scale=1.0/(self.reg_lambda**2))

        # Put all into one array
        sample[idx_too_small] = theta_from_too_small
        sample[idx_fine] = theta_from_fine

        assert np.all(sample > 0), "some thetas are no positive!"

        return sample



















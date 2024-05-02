import numpy as np
import scipy.stats as st

class Poisson:
    def __init__(self, k, theta):
        self.k0 = self.k = np.array([k])
        self.theta0 = self.theta = np.array([theta])

    def pdf(self, data):
        return st.nbinom.pmf(data, self.k, 1/(1+self.theta))

    def update_parameters(self, data):
        kT0 = np.concatenate((self.k0, self.k+data))
        thetaT0 = np.concatenate((self.theta0, self.theta/(1+self.theta)))

        self.k = kT0
        self.theta = thetaT0

    def prune(self, gap_size):
        self.mu = self.k[:gap_size + 1]
        self.kappa = self.theta[:gap_size + 1]


class GaussianUnknownMean:
    def __init__(self, mean0, var0, varx):
        """Initialize model, for standard Bayes.
        Prior: Normal
        Likelihood: Normal known variance
        Predictive posterior: GaussNormalian
        """
        self.mean0 = mean0
        self.var0 = var0
        self.varx = varx
        self.mean_params = np.array([mean0])
        self.prec_params = np.array([1/var0])

    def pdf(self, observation, indices):
        """Compute predictive probabilities pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        post_means = self.mean_params[indices]
        post_stds = np.sqrt(self.var_params[indices])
        return st.norm(post_means, post_stds).pdf(observation)

    def update_params(self, observation):
        """Upon observing a new observation at time t,
        update all run length hypotheses.
        """
        new_prec_params = self.prec_params + (1/self.varx)
        new_mean_params = (self.mean_params * self.prec_params +
                           (observation / self.varx)) / new_prec_params

        self.mean_params = np.append([self.mean0], new_mean_params)
        self.prec_params = np.append([1/self.var0], new_prec_params)

    @property
    def var_params(self):
        """Helper function for computing the posterior variance.
        """
        return 1./self.prec_params + self.varx
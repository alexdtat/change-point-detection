import numpy as np
import scipy.stats as stats


class GaussianWithEstimation:
    def __init__(self):
        """
        Initializes a Gaussian likelihood with parameters estimation from a sample.
        NB: it operates with a [tail] gap's length, not with an absolute time.
        """
        self.sample_series = np.array([])
        self.means = np.array([])
        self.variances = np.array([])

        self.sample_sum = 0.
        self.squared_sample_sum = 0.
        self.gap_size = 0

    def accumulate_parameters(self, observation):
        """
        Accumulates data for the parameters estimation. Adds the observation to sum of the sample and
        the squared observation to the sum of squared sample values. Also appends the observation to the sample and
        increases the gap by one,
        :param observation: the next value from the time series.
        :return:
        """
        self.gap_size += 1
        self.sample_sum += observation
        self.squared_sample_sum += observation ** 2
        self.sample_series = np.append(self.sample_series, observation)

    def update_parameters(self, observation):
        """
        Updates the parameters estimation and appends newly evaluated mean and variance. Also accumulates data.
        :param observation: the next value from the time series.
        :return:
        """
        self.gap_size += 1
        self.sample_sum += observation
        self.squared_sample_sum += observation ** 2

        mean = self.sample_sum / self.gap_size
        variance = ((self.squared_sample_sum - (self.sample_sum ** 2) / self.gap_size) / (self.gap_size - 1))

        self.means = np.append(self.means, mean)
        self.variances = np.append(self.variances, variance)
        self.sample_series = np.append(self.sample_series, observation)

    def pdf(self, observation):
        """
        Evaluates the probability density function of the Gaussian distribution with estimated posterior hyperparameters.
        :param observation: the next value from the time series.
        :return: The probability density function of the Gaussian distribution with estimated posterior hyperparameters.
        It should be used for predictive probabilities evaluation ib the BOCPD.
        """
        return stats.norm(self.means, self.variances).pdf(observation)

    def prune(self, gap_size):
        """
        Prunes accumulated sample data before the [tail] gap of size gap_size. Discards all prior parameters.
        Should be used in point where a change is highly likely.
        Also updates a gap size.
        NB: it operates with a [tail] gap's length, not with an absolute time.
        :param gap_size: the size of the [tail] gap, which should be left after pruning.
        :return:
        """
        self.gap_size = gap_size
        self.sample_series = self.sample_series[-self.gap_size:]

        self.sample_sum = np.sum(self.sample_series)
        self.squared_sample_sum = 0.
        for x in self.sample_series:
            self.squared_sample_sum += x ** 2

        self.means = np.array([])
        self.variances = np.array([])

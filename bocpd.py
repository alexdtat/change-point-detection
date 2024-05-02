import numpy as np


class BOCPD:
    def __init__(self, hazard_function, observation_likelihood):
        """
        Initializes the detector with given functions. There are no observations yet.
        :param hazard_function: hazard function from the BOCPD algorithm.
        :param observation_likelihood: likelihood function from the BOCPD algorithm.
        """
        self.hazard_function = hazard_function
        self.observation_likelihood = observation_likelihood
        self.growth_probs = np.array([1.])
        self.start_time = 0
        self.time = -1
        self.gap_size = 0

    def update(self, observation):
        """
        Updates the growth and changepoint probabilities based on observation. Also updates model's parameters.
        :param observation: the next value from the time series.
        :return:
        """
        self.time += 1
        self.gap_size += 1
        gap_size = self.gap_size

        # Adjust allocated memory. Maybe can be replaced with appends as in likelihood functions.
        if len(self.growth_probs) == gap_size:
            self.growth_probs = np.resize(self.growth_probs, gap_size * 2)
            self.growth_probs[gap_size + 1:] = 0.

        # 3. Evaluate predictive probabilities for all run lengths and it's parameters inside likelihood functions.
        predictive_probs = self.observation_likelihood.pdf(observation)

        # 4. Evaluate the hazard function for the gap.
        hazard_val = self.hazard_function(np.array(range(gap_size)))

        # Evaluate the changepoint probability at *this* step (NB: generally it can be found later, with some delay).
        changepoint_prob = np.sum(
            self.growth_probs[0:gap_size] * predictive_probs * hazard_val)

        # Evaluate growth probabilities, shifting them down and to the right,
        # scaled by (1 - hazard function value) and prediction probabilities.
        self.growth_probs[1:gap_size + 1] = self.growth_probs[0:gap_size] * predictive_probs * (1. - hazard_val)

        # 5. Add CP probability.
        self.growth_probs[0] = changepoint_prob

        # 6. Evaluate evidence for growth probabilities renormalization.
        evidence = np.sum(self.growth_probs[0:gap_size + 2])

        # 7. Renormalize growth probabilities.
        self.growth_probs[0:gap_size + 2] = self.growth_probs[0:gap_size + 2] / evidence

        # 8. Update parameters of likelihood function for every possible run length (typically appends new values).
        self.observation_likelihood.update_parameters(observation)

    def most_likely_run_length(self):
        """
        Returns the most likely run-length and it's probability based on the growth and changepoint probabilities.
        Should be used for plots of run lengths.
        :return: the most likely run-length and it's probability.
        """
        return self.growth_probs.argmax(), self.growth_probs.max()

    def most_likely_non_max_run_length(self):
        """
        Returns the most likely non-max run-length and it's probability based on the growth and changepoint probabilities.
        Should be used for CPD, where it is being compared with a threshold. Max run length corresponds to an old CP.
        :return: the most likely non-max run-length and it's probability.
        """
        return self.growth_probs[:-1].argmax(), self.growth_probs[:-1].max()

    def set_gap(self, start_time):
        """
        Sets the size of the gap in the tail, where the BOCPD operates.
        :param start_time: new start time.
        :return:
        """
        self.start_time = start_time
        self.gap_size = self.time - self.start_time + 1

    def set_times(self, time):
        """
        Sets time; start time will be achieved on the next step.
        Should be used for synchronization in case some data was omitted from the BOCPD input,
        and we start it from the beginning (i.e. Gaussian likelihood's sample accumulation).
        :param time: new time value.
        :return:
        """
        self.time = time
        self.set_gap(time + 1)

    def prune(self, start_time):
        """
        Prunes data for all observations before start_time. Should be used in point where a change is highly likely.
        Also updates start time and gap size.
        :param start_time: time, before which data will be pruned.
        :return:
        """

        self.set_gap(start_time)
        self.observation_likelihood.prune(self.gap_size)
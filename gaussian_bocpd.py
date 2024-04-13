from bocpd import BOCPD
from gaussian_likelihood_with_estimation import GaussianWithEstimation


class GaussianBOCPD:

    def __init__(self, hazard_function, accumulation_steps_cap, threshold):
        """
        Initializes the detector with the given hazard function.
        Accumulation steps cap is used for parameters estimation. Threshold is used for a CPD itself.
        Likelihood function is Gaussian with estimation of parameters from a sample. There are no observations yet.
        Before
        :param hazard_function: hazard function from BOCPD algorithm.
        :param accumulation_steps_cap: Number of steps needed to accumulate a sample and
        to evaluate prior parameters for Gaussian likelihood function.
        :param threshold: Threshold with which we compare growth probabilities in a CPD.
        """
        self.accumulation_steps_cap = accumulation_steps_cap
        self.threshold = threshold
        self.time = -1
        self.left_accumulation_steps = accumulation_steps_cap

        self.changepoints = list()
        self.run_lengths = list()
        self.likelihood = GaussianWithEstimation()
        self.bocpd = BOCPD(hazard_function, self.likelihood)

    def update(self, observation):
        """
        Processes the next value from the time series.
        :param observation: the next value from the time series.
        :return:
        """
        self.time += 1

        # If at the moment we are in the state of sample accumulation,
        # the observation passes not into the BOCPD, but into the likelihood sample accumulation.
        if self.left_accumulation_steps > 0:
            self.likelihood.accumulate_parameters(observation)
            self.left_accumulation_steps -= 1

            # In this state are outside of the BOCPD, so we don't have any reliable run lengths.
            # In a plot there will be none points inside this gap.
            self.run_lengths.append(None)

            # If accumulation end at this step, we should evaluate first prior parameters from the sample and
            # synchronize times with the lagging BOCPD.
            if self.left_accumulation_steps == 0:
                self.likelihood.update_parameters(observation)
                self.bocpd.set_times(self.time)

        # If at the moment we are inside the BOCPD, we should update BOCPD and look at run lengths.
        else:
            self.bocpd.update(observation)
            (non_max_run_length, non_max_prob) = self.bocpd.most_likely_non_max_run_length()

            # We add the most likely run length to the run lengths probabilities plot data.
            run_length, prob = self.bocpd.most_likely_run_length()
            self.run_lengths.append(run_length)

            # Max run length corresponds to an old CP, so we need to find the most likely non-max run length,
            # which one's probability surpasses both the max run length's one and the threshold. It will be a CP.
            if non_max_prob > self.threshold and non_max_prob >= prob:
                changepoint = self.bocpd.start_time + non_max_run_length
                self.left_accumulation_steps = max(0, self.accumulation_steps_cap - non_max_run_length)
                self.bocpd.prune(changepoint)

                self.changepoints.append(changepoint)

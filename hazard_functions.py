import numpy as np


def constant_hazard(lambda_parameter, run_length):
    """
    Hazard function for a discrete exponential (geometric) distribution, which is a constant function.
    :param lambda_parameter: timescale parameter,
    :param run_length: run lengths array from a BOCPD.
    :return: Constant hazard function array for a discrete exponential (geometric) distribution.
    """
    return np.ones(run_length.shape) / lambda_parameter
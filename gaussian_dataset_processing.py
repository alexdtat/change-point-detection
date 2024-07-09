from functools import partial

import matplotlib.pyplot as plt
import pandas as pd

from bocpd import *
from gaussian_bocpd import GaussianBOCPD
from hazard_functions import constant_hazard

LAMBDA_WELL = 250
LAMBDA_GENERATED = 100
THRESHOLD = 0.5
ACCUMULATION_STEPS = 15

series = pd.read_csv('datasets/well-log.txt')['Response'].to_numpy().astype(np.float64)

np.random.seed(0)
generated_series = np.random.normal(size=1000)
generated_series[len(generated_series) // 4:len(generated_series) // 2] += 10.
generated_series[len(generated_series) // 2:3 * len(generated_series) // 4] -= 10.

indices = np.arange(generated_series.size)

bocpd = GaussianBOCPD(partial(constant_hazard, LAMBDA_WELL), ACCUMULATION_STEPS, THRESHOLD)
for observation in generated_series:
    bocpd.update(observation)

print(bocpd.changepoints)
plt.plot(indices, bocpd.run_lengths, label='*')
plt.show()

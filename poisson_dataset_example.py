from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from bocpd import BOCPD
from hazard_functions import constant_hazard
from likelihood_functions import Poisson

LAMBDA = 1000
K = 1.0
THETA = 1.0
START_YEAR = 1851

series = pd.read_excel('datasets/COAL MINING DISASTERS UK.xlsx')['Count'].to_numpy().astype(np.float64)
poisson_likelihood = Poisson(K, THETA)
model = BOCPD(partial(constant_hazard, LAMBDA), poisson_likelihood)
run_lengths_max = list([0])

for observation in series:
    model.update(observation)
    run_lengths_max.append(np.argmax(model.growth_probs))

years = np.arange(START_YEAR, START_YEAR + series.size + 1, 1)

plt.plot(years, run_lengths_max)

plt.title('Наиболее вероятная длина пробега в каждый год')
plt.xlabel('Год')
plt.ylabel('Наиболее вероятная длина пробега')

plt.show()
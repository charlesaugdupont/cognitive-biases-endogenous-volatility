from scipy.stats.qmc import LatinHypercube
from model import *

def generate_samples(num_samples, num_params, seed):
    lh = LatinHypercube(d=num_params, seed=seed)
    sample = lh.random(n=num_samples)
    return sample

def gini_coeff(data):
    if len(data) == 0:
        return 0
    data = np.asarray(data)
    data = np.sort(data)
    n = len(data)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * data)) / (n * np.sum(data)) - (n + 1) / n
    return gini

def mean_util(data):
    return np.mean(data)

def sen_welfare(data):
    return mean_util(data) * (1-gini_coeff(data))
import numpy as np

def interval_score(x, lower, upper, alpha=0.05):
    assert np.all(upper>=lower)
    return (upper - lower) + (2/alpha)*(lower-x)*(x<lower) + (2/alpha)*(x-upper)*(x>upper)
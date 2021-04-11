import numpy as np


def filter_delta(cutoff):
    # using that N(d1) = 1 - N(-d1)
    return lambda df: df[(np.abs(df['delta']) > cutoff) & (np.abs(df['delta']) < 1 - cutoff)]


def filter_maturity():
    return lambda df: df[(df['ttm'] > 0.05) & (df['ttm'] < 0.5)]


def filter_volume(cutoff):
    return lambda df: df[df['volume'] > cutoff]

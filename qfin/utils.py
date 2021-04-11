import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import norm
from scipy.optimize import newton


def bs_explicit_call(ttm, strike, spot, rate, SIGMA, dividends=0.):
    d1 = (np.log(spot / strike) + (rate - dividends + SIGMA ** 2 / 2) * ttm) / SIGMA / np.sqrt(ttm)
    d2 = d1 - SIGMA * np.sqrt(ttm)
    return np.exp(-dividends * ttm) * norm.cdf(d1) * spot - norm.cdf(d2) * strike * np.exp(-rate * ttm)


def bs_explicit_put(ttm, strike, spot, rate, SIGMA, dividends=0.):
    d1 = (np.log(spot / strike) + (rate - dividends + SIGMA ** 2 / 2) * ttm) / SIGMA / np.sqrt(ttm)
    d2 = d1 - SIGMA * np.sqrt(ttm)
    return norm.cdf(-d2) * strike * np.exp(-rate * ttm) - np.exp(-dividends * ttm) * norm.cdf(-d1) * spot


# same for calls and puts
def bs_vega(ttm, strike, spot, rate, SIGMA, dividends=0.):
    d1 = (np.log(spot / strike) + (rate + SIGMA ** 2 / 2) * ttm) / SIGMA / np.sqrt(ttm)
    return spot * np.exp(-dividends * ttm) * norm.pdf(d1) * np.sqrt(ttm)


def bs_call_delta(ttm, strike, spot, rate, SIGMA, dividends=0):
    d1 = (np.log(spot / strike) + (rate + SIGMA ** 2 / 2) * ttm) / SIGMA / np.sqrt(ttm)
    return np.exp(-dividends * ttm) * norm.cdf(d1)


def bs_put_delta(ttm, strike, spot, rate, SIGMA, dividends=0):
    d1 = (np.log(spot / strike) + (rate + SIGMA ** 2 / 2) * ttm) / SIGMA / np.sqrt(ttm)
    return -np.exp(-dividends * ttm) * norm.cdf(-d1)


def bs_gamma(ttm, strike, spot, rate, SIGMA, dividends=0):
    assert dividends == 0, "please implement dividends"
    d1 = (np.log(spot / strike) + (rate + 0.5 * SIGMA ** 2) * ttm) / (SIGMA * np.sqrt(ttm))
    return norm.pdf(d1) / (spot * SIGMA * np.sqrt(ttm))


def bs_iv(ttm, strike, spot, rate, price, otype='C', dividends=0., x0=0.7):

    assert otype.upper() in ('C', 'P'), "Option type must be 'C' or 'P'"

    def target(sigma):
        return otype.upper() == 'C' \
               and bs_explicit_call(ttm, strike, spot, rate, sigma, dividends) - price \
               or bs_explicit_put(ttm, strike, spot, rate, sigma, dividends) - price

    try:
        return newton(target, x0=x0)
    except RuntimeError as e:
        return np.nan


def sizeof_fmt(num, suffix='B'):

    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0

    return "%.1f%s%s" % (num, 'Yi', suffix)


# interpolation options functions
def interpolate(fn, model, period, asset):

    # interpolation parameters
    n = 200
    a1, a2 = -0.9, 5.

    output = np.empty(asset.underlying.paths.shape)
    output[:] = np.nan

    for idx, day in enumerate(period.date_range):

        dt = asset.maturity - day
        ttm = dt.days / 365
        if ttm <= 0:
            break

        x0 = model.parameters(day).to_numpy()
        xs = asset.strike * (1 + np.linspace(a1, a2, n))
        ys = fn(x0, ttm, asset.strike, xs, asset.underlying.rate)

        interp = InterpolatedUnivariateSpline(xs, ys, k=1)
        output[:, idx] = interp(asset.underlying.paths[:, idx])

    return output

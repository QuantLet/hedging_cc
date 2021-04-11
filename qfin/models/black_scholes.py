import numpy as np

from qfin.models.model import Model
from qfin.utils import bs_explicit_call, bs_explicit_put, bs_call_delta, bs_put_delta, bs_gamma, bs_vega


class BlackScholesModel(Model):

    name = "BLACK_SCHOLES"
    labels = "SIGMA",
    bounds = [(1e-5, np.inf)]

    def call(self, x0, ttm, strike, spot=1., rate=0., dividends=0.):
        return bs_explicit_call(ttm, strike, spot, rate, x0[0], dividends)

    def put(self, x0, ttm, strike, spot=1., rate=0., dividends=0.):
        return bs_explicit_put(ttm, strike, spot, rate, x0[0], dividends)

    def call_delta(self, x0, ttm, strike, spot=1., rate=0., dividends=0.):
        return bs_call_delta(ttm, strike, spot, rate, x0[0], dividends)

    def put_delta(self, x0, ttm, strike, spot=1., rate=0., dividends=0.):
        return bs_put_delta(ttm, strike, spot, rate, x0[0], dividends)

    def call_gamma(self, x0, ttm, strike, spot=1., rate=0., dividends=0.):
        return bs_gamma(ttm, strike, spot, rate, x0[0], dividends)

    def put_gamma(self, x0, ttm, strike, spot=1., rate=0., dividends=0.):
        return bs_gamma(ttm, strike, spot, rate, x0[0], dividends)

    def call_vega(self, x0, ttm, strike, spot=1., rate=0., dividends=0.):
        return bs_vega(ttm, strike, spot, rate, x0[0], dividends)

    def put_vega(self, x0, ttm, strike, spot=1., rate=0., dividends=0.):
        return bs_vega(ttm, strike, spot, rate, x0[0], dividends)

    def cf(self, x0, u, ttm, x, rate):
        return np.exp(1j * u * (x + (rate - 1 / 2 * x0[0] ** 2) * ttm) - 1 / 2 * u ** 2 * x0[0] ** 2 * ttm)

    def paths(self, paths, period, s0, rate, npaths):

        dt = 1 / 365
        parameters = self.parameters(period.date_range[:-1]).to_numpy().reshape(1, -1)

        paths[:, 0] = 0
        paths[:, 1:period.days] = (rate - parameters ** 2 / 2) * dt + parameters * np.sqrt(dt) * np.random.standard_normal((npaths, period.days - 1))
        paths[:, :period.days] = s0 * np.exp(np.cumsum(paths[:, :period.days], axis=1))

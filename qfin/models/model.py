from abc import ABC
import numpy as np
import pandas as pd

from numpy.fft import *

from qfin.utils import bs_iv
from qfin.volsurf import VolatilitySurface


class Model(ABC):

    labels = ...
    bounds = ...
    cf = ...
    name = ...

    def __init__(self, alphas=(0.8, -1.5), accuracy_factor=1, fourier_discretization_steps=2**12):
        self.alphas = alphas
        self.accuracy_factor = accuracy_factor
        self.fourier_discretization_steps = fourier_discretization_steps

    def call(self, x0, ttm, strike, spot=1., rate=0., dividends=0.):
        alpha = self.alphas[0]
        return self.fft(x0, ttm, strike, spot, rate, dividends, alpha)

    def put(self, x0, ttm, strike, spot=1., rate=0., dividends=0.):
        alpha = self.alphas[1]
        return self.fft(x0, ttm, strike, spot, rate, dividends, alpha)

    def iv(self, x0, ttm, strike, spot=1., rate=0., dividends=0.):
        return spot <= strike \
               and self.call_iv(x0, ttm, strike, spot, rate, dividends) \
               or self.put_iv(x0, ttm, strike, spot, rate, dividends)

    def call_iv(self, x0, ttm, strike, spot=1., rate=0., dividends=0.):
        call = self.call(x0, ttm, strike, spot, rate, dividends)
        return bs_iv(ttm, strike, spot, rate, call, 'C')

    def put_iv(self, x0, ttm, strike, spot=1., rate=0., dividends=0.):
        put = self.put(x0, ttm, strike, spot, rate, dividends)
        return bs_iv(ttm, strike, spot, rate, put, 'P')

    def call_delta(self, x0, ttm, strike, spot=1., rate=0., dividends=0.):
        ds = 0.1 # @todo fix this
        c1 = self.call(x0, ttm, strike, spot + ds, rate, dividends)
        c2 = self.call(x0, ttm, strike, spot, rate, dividends)
        delta = (c1 - c2) / ds
        return np.clip(delta, 0, 1)

    def put_delta(self, x0, ttm, strike, spot=1., rate=0., dividends=0.):
        ds = 0.1 # @todo fix this
        c1 = self.put(x0, ttm, strike, spot + ds, rate, dividends)
        c2 = self.put(x0, ttm, strike, spot, rate, dividends)
        delta = (c1 - c2) / ds
        return np.clip(delta, -1, 0)

    def call_gamma(self, x0, ttm, strike, spot=1., rate=0., dividends=0.):
        ds = 0.1 # @todo fix this
        c1 = self.call(x0, ttm, strike, spot - ds, rate, dividends)
        c2 = self.call(x0, ttm, strike, spot, rate, dividends)
        c3 = self.call(x0, ttm, strike, spot + ds, rate, dividends)
        gamma = (c1 - 2 * c2 + c3) / ds ** 2
        return gamma

    # should be the same as call, actually
    def put_gamma(self, x0, ttm, strike, spot=1., rate=0., dividends=0.):
        ds = 0.1 # @todo fix this
        c1 = self.put(x0, ttm, strike, spot - ds, rate, dividends)
        c2 = self.put(x0, ttm, strike, spot, rate, dividends)
        c3 = self.put(x0, ttm, strike, spot + ds, rate, dividends)
        gamma = (c1 - 2 * c2 + c3) / ds ** 2
        return gamma

    def call_vega(self, x0, ttm, strike, spot=1., rate=0., dividends=0.):
        return NotImplementedError(f"Vega not implemented for the model {self.name}")

    def put_vega(self, x0, ttm, strike, spot=1., rate=0., dividends=0.):
        return NotImplementedError(f"Vega not implemented for the model {self.name}")

    def vs(self, x0, points):
        rows = [[ttm, moneyness, self.iv(x0, ttm, moneyness)] for ttm, moneyness in points]
        data = pd.DataFrame(rows, columns=['ttm', 'moneyness', 'iv'])
        return VolatilitySurface(data)

    def fft(self, x0, ttm, strike, spot, rate, dividends, alpha):

        k = np.log(strike / spot)
        x = 0.

        g = self.accuracy_factor
        n = g * self.fourier_discretization_steps

        eps = (g * 150.) ** -1
        eta = 2 * np.pi / (n * eps)
        b = 0.5 * n * eps - k
        u = np.arange(1, n + 1)
        v0 = eta * (u - 1)

        delt = np.zeros(n, dtype=np.float)
        delt[0] = 1
        j = np.arange(1, n + 1)
        simpson = (3 + (-1) ** j - delt) / 3

        v = v0 - (alpha + 1) * 1j
        mod_char_fun = np.exp(-rate * ttm) * self.cf(x0, v, ttm, x, rate - dividends)
        mod_char_fun /= (alpha ** 2 + alpha - v0 ** 2 + 1j * (2 * alpha + 1) * v0)

        fft_func = np.exp(1j * np.tensordot(b, v0, axes=0)) * mod_char_fun * eta * simpson

        payoff = fft(fft_func).real
        call_value_m = payoff * np.atleast_1d(np.exp(-alpha * k))[..., np.newaxis] / np.pi

        pos = np.rint(0.5 * n).astype(dtype=int)
        call_value = call_value_m[:, ..., pos]

        retval = call_value * spot

        # de-vectorize if input was scalar
        if np.isscalar(spot) and np.isscalar(strike):
            retval = retval[0]

        return retval

    # accepts a single day or list of days as the argument
    def parameters(self, day):
        fname = f"_output/calibration/results/20210227_195345/{self.name}/parameters.csv"
        parameters = pd.read_csv(fname).sort_values('date')
        parameters['date'] = pd.to_datetime(parameters['date'], format='%Y%m%d')
        parameters = parameters.set_index('date', drop=True)
        return parameters.loc[day, list(self.labels)]

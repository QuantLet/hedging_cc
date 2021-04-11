import numpy as np

from qfin.models.model import Model
from qfin.utils import bs_vega


class MertonModel(Model):
    name = "MERTON"
    labels = "LAMBDA", "MU_Y", "SIGMA_Y", "SIGMA"

    bounds = [
        (1e-5, np.inf),         # LAMBDA    -- jump intensity
        (-np.inf, np.inf),      # MU_Y      -- mean of the jump
        (1e-5, np.inf),         # SIGMA_Y   -- vol of the jump
        (1e-5, np.inf),         # SIGMA     -- vol of the diffusion
    ]

    # same as BS vega
    def call_vega(self, x0, ttm, strike, spot=1., rate=0., dividends=0.):
        return bs_vega(ttm, strike, spot, rate, x0[3], dividends)

    # same as BS vega
    def put_vega(self, x0, ttm, strike, spot=1., rate=0., dividends=0.):
        return bs_vega(ttm, strike, spot, rate, x0[3], dividends)

    def cf(self, x0, u, t, x, rate):
        LAMBDA, MU_Y, SIGMA_Y, SIGMA = x0
        lognormal_jump_cf = LAMBDA * t * (-MU_Y * u * 1j + (np.exp(u * 1j * np.log(1.0 + MU_Y) + 0.5 * SIGMA_Y ** 2 * u * 1j * (u * 1j - 1.0)) - 1.0))
        bs_cf = np.exp(1j * u * (x + (rate - 1 / 2 * SIGMA ** 2) * t) - 1 / 2 * u ** 2 * SIGMA ** 2 * t)
        return bs_cf * np.exp(lognormal_jump_cf)

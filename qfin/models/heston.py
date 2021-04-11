import numpy as np

from qfin.models.model import Model


class HestonModel(Model):
    name = "HESTON"
    labels = "KAPPA", "RHO", "V0", "VBAR", "XI"

    bounds = [
        (1e-5, np.inf),     # KAPPA     -- mean-reversion
        (-1, 1),            # RHO       -- correlation
        (1e-5, np.inf),     # V0        -- short-term variance
        (1e-5, np.inf),     # VBAR      -- long-term variance
        (1e-5, np.inf),     # XI        -- vol of vol
    ]

    def call_vega(self, x0, ttm, strike, spot=1., rate=0., dividends=0.):

        eps = 0.01 # @todo fix this
        KAPPA, RHO, V0, VBAR, XI = x0

        call_base = self.call(x0, ttm, strike, spot, rate)

        x0[2] = V0 + eps
        x0[3] = VBAR
        call_v0 = self.call(x0, ttm, strike, spot, rate)
        d_v0 = (call_v0 - call_base) / eps

        x0[2] = V0
        x0[3] = VBAR + eps
        call_vbar = self.call(x0, ttm, strike, spot, rate)
        d_vbar = (call_vbar - call_base) / eps

        # total differential
        cash_vega = 2 * np.sqrt(V0) * d_v0 * np.sqrt(eps) + 2 * np.sqrt(VBAR) * d_vbar * np.sqrt(eps)
        return cash_vega / np.sqrt(eps)

    def cf(self, x0, u, ttm, x, rate):

        KAPPA, RHO, V0, VBAR, XI = x0

        alpha = (- u ** 2 - 1j * u) / 2
        beta = KAPPA - RHO * XI * 1j * u
        gamma = XI ** 2 / 2

        d = np.sqrt(beta ** 2 - 4 * alpha * gamma)

        r_pos = (beta + d) / 2 / gamma
        r_neg = (beta - d) / 2 / gamma
        g = r_neg / r_pos

        # alpha_0
        def C(t):
            return r_neg * t - 2 / XI ** 2 * np.log((1 - g * np.exp(-d * ttm)) / (1 - g))

        # beta
        def D(t):
            return r_neg * (1 - np.exp(-d * ttm)) / (1 - g * np.exp(-d * ttm))

        return np.exp(VBAR * KAPPA * C(ttm) +
                      V0 * D(ttm) +
                      1j * u * (x + rate * ttm))



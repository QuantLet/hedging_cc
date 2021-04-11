import numpy as np

from qfin.models.model import Model


class SVJModel(Model):

    name = "SVJ"
    labels = "KAPPA", "RHO", "V0", "VBAR", "XI", "LAMBDA", "MU_Y", "SIGMA_Y"

    bounds = [
        (1e-5, np.inf),         # KAPPA     -- mean-reversion
        (-1, 1),                # RHO       -- correlation
        (1e-5, np.inf),         # V0        -- short-term variance
        (1e-5, np.inf),         # VBAR      -- long-term variance
        (1e-5, np.inf),         # XI        -- vol of vol
        (1e-5, np.inf),         # LAMBDA    -- jump intensity
        (-np.inf, np.inf),      # MU_Y      -- mean of the jumps
        (1e-5, np.inf),         # SIGMA_Y   -- volatility of the jumps
    ]

    def M76_char_func(self, x0, u, t, x, rate):
        """ Valuation of European call option in M76 model via Lewis (2001)
        Fourier-based approach: characteristic function.
        Parameter definitions see function M76_call_value."""
        KAPPA, RHO, V0, VBAR, XI, LAMBDA, MU_Y, SIGMA_Y = x0

        omega = -LAMBDA * (np.exp(MU_Y + 0.5 * SIGMA_Y ** 2) - 1)
        char_func_value = np.exp((1j * u * omega + LAMBDA * (np.exp(1j * u * MU_Y - u ** 2 * SIGMA_Y ** 2 * 0.5) - 1)) * t)

        return char_func_value

    def H93_char_func(self, x0, u, t, x, rate):
        """ Valuation of European call option in H93 model via Lewis (2001)
        Fourier-based approach: characteristic function.
        Parameter definitions see function BCC_call_value."""
        KAPPA, RHO, V0, VBAR, XI, LAMBDA, MU_Y, SIGMA_Y = x0

        c1 = KAPPA * VBAR
        c2 = -np.sqrt((RHO * XI * u * 1j - KAPPA) ** 2 -
                      XI ** 2 * (-u * 1j - u ** 2))
        c3 = (KAPPA - RHO * XI * u * 1j + c2) \
             / (KAPPA - RHO * XI * u * 1j - c2)
        H1 = (rate * u * 1j * t + (c1 / XI ** 2) *
              ((KAPPA - RHO * XI * u * 1j + c2) * t -
               2 * np.log((1 - c3 * np.exp(c2 * t)) / (1 - c3))))
        H2 = ((KAPPA - RHO * XI * u * 1j + c2) / XI ** 2 *
              ((1 - np.exp(c2 * t)) / (1 - c3 * np.exp(c2 * t))))
        char_func_value = np.exp(H1 + H2 * V0)

        return char_func_value

    def call_vega(self, x0, ttm, strike, spot=1., rate=0., dividends=0.):

        # @todo fix this
        eps = 0.01
        KAPPA, RHO, V0, VBAR, XI, LAMBDA, MU_Y, SIGMA_Y = x0

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

    def cf(self, x0, u, t, x, rate):
        """ Valuation of European call option in BCC97 model via Lewis (2001)
        Fourier-based approach: characteristic function.
        Parameter definitions see function BCC_call_value."""
        BCC1 = self.H93_char_func(x0, u, t, x, rate)
        BCC2 = self.M76_char_func(x0, u, t, x, rate)
        return BCC1 * BCC2 * np.exp(1j * u * (x + rate * t))

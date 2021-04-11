import numpy as np

from qfin.models.model import Model


# https://www.darrellduffie.com/uploads/pubs/DuffiePanSingleton2000.pdf
class SVCJModel(Model):

    name = "SVCJ"
    labels = "KAPPA", "RHO", "V0", "VBAR", "XI", "LAMBDA", "MU_Y", "SIGMA_Y", "MU_V"

    bounds = [
        (1e-5, np.inf),         # KAPPA     -- mean-reversion
        (-1, 1),                # RHO       -- correlation
        (1e-5, np.inf),         # V0        -- short-term variance
        (1e-5, np.inf),         # VBAR      -- long-term variance
        (1e-5, np.inf),         # XI        -- vol of vol
        (1e-5, np.inf),         # LAMBDA    -- jump intensity
        (-np.inf, np.inf),      # MU_Y      -- mean of the jumps
        (1e-5, np.inf),         # SIGMA_Y   -- volatility of the jumps
        (1e-5, np.inf),         # MU_V      -- mean of the variance jumps
    ]

    def call_vega(self, x0, ttm, strike, spot=1., rate=0., dividends=0.):

        eps = 0.01 # @todo fix this
        KAPPA, RHO, V0, VBAR, XI, LAMBDA, MU_Y, SIGMA_Y, MU_V = x0

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

        KAPPA, RHO, V0, VBAR, XI, LAMBDA, MU_Y, SIGMA_Y, MU_V = x0

        u *= 1j

        # empirical evidence discards necessity for RHO_J
        LAMBDA_Y = LAMBDA
        LAMBDA_V = LAMBDA
        LAMBDA_C = 0

        a = u * (1 - u)
        b = XI * RHO * u - KAPPA
        gamma = np.sqrt(b ** 2 + a * XI ** 2)

        # dividends
        zeta_bar = 0

        alpha_0 = -rate * t
        alpha_0 = np.add(alpha_0, (rate - zeta_bar) * u * t, casting="unsafe")
        alpha_0 -= KAPPA * VBAR * ((gamma + b) / XI ** 2 * t + 2 / XI ** 2 * np.log(1 - (gamma + b) / 2 / gamma * (1 - np.exp(-gamma * t))))

        lambda_bar = LAMBDA_V + LAMBDA_Y + LAMBDA_C

        theta_y = lambda c: np.exp(MU_Y * c + SIGMA_Y ** 2 * c ** 2 / 2)
        theta_v = lambda c: 1 / (1 - MU_V * c)
        theta_c = 0

        theta = lambda c1, c2: (LAMBDA_Y * theta_y(c1) + LAMBDA_V * theta_v(c2) + LAMBDA_C * theta_c) / lambda_bar
        mu_bar = theta(1, 0) - 1

        integ = LAMBDA_Y * t * np.exp(MU_Y * u + SIGMA_Y ** 2 * u ** 2 / 2)
        integ += LAMBDA_V * t * (gamma - b) / (gamma - b + MU_V * a)
        integ -= LAMBDA_V * (2 * MU_V * a) / (gamma ** 2 - (b - MU_V * a) ** 2) * np.log(1 - (gamma + b - MU_V * a) / 2 / gamma * (1 - np.exp(-gamma * t)))
        integ /= lambda_bar

        alpha_bar = alpha_0 - lambda_bar * t * (1 + mu_bar * u) + lambda_bar * integ
        beta_bar = - a * (1 - np.exp(-gamma * t)) / (2 * gamma - (gamma + b) * (1 - np.exp(-gamma * t)))

        return np.exp(alpha_bar + beta_bar * V0) * np.exp(1j * u * (x + rate * t))

    # http://apps.olin.wustl.edu/faculty/belaygorod/Change_of_Measure.pdf, page 8
    def paths(self, paths, period, s0, rate, npaths):

        dt = 1/365

        w = np.random.standard_normal([npaths, period.days])
        w1 = np.random.standard_normal([npaths, period.days])
        w2 = np.zeros([npaths, period.days])

        variance = np.zeros([npaths, period.days])

        x0 = self.parameters(period.date_range[0])
        KAPPA, RHO, V0, VBAR, XI, LAMBDA, MU_Y, SIGMA_Y, MU_V = x0

        paths[:, 0] = s0
        variance[:, 0] = V0

        for i in range(1, period.days):

            x0 = self.parameters(period.date_range[i]).to_numpy()
            KAPPA, RHO, V0, VBAR, XI, LAMBDA, MU_Y, SIGMA_Y, MU_V = x0

            # assume zero correlation
            corr = 0

            w2[:, i] = RHO * w[:, i] + np.sqrt(1 - RHO ** 2) * w1[:, i]
            z_v = np.random.exponential(MU_V, size=npaths)
            z_y = np.random.standard_normal(npaths) * SIGMA_Y + MU_Y + corr * z_v
            dj = np.random.binomial(1, LAMBDA * dt, size=npaths)

            variance[:, i] = variance[:, i - 1] + \
                              KAPPA * (VBAR - np.maximum(0, variance[:, i - 1])) * dt + \
                              XI * np.sqrt(np.maximum(0, variance[:, i - 1]) * dt) * w2[:, i - 1] + \
                              z_v * dj

            paths[:, i] = paths[:, i - 1] * \
                            (1 + (rate - LAMBDA * (MU_Y + corr * MU_V)) * dt +
                             np.sqrt(np.maximum(0, variance[:, i - 1]) * dt) * w1[:, i - 1] +
                             z_y * dj)

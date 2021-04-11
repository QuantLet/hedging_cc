import numpy as np
import logging

from qfin.hedges import OptionHedge
from qfin.utils import interpolate, bs_explicit_call
from qfin.volsurf import VolatilitySurface

logger = logging.getLogger(__name__)


class MinimumVarianceHedge(OptionHedge):

    @property
    def name(self):
        return "MinimumVarianceHedge"

    def generate(self):

        assert self.hedgeable.model.name in ['MERTON', 'VARIANCE_GAMMA', 'HESTON', 'SVJ', 'SVCJ'], "implemented only for HESTON, SVJ, SVCJ, MERTON, VARIANCE_GAMMA"

        logger.info(f">> Generating interpolated deltas.")
        delta = interpolate(self.hedgeable.model.call_delta, self.hedgeable.model, self.period, self.hedgeable)

        # Kienitz
        if self.hedgeable.model.name in ['HESTON', 'SVJ', 'SVCJ']:

            logger.info(f">> Generating interpolated vegas.")
            vega = interpolate(self.hedgeable.model.call_vega, self.hedgeable.model, self.period, self.hedgeable)

            RHO, XI = self.hedgeable.model.parameters(self.period.date_range)[['RHO', 'XI']].to_numpy().T
            self.strategies[:] = np.array([delta + vega * RHO * XI / self.assets[0].paths])

        # Kienitz p. 109
        elif self.hedgeable.model.name == 'MERTON':

            logger.info(f">> Calculating minimum variance hedge integral for MERTON.")

            for idx in range(self.days):

                date = self.period.date_range[idx]

                vs = VolatilitySurface.from_date(date, svi_caching=True)
                vs.calibrate()

                ttm = self.hedgeable.ttm(date)
                spot = self.hedgeable.underlying.paths[:, idx]

                LAMBDA, MU_Y, SIGMA_Y, SIGMA = self.hedgeable.model.parameters(date)

                # we integrate against the normal density
                # pick 7 points centered around mean at -3*std, -2*std, -1*std, 0, 1*std, 2*std, 3*std
                dt = 6 * SIGMA_Y / 7
                xs = MU_Y + np.arange(-3*SIGMA_Y, 3*SIGMA_Y, dt).reshape(-1, 1)

                nu = LAMBDA / np.sqrt(2*np.pi) / SIGMA_Y * np.exp(-(xs - MU_Y)**2 / 2 / SIGMA_Y**2)

                iv1 = vs.iv(ttm, self.hedgeable.strike / spot)
                c1 = bs_explicit_call(ttm, self.hedgeable.strike, spot, self.rate, iv1)

                iv2 = vs.iv(ttm, self.hedgeable.strike / (spot * np.exp(xs)))
                c2 = bs_explicit_call(ttm, self.hedgeable.strike, spot * np.exp(xs), self.rate, iv2)

                i0 = dt * np.sum((np.exp(xs) - 1) ** 2 * nu, axis=0)
                i1 = dt * np.sum(c1 * (np.exp(xs) - 1) * nu, axis=0)
                i2 = dt * np.sum(c2 * (np.exp(xs) - 1) * nu, axis=0)

                self.strategies[0, :, idx] = (SIGMA ** 2 * delta[:, idx]) / (SIGMA ** 2 + i0) + (i2 - i1) / (SIGMA ** 2 + i0) / spot

        # Kienitz, p. 109
        elif self.hedgeable.model.name == 'VARIANCE_GAMMA':

            logger.info(f">> Calculating minimum variance hedge integral for VARIANCE_GAMMA.")

            for idx in range(self.days):

                date = self.period.date_range[idx]

                vs = VolatilitySurface.from_date(date, svi_caching=True)
                vs.calibrate()

                ttm = self.hedgeable.ttm(date)
                spot = self.hedgeable.underlying.paths[:, idx]

                NU, SIGMA, THETA = self.hedgeable.model.parameters(date)

                dt = 6 * SIGMA**2/THETA / 7
                xs = np.arange(-3*SIGMA**2/THETA, 3*SIGMA**2/THETA, dt).reshape(-1, 1)

                nu = np.exp(THETA * xs / SIGMA**2) / THETA / np.abs(xs) * np.exp(-np.sqrt(THETA**2 + 2 * SIGMA**2 / NU * np.abs(xs)))

                iv1 = vs.iv(ttm, self.hedgeable.strike / spot)
                c1 = bs_explicit_call(ttm, self.hedgeable.strike, spot, self.rate, iv1)

                iv2 = vs.iv(ttm, self.hedgeable.strike / (spot * np.exp(xs)))
                c2 = bs_explicit_call(ttm, self.hedgeable.strike, spot * np.exp(xs), self.rate, iv2)

                i0 = dt * np.sum((np.exp(xs) - 1) ** 2 * nu, axis=0)
                i1 = dt * np.sum(c1 * (np.exp(xs) - 1) * nu, axis=0)
                i2 = dt * np.sum(c2 * (np.exp(xs) - 1) * nu, axis=0)

                self.strategies[0, :, idx] = (SIGMA ** 2 * delta[:, idx]) / (SIGMA ** 2 + i0) + (i2 - i1) / (SIGMA ** 2 + i0) / spot

        # elif self.hedgeable.model.name == 'CGMY':
        #
        #     logger.info(f">> Calculating minimum variance hedge integral for CGMY.")
        #
        #     for idx in range(self.days):
        #
        #         date = self.period.date_range[idx]
        #
        #         vs = VolatilitySurface.from_date(date, svi_caching=True)
        #         vs.calibrate()
        #
        #         ttm = self.hedgeable.ttm(date)
        #         spot = self.hedgeable.underlying.paths[:, idx]
        #
        #         C, G, M, Y = self.hedgeable.model.parameters(date)
        #
        #         dt = 6 * (1/G) / 7
        #         xs = np.arange(-3*(1/G), 3*(1/G), dt).reshape(-1, 1)
        #
        #         # Kienitz p. 120
        #         nu = (xs < 0) * C * np.exp(-G * np.abs(xs)) / np.abs(xs) ** (1 + Y)
        #         nu += (xs > 0) * C * np.exp(-M * xs) / xs ** (1 + Y)
        #
        #         iv1 = vs.iv(ttm, self.hedgeable.strike / spot)
        #         c1 = bs_explicit_call(ttm, self.hedgeable.strike, spot, self.rate, iv1)
        #
        #         iv2 = vs.iv(ttm, self.hedgeable.strike / (spot * np.exp(xs)))
        #         c2 = bs_explicit_call(ttm, self.hedgeable.strike, spot * np.exp(xs), self.rate, iv2)
        #
        #         i0 = dt * np.sum((np.exp(xs) - 1) ** 2 * nu, axis=0)
        #         i1 = dt * np.sum(c1 * (np.exp(xs) - 1) * nu, axis=0)
        #         i2 = dt * np.sum(c2 * (np.exp(xs) - 1) * nu, axis=0)
        #
        #         self.strategies[0, :, idx] = (SIGMA ** 2 * delta[:, idx]) / (SIGMA ** 2 + i0) + (i2 - i1) / (SIGMA ** 2 + i0) / spot

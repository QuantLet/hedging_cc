import numpy as np

from qfin.models.model import Model


class VarianceGammaModel(Model):
    name = "VARIANCE_GAMMA"
    labels = "NU", "SIGMA", "THETA"

    bounds = [
        (1e-5, np.inf),         # NU        -- variance rate of the gamma process
        (1e-5, np.inf),         # SIGMA     -- vol of the diffusion
        (-np.inf, np.inf)       # THETA     -- drift
    ]

    # https://engineering.nyu.edu/sites/default/files/2018-09/CarrEuropeanFinReview1998.pdf
    # formula (22)
    def cf(self, x0, u, t, x, rate):
        NU, SIGMA, THETA = x0
        vgp = (1 - 1j * u * THETA * NU + 0.5 * (SIGMA ** 2) * NU * (u ** 2)) ** (-t / NU)
        # need compensator for the risk-neutral measure
        cmp = 1 / NU * np.log(1 - THETA * NU - SIGMA ** 2 * NU / 2) * t
        return np.exp(1j * u * (x + (rate * t)) + 1j * u * cmp) * vgp

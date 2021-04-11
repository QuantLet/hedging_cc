import numpy as np
import logging

from qfin.hedges import OptionHedge
from qfin.utils import interpolate

logger = logging.getLogger(__name__)


class DeltaGammaHedge(OptionHedge):

    @property
    def name(self):
        return "DeltaGammaHedge"

    def generate(self):

        cutoff = 5.

        # this is the option to be hedged
        call1 = self.hedgeable

        # this is another option used for hedges
        call2 = self.assets[1]

        logger.info(f">> Generating interpolated deltas.")
        d1 = interpolate(self.hedgeable.model.call_delta, self.hedgeable.model, self.period, call1)
        d2 = interpolate(self.hedgeable.model.call_delta, self.hedgeable.model, self.period, call2)

        logger.info(f">> Generating interpolated gammas.")
        g1 = interpolate(self.hedgeable.model.call_gamma, self.hedgeable.model, self.period, call1)
        g2 = interpolate(self.hedgeable.model.call_gamma, self.hedgeable.model, self.period, call2)

        # avoid huge position in second asset by gamma cutoff
        g = g1 / g2
        g = np.where((g == -np.inf) + (g == np.inf), 0, g)
        g *= (np.abs(g) < cutoff)

        d = d1 - g * d2

        self.strategies[:] = np.array([d, g])

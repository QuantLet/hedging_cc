import logging
import numpy as np

from qfin.hedges import OptionHedge
from qfin.utils import interpolate

logger = logging.getLogger(__name__)


class DeltaVegaHedge(OptionHedge):

    @property
    def name(self):
        return "DeltaVegaHedge"

    def generate(self):

        cutoff = 5.

        # this is the option to be hedged
        call1 = self.hedgeable

        # this is another option used for hedges
        call2 = self.assets[1]

        logger.info(f">> Generating interpolated deltas.")
        d1 = interpolate(self.hedgeable.model.call_delta, self.hedgeable.model, self.period, call1)
        d2 = interpolate(self.hedgeable.model.call_delta, self.hedgeable.model, self.period, call2)

        logger.info(f">> Generating interpolated vegas.")
        v1 = interpolate(self.hedgeable.model.call_vega, self.hedgeable.model, self.period, call1)
        v2 = interpolate(self.hedgeable.model.call_vega, self.hedgeable.model, self.period, call2)

        # avoid huge position in second asset by vega cutoff
        v = v1 / v2
        v = np.where((v == -np.inf) + (v == np.inf), 0, v)
        v *= (np.abs(v) < cutoff)

        d = d1 - v * d2

        self.strategies[:] = np.array([d, v])

import numpy as np

from qfin.hedges import OptionHedge
from qfin.utils import interpolate


class DeltaHedge(OptionHedge):

    @property
    def name(self):
        return "DeltaHedge"

    def generate(self):
        delta = interpolate(self.hedgeable.model.call_delta, self.hedgeable.model, self.period, self.hedgeable)
        self.strategies[:] = np.array([delta])

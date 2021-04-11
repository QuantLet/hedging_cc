import numpy as np
from scipy.special import gamma

from qfin.models.model import Model


# The Fine Structure of Asset Returns: An Empirical Investigation
# Author(s): Peter Carr, Hélyette Geman, Dilip B. Madan and Marc Yor
class CGMYModel(Model):
    name = "CGMY"
    labels = "C", "G", "M", "Y"

    bounds = [
        (1e-5, np.inf),         # C -- overall level of activity
        (1e-5, np.inf),         # G -- exponential decay of the Levy density on the right
        (1e-5, np.inf),         # M -- exponential decay of the Levy density on the left
        (-np.inf, 2),           # Y -- fine structure of the stochastic process
    ]

    def cf(self, x0, u, t, x, rate):

        C, G, M, Y = x0

        def cgmy_process_cf(u, t):
            part_1 = C * gamma(-Y)
            part_2 = (M - 1j * u) ** Y - M ** Y + (G + 1j * u) ** Y - G ** Y
            return np.exp(t * part_1 * part_2)

        cgmy_omega = -(1 / t) * np.log(cgmy_process_cf(-1j, t))
        return np.exp(1j * u * (x + rate * t)) * np.exp(1j * u * cgmy_omega * t) * cgmy_process_cf(u, t)

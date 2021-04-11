from datetime import datetime
import numpy as np

from qfin.assets.asset import Asset
from qfin.utils import bs_explicit_call
from qfin.volsurf import VolatilitySurface


class Call(Asset):

    def __init__(self, underlying, maturity, strike, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.underlying = underlying
        self.maturity = maturity
        self.strike = strike

    @property
    def asset_name(self):
        maturity_str = datetime.strftime(self.maturity, '%-d%b%y').upper()
        return f"BTC-{maturity_str}-{self.strike}-C"

    def ttm(self, day):
        dt = self.maturity - day
        return dt.days / 365

    def payoff(self, underlying):
        return np.maximum(underlying - self.strike, 0)

    def generate(self):

        for idx, date in enumerate(self.period.date_range):

            vs = VolatilitySurface.from_date(date, svi_caching=True)
            vs.calibrate()

            ttm = self.ttm(date)
            spot = self.underlying.paths[:, idx]
            self.paths[:, idx] = bs_explicit_call(ttm, self.strike, spot, self.underlying.rate, vs.iv(ttm, self.strike / spot))

        dt = self.maturity - self.period.t0
        self.paths[:, dt.days] = np.maximum(self.underlying.paths[:, dt.days] - self.strike, 0)

import pandas as pd
from datetime import datetime
import numpy as np
import re
import math
import matplotlib.pyplot as plt
import os
import logging

from scipy.interpolate import interp1d
from scipy.optimize import NonlinearConstraint, minimize

from qfin.utils import bs_explicit_call, bs_iv
from qfin.volsurf.utils import raw_total_variance, get_crossings

logger = logging.getLogger(__name__)


class VolatilitySurface:
    """SVI Volatility Surface

    [Gatheral, Jacquier] Arbitrage-free SVI volatility surfaces
    https://arxiv.org/pdf/1204.0646.pdf

    Robust Calibration For SVI Model Arbitrage Free
    https://hal.archives-ouvertes.fr/hal-02490029/file/Robust_Calibration_For_SVI_Model_Arbitrage_Free_v4.pdf

    Args:
         data: dataframe with ttm, moneyness, iv
         name: name of the dataframe where calibrated SVI parameters are stored
         svi_gamma: parameter for penalization of crossedness to avoid calendar spread arbitrage
         svi_caching: if enabled, calibration results are stored and next time loaded from file
    """

    def __init__(self, data, name=None, svi_gamma=0.1, svi_caching=False):

        self.data = data
        self.name = name
        self.svi_gamma = svi_gamma
        self.svi_caching = svi_caching

        self.svi_parameters = None
        self.svi_interp = None

        self._svi_calibrated = False

    @classmethod
    def from_date(cls, date, filters=(), *args, **kwargs):

        date_str = datetime.strftime(date, "%Y%m%d")
        path = f"_input/tickers/{date_str}.csv"

        df = pd.read_csv(path)

        # preprocess data
        df = df[(df['best_ask_price'] != 0) & (df['best_ask_amount'] != 0) &
                (df['best_bid_price'] != 0) & (df['best_bid_amount'] != 0)]

        df = df[~df['instrument_name'].str.startswith('ETH')]
        df = df[df['instrument_name'].str.endswith('C') | df['instrument_name'].str.endswith('P')]
        df = df.loc[[odf.index[0] for instrument_name, odf in df.groupby('instrument_name')]]

        df = df.reset_index(drop=True)

        # extract option data
        pattern = r"^(\w+)-(\d\d?\w\w\w\d\d)-(\d+)-(\w)$"
        cols = pd.DataFrame([(maturity, float(strike), option_type) for btc, maturity, strike, option_type in
                             df['instrument_name'].apply(lambda s: re.match(pattern, s).groups())],
                            columns=['maturity', 'strike', 'option_type'])

        df[['maturity', 'strike', 'option_type']] = cols

        # replace maturity column with numerical value in years
        def ttm(maturity):

            maturity = datetime.strptime(maturity, '%d%b%y')

            assert maturity >= date

            if date == maturity:
                return 1 / 365

            # depends on the day count conventions
            return (maturity - date).days / 365

        df['ttm'] = df['maturity'].apply(ttm)

        # set volume
        df['volume'] = df['open_interest']

        # normalize volatility
        df = df[~df['mark_iv'].isna()]
        df['mark_iv'] = df['mark_iv'].astype(np.float)
        df['mark_iv'] /= 100.
        df = df[df['mark_iv'] != 0]

        df = df.rename(columns={'underlying_price': 'spot', 'option_type': 'type', 'mark_iv': 'iv'})

        # pick one of the two items
        idxs = []

        for (ttm, strike), odf in df.groupby(['ttm', 'strike']):
            if len(odf) == 1:
                idxs += [odf.index[0]]
            elif len(odf) == 2:
                # pick calls over puts
                idxs += [odf.index[1] if odf['delta'].iloc[1] > odf['delta'].iloc[0] else odf.index[0]]

        df = df.loc[idxs]
        df['moneyness'] = df['strike'] / df['spot']

        # postprocess data with filters
        for filter in filters:
            df = filter(df)

        df = df[['ttm', 'moneyness', 'iv', 'volume', 'delta']]
        df = df.sort_values(['ttm', 'moneyness']).reset_index()

        return cls(df, name=date_str, *args, **kwargs)

    @property
    def points(self):
        return self.data[['ttm', 'moneyness']].to_numpy()

    @property
    def svi_path(self):
        return f"_output/volsurf/svi/{self.name}.csv"

    def __sub__(self, other):
        iv1 = self.data.fillna(+np.inf).set_index(['ttm', 'moneyness'])['iv']
        iv2 = other.data.fillna(+np.inf).set_index(['ttm', 'moneyness'])['iv']
        return np.sqrt(np.mean((iv1 - iv2) ** 2))

    # direct interpolation of the SVI parameters
    def iv(self, ttm, moneyness):
        if not self._svi_calibrated:
            raise RuntimeError("Volatility surface is not calibrated.")

        ks = np.log(moneyness)
        return np.sqrt(raw_total_variance(ks, *self.svi_interp(ttm)) / ttm)

    # interpolation of the call prices according to Gatheral, Jacquier
    def iv2(self, ttm, moneyness):

        # implement interpolation only, extrapolation also possible, see paper
        ttms = np.array(sorted(set(self.data['ttm'])))
        assert np.min(ttms) <= ttm <= np.max(ttms)

        # locate two closest smiles to the given ttm
        ttm1 = ttms[np.argmin(ttms < ttm) - 1]
        ttm2 = ttms[np.argmax(ttms >= ttm)]

        x1 = self.svi_parameters.loc[np.argmax(self.svi_parameters['ttm'] == ttm1)].to_numpy()[1:6]
        tv1 = raw_total_variance(0, *x1)
        iv1 = np.sqrt(raw_total_variance(np.log(moneyness), *x1) / ttm1)
        cl1 = bs_explicit_call(ttm1, moneyness, 1, 0, iv1)

        x2 = self.svi_parameters.loc[np.argmax(self.svi_parameters['ttm'] == ttm2)].to_numpy()[1:6]
        tv2 = raw_total_variance(0, *x2)
        iv2 = np.sqrt(raw_total_variance(np.log(moneyness), *x2) / ttm2)
        cl2 = bs_explicit_call(ttm2, moneyness, 1, 0, iv2)

        v = tv1 + (ttm - ttm1) / (ttm2 - ttm1) * (tv2 - tv1)

        alpha = (np.sqrt(tv2) - np.sqrt(v)) / (np.sqrt(tv2) - np.sqrt(tv1))
        cl = alpha * cl1 + (1 - alpha) * cl2
        iv = bs_iv(ttm, moneyness, 1, 0, cl, 'C')

        return iv

    def calibrate(self):

        if self.svi_caching and os.path.exists(self.svi_path):
            self.svi_parameters = pd.read_csv(self.svi_path)

        else:

            logger.info(f">> Calibrating {self.name} SVI volatility surface.")

            rows = []
            old_x0 = None

            for ttm, odf in self.data.groupby('ttm'):

                ks, ivs = odf[['moneyness', 'iv']].to_numpy().T
                ks = np.log(ks)

                idx = np.round(np.linspace(0, 1, 4) * (len(ks) - 1)).astype(int)
                xs1 = ks[idx]
                ys1 = ivs[idx]

                # pull
                minvar = np.min(ys1) ** 2
                maxvar = np.max(ys1) ** 2

                def cost(x):

                    a, b, rho, m, sigma = x
                    w = a + b * (rho * (ks - m) + np.sqrt((ks - m) ** 2 + sigma ** 2))

                    penalty = 0.

                    if old_x0 is not None:
                        cs = get_crossings(old_x0, x)

                        if len(cs) > 0:
                            kts = np.array([cs[0] - 1, *(cs[1:] + cs[:-1]) / 2, cs[-1] + 1])
                            crossedness = np.maximum(raw_total_variance(kts, *x) - raw_total_variance(kts, *old_x0), 0)
                            penalty = self.svi_gamma * np.sum(crossedness)

                    return np.sqrt(np.mean((ivs ** 2 - w / ttm) ** 2)) + penalty

                def cons1(x):
                    a, b, rho, m, sigma = x
                    return (a - m * b * (rho + 1)) * (4 - a + b * m * (rho + 1)) / b ** 2 / (rho + 1) ** 2

                def cons2(x):
                    a, b, rho, m, sigma = x
                    return (a - m * b * (rho - 1)) * (4 - a + b * m * (rho - 1)) / b ** 2 / (rho - 1) ** 2

                def cons3(x):
                    a, b, rho, m, sigma = x
                    return b ** 2 * (rho + 1) ** 2

                def cons4(x):
                    a, b, rho, m, sigma = x
                    return b ** 2 * (rho - 1) ** 2

                constraints = [
                    NonlinearConstraint(cons1, 1, np.inf),
                    NonlinearConstraint(cons2, 1, np.inf),
                    NonlinearConstraint(cons3, 0, 4),
                    NonlinearConstraint(cons4, 0, 4),
                ]

                x0 = np.array([minvar, 0.1, 0., 0., 1.])
                bounds = [(1e-5, maxvar), (1e-2, 1), (-0.5, 0.5), (2 * np.min(ks), 2 * np.max(ks)), (1e-2, 1)]

                result = minimize(cost, x0, method='SLSQP', bounds=bounds, constraints=constraints)
                a, b, rho, m, sigma = result['x']
                penalty = result['fun']

                old_x0 = a, b, rho, m, sigma

                rows += [[ttm, a, b, rho, m, sigma, penalty]]

            columns = ['ttm', 'a', 'b', 'rho', 'm', 'sigma', 'penalty']
            self.svi_parameters = pd.DataFrame(rows, columns=columns)

            logger.info("Finished.")

            if self.svi_caching:
                os.makedirs(os.path.dirname(self.svi_path), exist_ok=True)
                self.svi_parameters.to_csv(self.svi_path, index=False)

        xs, *ys = self.svi_parameters[['ttm', 'a', 'b', 'rho', 'm', 'sigma']].to_numpy().T
        self.svi_interp = interp1d(xs, ys, bounds_error=False, fill_value="extrapolate")

        self._svi_calibrated = True

    def plot(self, fig=None, log_scale=True):

        colors = np.array(['b', 'r', 'g', 'c', 'm', 'y'])
        groups = self.data.groupby('ttm')

        if fig is not None:
            msg = "Volatility surface should have the same number of smiles."
            assert all(group[0] == ax.ttm for group, ax in zip(groups, fig.get_axes())), msg
            color = colors[np.argmax(fig.get_axes()[0].lines[-1].get_color() == colors) + 1]

        else:

            ncols = 3
            nrows = math.ceil(len(groups) / ncols)
            fig = plt.figure(constrained_layout=False, figsize=(12, 3 * nrows))
            gs1 = fig.add_gridspec(ncols=ncols, nrows=nrows, left=0.05, right=0.95, hspace=0.4, wspace=0.25)
            color = 'b'

            for idx, (ttm, df) in enumerate(groups):

                ax = fig.add_subplot(gs1[idx])
                ax.ttm = ttm
                ax.set_title(f"TTM = {ttm:.4f}")

        for idx, ((ttm, df), ax) in enumerate(zip(groups, fig.get_axes())):

            ks = df.reset_index()['moneyness']
            ivs = df['iv']

            xs = np.linspace(np.min(ks), np.max(ks), 101)
            log_xs = np.log(xs)

            if log_scale:
                ks = np.log(ks)

            ax.plot(ks, ivs, marker='o', c=color, linewidth=0)

            # svi
            if self._svi_calibrated:

                idx = np.argmax(np.isclose(self.svi_parameters['ttm'], ttm))
                a, b, rho, m, sigma = self.svi_parameters.loc[idx, ['a', 'b', 'rho', 'm', 'sigma']]
                w = a + b * (rho * (log_xs - m) + np.sqrt((log_xs - m) ** 2 + sigma ** 2))
                ys = np.sqrt(w / ttm)

                ax.plot(log_xs if log_scale else xs, ys, c=color)

        return fig

    def plot_total_variance(self, xs=None, log_scale=True):

        if not self._svi_calibrated:
            raise RuntimeError("Volatility surface is not calibrated.")

        if xs is None:
            xs = np.linspace(-1, 1, 101)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title('Total variance')

        for ttm, odf in self.data.groupby('ttm'):

            ks, ivs = odf[['moneyness', 'iv']].to_numpy().T
            ks = np.log(ks)

            idx = np.argmax(np.isclose(self.svi_parameters['ttm'], ttm))
            a, b, rho, m, sigma = self.svi_parameters.loc[idx, ['a', 'b', 'rho', 'm', 'sigma']]
            w = a + b * (rho * (xs - m) + np.sqrt((xs - m) ** 2 + sigma ** 2))
            ys = np.sqrt(w / ttm)

            if not log_scale:
                ks = np.exp(ks)
                xs = np.exp(xs)

            # plot total variance
            ax.scatter(ks, ivs ** 2 * ttm, s=10)
            ax.plot(xs, ys ** 2 * ttm)

        return fig

    def plot_points(self, log_scale=True):

        ys, xs = self.data[['ttm', 'moneyness']].to_numpy().T
        fig, ax = plt.subplots(figsize=(8, 5))

        if log_scale:
            xs = np.log(xs)

        ax.scatter(xs, ys, c='b', marker='x')

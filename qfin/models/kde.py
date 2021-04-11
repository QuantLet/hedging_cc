import arch
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity


class KDEModel:

    name = "KDE"

    def __init__(self, interval=365, kernel='gaussian', bandwidth=0.1):
        self.interval = interval
        self.bandwidth = bandwidth
        self.kernel = kernel

    def calibrate(self, returns):

        windows = [returns[i:i+self.interval] for i in range(len(returns) - self.interval + 1)]
        rows = []

        for window in windows:
            garch = arch.arch_model(window - window.mean(), mean="Zero", rescale=False)
            res = garch.fit(disp="off")
            alpha0, alpha1, beta = res.params
            volest = alpha0 + (alpha1 * (window[-1] - window.mean()) ** 2 + beta * res.conditional_volatility[-1] ** 2)
            forecast_date = window.index[-1]
            rows.append([forecast_date, np.sqrt(volest)])

        df = pd.DataFrame(rows, columns=['date', 'vol']).set_index('date')
        self.garchvol = df['vol']
        self.kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)

        degarched = returns[df.index] / self.garchvol
        self.kde.fit(np.array(degarched).reshape(-1, 1))

    def paths(self, paths, period, s0, rate, npaths):

        msg = "Some dates for the given period are missing in the model."
        assert all(day in self.garchvol.index for day in period.date_range), msg

        vols = self.garchvol.loc[period.date_range[:-1]].to_numpy().reshape(1, -1)

        paths[:, 0] = 0
        paths[:, 1:period.days] = vols * self.kde.sample(npaths * (period.days - 1)).reshape(npaths, period.days - 1)
        paths[:, :period.days] = s0 * np.exp(np.cumsum(paths[:, :period.days], axis=1))

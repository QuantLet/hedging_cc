from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class Strategy(ABC):

    def __init__(self, assets, v0, days, directory, rate=0.):

        msg = "All asset paths should have the same shape."
        assert all(asset.npaths == assets[0].npaths for asset in assets), msg

        msg = "All assets should correspond to the same period."
        assert all(asset.period == assets[0].period for asset in assets), msg

        self.v0 = v0
        self.days = days
        self.assets = assets
        self.rate = rate
        self.directory = directory

        self.period = assets[0].period
        self.npaths = assets[0].npaths

        self.strategies = None
        self.pnl = None

        self._initialized = False

    @property
    @abstractmethod
    def name(self):
        return NotImplementedError()

    @property
    @abstractmethod
    def fname(self):
        return NotImplementedError()

    @property
    @abstractmethod
    def title(self):
        return NotImplementedError()

    @abstractmethod
    def generate(self):
        return NotImplementedError()

    @property
    def df_path(self):
        return f'_output/hedges/pnl/{self.directory}/{self.fname}.csv'

    @property
    def plot_path(self):
        return f'_output/hedges/pnl/{self.directory}/{self.fname}.pdf'

    def init(self):
        """
        It takes longer to fill with nan instead of zeros, but this way we can eliminate inconsistencies.
        For example, we ensure that the option does not have price zero past maturity.
        Instead, the price past maturity should not be defined.
        """

        if not self._initialized:

            self.strategies = np.empty((len(self.assets), self.npaths, self.period.days))
            self.strategies.fill(np.nan)

            self.pnl = np.empty(self.npaths)
            self.pnl.fill(np.nan)

            self.generate()

            self._initialized = True

    def integrate(self):

        dt = 1 / 365
        self.pnl = self.v0 - np.sum([strategy[:, 0] * asset.paths[:, 0] for asset, strategy in zip(self.assets, self.strategies)], axis=0)

        for i in range(1, self.days):
            self.pnl = np.exp(self.rate * dt) * self.pnl - np.sum([(strategy[:, i] - strategy[:, i - 1]) * asset.paths[:, i] for asset, strategy in zip(self.assets, self.strategies)], axis=0)

        self.pnl *= np.exp(self.rate * dt)
        self.pnl += np.sum([(strategy[:, self.days - 1]) * asset.paths[:, self.days] for asset, strategy in zip(self.assets, self.strategies)], axis=0)

    def save(self):
        df = pd.DataFrame(self.pnl)
        os.makedirs(os.path.dirname(self.df_path), exist_ok=True)
        df.to_csv(self.df_path, index=False, header=False)
        logger.info(f"Dataframe written to {self.df_path}.")

    def plot(self):
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.hist(self.pnl / self.v0, bins=100, density=True)
        ax.set_title(self.title)
        os.makedirs(os.path.dirname(self.plot_path), exist_ok=True)
        fig.savefig(self.plot_path, transparent=True)
        logger.info(f"Plot written to {self.plot_path}.")


class OptionHedge(Strategy):

    def __init__(self, hedgeable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hedgeable = hedgeable

    @property
    def fname(self):
        return f"PNL__{self.hedgeable.underlying.model.name}__{self.hedgeable.model.name}__{self.name}__{self.period.name}__{int(self.hedgeable.strike)}__{self.days}__{self.npaths}"

    @property
    def title(self):
        return f'{self.hedgeable.underlying.model.name} hedged with {self.hedgeable.model.name} {self.name} {self.period.name}, STRIKE = {int(self.hedgeable.strike)}, MATURITY = {self.days}D, V0 = {self.v0}, PATHS = {self.npaths}'

    def hedge(self):
        self.integrate()
        self.pnl -= self.hedgeable.payoff(self.assets[0].paths[:, self.days - 1])

    def execute(self):

        for asset in self.assets:
            asset.init()

        self.init()
        self.hedge()
        self.save()
        self.plot()

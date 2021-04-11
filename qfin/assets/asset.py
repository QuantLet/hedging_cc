import os
import logging
from abc import ABC, abstractmethod

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from qfin.utils import sizeof_fmt

logger = logging.getLogger(__name__)


class Asset(ABC):

    def __init__(self, model, period, npaths, caching=True):
        self.model = model
        self.period = period
        self.npaths = npaths
        self.caching = caching

        self.paths = None

        self._initialized = False

    @property
    @abstractmethod
    def asset_name(self):
        pass

    @property
    def fname(self):
        return f"PATHS__{self.model.name}__{self.period.name}__{self.asset_name}__{self.npaths}"

    @property
    def df_path(self):
        return f"_output/hedges/paths/{self.fname}.csv"

    @property
    def plot_path(self):
        return f"_output/hedges/paths/{self.fname}.pdf"

    @abstractmethod
    def generate(self):
        pass

    def init(self):

        if not self._initialized:

            self.paths = np.empty((self.npaths, self.period.days))
            self.paths.fill(np.nan)

            if self.caching and os.path.exists(self.df_path):
                logger.info(f"Loading {self.fname} paths from file.")
                self.paths[:] = pd.read_csv(self.df_path, header=None).to_numpy()

            else:
                logger.info(f"Generating {self.fname} paths.")
                self.generate()

                if self.caching:
                    self.save()
                    self.plot()

            self._initialized = True

    def save(self):
        os.makedirs(os.path.dirname(self.df_path), exist_ok=True)
        df = pd.DataFrame(self.paths)
        df.to_csv(self.df_path, index=False, header=False)
        logger.info(f"{self.npaths} spot paths ({sizeof_fmt(self.paths.nbytes)}) written to {self.df_path}.")

    def plot(self):

        # plot paths
        title = f"{self.model.name} {self.period.name} {self.asset_name} PATHS"
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(self.period.date_range, self.paths[:1000].T)
        ax.set_xlabel('time step')
        ax.set_ylabel('asset price process')
        ax.set_title(title)

        # save plot
        os.makedirs(os.path.dirname(self.plot_path), exist_ok=True)
        fig.savefig(self.plot_path, transparent=True)
        plt.close(fig)
        logger.info(f"Plot written to {self.plot_path}.")

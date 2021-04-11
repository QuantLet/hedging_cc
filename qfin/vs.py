import pandas as pd
import datetime
import numpy as np
import re

from qfin.utils import bs_iv, bs_call_delta, bs_put_delta


def from_tickers(path, date):

    df = pd.read_csv(path)

    # filter data
    df = df[(df['best_ask_price'] != 0) & (df['best_ask_amount'] != 0) &
            (df['best_bid_price'] != 0) & (df['best_bid_amount'] != 0)]

    df = df[~df['instrument_name'].str.startswith('ETH')]
    df = df[df['instrument_name'].str.endswith('C') | df['instrument_name'].str.endswith('P')]
    df = df.loc[[odf.index[0] for instrument_name, odf in df.groupby('instrument_name')]]

    df = df.reset_index(drop=True)

    # extract option data
    cols = pd.DataFrame([(maturity, float(strike), option_type) for btc, maturity, strike, option_type in
                         df['instrument_name'].apply(lambda s: re.match(r"^(\w+)-(\d\d?\w\w\w\d\d)-(\d+)-(\w)$", s).groups())],
                        columns=['maturity', 'strike', 'option_type'])

    df[['maturity', 'strike', 'option_type']] = cols

    # replace maturity column with numerical value in years
    def ttm(maturity):

        today = datetime.datetime.strptime(date, '%Y%m%d')
        maturity = datetime.datetime.strptime(maturity, '%d%b%y')

        assert maturity >= today

        if today == maturity:
            return 1 / 365.25

        # depends on the day count conventions
        return (maturity - today).days / 365.25

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
            idxs += [odf.index[1] if odf['volume'].iloc[1] > odf['volume'].iloc[0] else odf.index[0]]

    df = df.loc[idxs]

    df = df[df['volume'] > 100]

    df['moneyness'] = df['strike'] / df['spot']
    df = df[(0.5 <= df['moneyness']) & (df['moneyness'] <= 2)]

    # filter between 25P and 25C
    df['call_delta'] = np.vectorize(bs_call_delta)(df['ttm'], df['moneyness'], 1., 0., df['iv'])
    df['put_delta'] = -np.vectorize(bs_put_delta)(df['ttm'], df['moneyness'], 1., 0., df['iv'])
    df = df[(df['call_delta'] > 0.25) & (df['put_delta'] > 0.25)]

    df = df[['ttm', 'moneyness', 'iv', 'volume']]
    df = df.set_index(['ttm', 'moneyness']).sort_index()

    return df


def from_quotes(path, date, rate=0):

    df = pd.read_csv(path)
    df = df.rename(columns={'mark_price': 'price', 'underlying_price': 'spot'})
    df['price'] = df['price'] * df['spot']

    # filter columns
    df = df[['spot', 'maturity', 'strike', 'price', 'type', 'volume']]

    # set zero rate to 0
    df['rate'] = rate

    # replace maturity column with numerical value in years
    def ttm(maturity):

        today = datetime.datetime.strptime(date, '%Y%m%d')
        maturity = datetime.datetime.strptime(maturity, '%d%b%y')

        assert maturity >= today

        if today == maturity:
            return 1 / 365.25

        # depends on the day count conventions
        return (maturity - today).days / 365.25

    df['ttm'] = df['maturity'].apply(ttm)

    df['iv'] = [bs_iv(ttm, strike, spot, 0, price, otype)
                for ttm, strike, spot, rate, price, otype in
                df[['ttm', 'strike', 'spot', 'rate', 'price', 'type']].values]

    df = df.drop(columns=['maturity'])

    # keep only instruments with volume > 0
    df = df[df['volume'] > 0]

    # keep only instruments where IV could be calculated
    df = df[~np.isnan(df['iv'])]

    df = df.drop_duplicates(subset=['ttm', 'strike'], keep='first')
    df['moneyness'] = df['strike'] / df['spot']

    df = df[['ttm', 'moneyness', 'iv', 'volume']]
    df = df.set_index(['ttm', 'moneyness']).sort_index()

    return df

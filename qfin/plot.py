import math
import matplotlib.pyplot as plt


def plot_smiles(*vss):

    colors = 'b', 'r', 'g', 'c', 'm', 'y'

    msg = "Please define more colors to plot volatility surfaces."
    assert len(vss) <= len(colors), msg

    dfs = [df.rename(columns={"iv": f"iv{idx}", "volume": f"volume{idx}"}) for idx, df in enumerate(vss)]
    dfs = dfs[0].join(dfs[1:])
    groups = dfs.groupby('ttm')

    ncols = 3
    nrows = math.ceil(len(groups) / ncols)
    fig = plt.figure(constrained_layout=False, figsize=(12, 3 * nrows))
    gs1 = fig.add_gridspec(ncols=ncols, nrows=nrows, left=0.05, right=0.95, hspace=0.4, wspace=0.25)

    for idx, (ttm, df) in enumerate(groups):
        ax = fig.add_subplot(gs1[idx])
        strikes = df.reset_index()['strike']
        ax.set_title(f"TTM = {ttm:.4f}")
        ivss = df[[f"iv{idx}" for idx in range(len(vss))]]

        for column, color in zip(ivss.columns, colors):
            ax.plot(strikes, ivss[column], marker='o', c=color)

    return fig


def plot_points(vs):
    xs = [x[1] for x in vs.index]
    ys = [x[0] for x in vs.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, c='b', marker='x')

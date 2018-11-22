import pandas
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import os
import logging
logger = logging.getLogger(__name__)


def get_annual_totals(filename):

    rainfall = {}
    sunshine = {}

    with pandas.HDFStore(filename, mode='r') as store:

        for key in store.keys():

            if 'hly' in key:
                continue  # Skip hourly data

            if 'scen' not in key:
                continue

            df = store[key]

            # Massage the dates into a timeseries index.
            # NOTE: UKCP09 uses notional years starting in 3000.
            # pandas doesn't like dates this far in the future say we move them to 2000
            dates = {k: df.index.get_level_values(k) for k in ('year', 'month', 'day')}
            dates['year'] = dates['year'] - 1000
            dates = pandas.to_datetime(dates)
            df.index = dates

            # Total annual rainfall & sunshine hours
            rainfall[key] = df['precip_dtotal'].resample('A').sum()
            sunshine[key] = df['sunshine_dtotal'].resample('A').sum()

    rainfall = pandas.DataFrame(rainfall)
    sunshine = pandas.DataFrame(sunshine)

    return rainfall, sunshine


def plot_rainfall_sunshine_correlation(rainfall, sunshine, figure_filename=None):

    q = np.linspace(0, 1)

    rainfall_quant = rainfall.quantile(q)
    sunshine_quant = sunshine.quantile(q)

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(6, 12), sharey=True, sharex=True)

    ax1.plot(rainfall_quant, sunshine_quant, color='grey', alpha=0.5)
    ax1.grid()
    ax1.set_ylabel('Annual sunshine total [hours]')

    gamma = 0.5
    ax2.hist2d(rainfall_quant.values.flatten(), sunshine_quant.values.flatten(), bins=50, norm=mcolors.PowerNorm(gamma))
    ax2.grid()
    ax2.set_xlabel('Annual rainfall total [mm]')
    ax2.set_ylabel('Annual sunshine total [hours]')
    plt.tight_layout()
    if figure_filename is not None:
        fig.savefig(figure_filename)
    return fig


def subsample_selection(rainfall, sunshine, sample_sizes, sample_tries=200, figure_filename=None):
    q = np.linspace(0, 1)

    rainfall_quant = rainfall.quantile(q)
    sunshine_quant = sunshine.quantile(q)

    Hfull, xedges, yedges = np.histogram2d(rainfall_quant.values.flatten(),
                                           sunshine_quant.values.flatten(),
                                           bins=50, density=True)

    def find_sample(size, tries=10):
        """Perform Monte Carlo selection to find the best fitting subsample of the 2D density function. """
        best_fit = np.inf
        best_indices = None
        for i in range(tries):

            sample_indices = sorted(np.random.choice(rainfall_quant.shape[1], size=size))
            assert len(sample_indices) == size
            rainfall_sample = rainfall_quant.iloc[:, sample_indices]
            sunshine_sample = sunshine_quant.iloc[:, sample_indices]

            H, _, _ = np.histogram2d(rainfall_sample.values.flatten(),
                                     sunshine_sample.values.flatten(),
                                     bins=(xedges, yedges), density=True)

            fit = np.sqrt(np.sum((H - Hfull) ** 2))

            if fit < best_fit:
                best_fit = fit
                best_indices = sample_indices

        return best_indices, best_fit

    sample_indices = {}
    sample_fit = []

    for size in sample_sizes:
        indices, fit = find_sample(size, tries=sample_tries)
        sample_indices[size] = indices

        sample_fit.append(fit)

    if figure_filename is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(sample_sizes, sample_fit, '-o')
        ax.grid()
        ax.set_xlabel('Sub-sample size')
        ax.set_ylabel('Fit error');
        fig.savefig(figure_filename)

    return sample_indices


def save_subamples(filename, rainfall, sample_indices, output_filename):
    for size, indices in sample_indices.items():
        rainfall_sample = rainfall.iloc[:, indices]

        base, ext = os.path.splitext(output_filename)
        out = base + f'_sub{size:03d}' + ext
        
        logger.info(f'Writing sub-sampled of size {size} output to: {out}')
        with pandas.HDFStore(filename, mode='r') as in_store, \
                pandas.HDFStore(out, mode='w', complib='zlib', complevel=9) as out_store:

            for c in rainfall_sample.columns:
                out_store[c] = in_store[c]
                # Also copy the cntr data
                c_cntr = c.replace('scen', 'cntr')
                out_store[c_cntr] = in_store[c_cntr]

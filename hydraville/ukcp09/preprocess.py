import zipfile
import os
import pandas
import logging
logger = logging.getLogger(__name__)


def load_column_headers(filename):
    with zipfile.ZipFile(filename) as zf:
        with zf.open('column_headers_daily.csv') as fh:
            headers = pandas.read_csv(fh, header=None, skipinitialspace=True).T
            headers.columns = ['headers', 'units']
            return headers


def load_ukcp09_zipfile(filename, columns):
    logger.info('Loading data from archive: "{}"'.format(filename))
    with zipfile.ZipFile(filename) as zf:
        # Now read each individual CSV
        for fname in zf.namelist():
            name, ext = os.path.splitext(fname)
            if ext != '.csv' or not name.startswith('r_'):
                continue  # skip anything that's not a CSV or data file.

            logger.info('Open archive file: "{}"'.format(fname))
            with zf.open(fname) as fh:
                df = pandas.read_csv(fh, header=None)
                df.columns = columns
                df.name = name
                df.set_index(['year', 'month', 'day'], inplace=True)
                yield df


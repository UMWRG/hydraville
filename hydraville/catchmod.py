import numpy as np
import pandas
import os
from pycatchmod.io.json import catchment_from_json
from collections import defaultdict
from . import CATCHMOD_DIR
import logging
logger = logging.getLogger(__name__)


def generate_catchmod_models():

    for filename in sorted(os.listdir(CATCHMOD_DIR)):
        base, ext = os.path.splitext(filename)
        if ext != '.json':
            continue

        yield catchment_from_json(os.path.join(CATCHMOD_DIR, filename))


def _run_catchmod(C, rainfall, temperature, dates):
    """Convenience function for running catchmod
    """

    C.reset()

    # number of scenarios
    N = C.subcatchments[0].soil_store.initial_upper_deficit.shape[0]
    assert(rainfall.shape[1] == N)

    # input timesteps
    M = rainfall.shape[0]
    # output timesteps
    if dates is not None:
        M2 = len(dates)
    else:
        M2 = M

    perc = np.zeros((len(C.subcatchments), N))
    outflow = np.zeros_like(perc)

    pet = np.zeros(N)
    flow = np.zeros([M2, N])
    flows = np.zeros([M2, len(C.subcatchments)])

    # TODO: add option to enable/disable extra leap days

    i = 0
    for j in range(M2):
        date = dates[j]
        if M != M2:
            if date.month == 2 and date.day == 29:
                # input data is missing leap days, use previous day
                i -= 1

        r = rainfall[i, ...].reshape(N).astype(np.float64)
        t = temperature[i, ...].reshape(N).astype(np.float64)

        C.step(date.dayofyear, r, t, pet, perc, outflow)

        flows[j, ...] = outflow[:, 0]
        flow[j, ...] = outflow.sum(axis=0).reshape(rainfall.shape[1:])

        i += 1

    return flow


def run_ukcp09_weather(models, weather_filename, flows_filename, max_scenarios_to_process=None):

    with pandas.HDFStore(weather_filename, mode='r') as store, \
         pandas.HDFStore(flows_filename, mode='w', complib='zlib', complevel=9) as out_store:
            # TODO make this parallel using multiprocessing?
            i = 0
            for key in store.keys():
                if 'hly' in key:
                    continue  # Skip hourly data

                logger.info('Predicting flow for scenario: "{}"'.format(key))
                df = store[key]

                # Massage the dates into a timeseries index.
                # NOTE: UKCP09 uses notional years starting in 3000.
                # pandas doesn't like dates this far in the future say we move them to 2000
                dates = {k: df.index.get_level_values(k) for k in ('year', 'month', 'day')}
                dates['year'] = dates['year'] - 1000
                dates = pandas.to_datetime(dates)

                rainfall = df['precip_dtotal'].values[:, np.newaxis]
                pet = df['pet_dmean'].values[:, np.newaxis]

                flows = {}
                for model in models:
                    flow = _run_catchmod(model, rainfall, pet, dates)
                    flow /= 1e3  # Convert to Mm3/day
                    flows[model.name] = flow[:, 0]

                flows = pandas.DataFrame(data=flows, index=dates)
                flows.name = key
                out_store[key] = flows

                i += 1
                if max_scenarios_to_process is not None and i >= max_scenarios_to_process:
                    break


def concat_flows_by_catchment(flows_filename, flows_by_catchment_filename):

    with pandas.HDFStore(flows_filename, mode='r') as store:

            data = defaultdict(lambda: defaultdict(dict))

            i = 0
            for key in store.keys():
                if 'hly' in key:
                    continue  # Skip hourly data

                df = store[key]

                for col in df.columns:
                    if 'cntr' in key:
                        scenario = 'cntr'
                    else:
                        scenario = 'scen'
                    data[col][scenario][key.replace("/","")] = df[col]

    with pandas.HDFStore(flows_by_catchment_filename, mode='w', complib='zlib', complevel=9) as out_store:

        for col in data.keys():
            for scen in data[col].keys():
                logger.info(f'Writing combined flow for catchment-scenario: "{col}_{scen}"')
                df = pandas.concat(data[col][scen], axis=1)
                out_store[f"{col}_{scen}"] = df

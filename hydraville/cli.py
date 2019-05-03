import click
import os
import pandas
from pywr.model import Model
from pywr.recorders import TablesRecorder
from pywr.recorders.progress import ProgressRecorder
from matplotlib    import pyplot   as plt
import tables
import numpy as np
import json
import random
from .model import make_model
from .moea import PyretoJSONPlatypusWrapper
from . import ukcp09, catchmod

import logging
logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


DMU_OPTIONS = {
    'static': {},
    'scheduled': {},
}

MODEL_OPTIONS = {
    'water-simple': {
        'template': 'model_template.json',
        'networks': [
            'water'
        ]
    },
    'energy-simple': {
        'template': 'model_template.json',
        'networks': [
            'energy'
        ]
    },
    'water-energy': {
        'template': 'model_template.json',
        'networks': [
            'water',
            'energy',
            'water_energy'
        ]
    },
}

# TODO make this user configurable
SAMPLE_SIZES = (10, 25, 50, 100, 200)


def _create_model_from_context(context):

    parameter_templates = context.get('parameter_templates', None)
    parameter_template_options = context.get('parameter_template_options', None)
    ensembles = context.get('ensembles')
    seed = context.get('seed')
    model_options = context.get('model_options')

    data = make_model(ensembles=ensembles, seed=seed,
                      parameter_templates=parameter_templates,
                      parameter_template_options=parameter_template_options,
                      **model_options
                      )

    # Validate the data by loading in Pywr
    out = context['out']

    Model.loads(json.dumps(data), path=os.path.dirname(out))

    with open(out, 'w') as fh:
        json.dump(data, fh, indent=2)


@cli.command()
@click.argument('out', type=click.Path(file_okay=True, dir_okay=False, exists=False))
@click.option('-m', '--model', type=click.Choice(MODEL_OPTIONS.keys()), default='water-simple')
@click.option('-d', '--dmu', type=click.Choice(DMU_OPTIONS.keys()), default=None)
@click.option('-e', '--ensembles', type=int, default=None)
@click.option('-s', '--seed', type=int, default=None)
@click.pass_obj
def create(obj, out, model, dmu, ensembles, seed):
    """ Create the Pywr JSON for a particular model configuration. """
    obj['out'] = out
    obj['model_options'] = MODEL_OPTIONS[model]

    if dmu is not None:
        raise NotImplementedError('Decision making under uncertainty configuration is not supported yet.')
        # dmu_options = DMU_OPTIONS[dmu]
        # obj['parameter_templates'] = dmu_options.get('templates', [f'{dmu}.json'])
        # obj['parameter_template_options'] = dmu_options.get('options', {})

    obj['ensembles'] = ensembles
    obj['seed'] = seed
    _create_model_from_context(obj)


@cli.command()
@click.argument('filename', type=click.Path(file_okay=True, dir_okay=False, exists=True))
def run(filename):
    """ Run the Pywr model. """
    logger.info('Loading model from file: "{}"'.format(filename))
    model = Model.load(filename)

    ProgressRecorder(model)

    base, ext = os.path.splitext(filename)
    TablesRecorder(model, f"{base}_outputs.h5", parameters=[p for p in model.parameters if p.name is not None])

    logger.info('Starting model run.')
    ret = model.run()
    logger.info(ret)

    # Save the metrics
    metrics = {}
    metrics_aggregated = {}
    for rec in model.recorders:
        try:
            metrics[rec.name] = np.array(rec.values())
            metrics_aggregated[rec.name] = np.array(rec.aggregated_value())
        except NotImplementedError:
            pass
    metrics = pandas.DataFrame(metrics).T
    metrics_aggregated = pandas.Series(metrics_aggregated)

    writer = pandas.ExcelWriter(f"{base}_metrics.xlsx")
    metrics.to_excel(writer, 'scenarios')
    metrics_aggregated.to_excel(writer, 'aggregated')
    writer.save()

    # Write dataframes
    store = pandas.HDFStore(f"{base}_dataframes.h5", mode='w')
    for rec in model.recorders:
        if hasattr(rec, 'to_dataframe'):
            df = rec.to_dataframe()
            store[rec.name] = df

        try:
            values = np.array(rec.values())
        except NotImplementedError:
            pass
        else:
            store[f'{rec.name}_values'] = pandas.Series(values)

    store.close()

@cli.command()
@click.argument('filename', type=click.Path(file_okay=True, dir_okay=False, exists=True))
def postprocess(filename):

    plt.rcParams.update({'figure.max_open_warning': 0})

    directory, base_ext = os.path.split(filename)
    output_directory = os.path.join(directory, 'outputs', 'figures')
  
    with tables.open_file(filename) as fl:
        tbl = fl._get_node('/time')
        date_index = pandas.to_datetime({k: tbl.col(k) for k in ('year','month','day')})
        data = {}
        for ca in fl.walk_nodes('/', 'CArray'):
            data[ca._v_name] = pandas.DataFrame(ca.read(), index=date_index)

    df = pandas.concat(data, axis=1)
    
    FLOW_UNITS   = 'Hm^3'
    ENERGY_UNITS = 'MWh'
    PERCENTILES  = np.linspace(0,100)

    Y_LABEL_MAP = {
        'catchment1'   : f'Catchment 1 inflow [${FLOW_UNITS}/day$]',
        'catchment2'   : f'Catchment 2 inflow [${FLOW_UNITS}/day$]',
        'catchment3'   : f'Catchment 3 inflow [${FLOW_UNITS}/day$]',
        'catchment4'   : f'Catchment 4 inflow [${FLOW_UNITS}/day$]',
        'catchment5'   : f'Catchment 5 inflow [${FLOW_UNITS}/day$]',
#        'irrigation1'  : f'Irrigation Sector 1 [${FLOW_UNITS}/day$]',
#        'irrigation2'  : f'Irrigation Sector 2 [${FLOW_UNITS}/day$]',
#        'water_supply1': f'Water Supply [${FLOW_UNITS}/day$]',
#        'reservoir1'   : f'Reservoir 1 Storage [${FLOW_UNITS}$]',
#        'reservoir2'   : f'Reservoir 2 Storage [${FLOW_UNITS}$]',
#        'turbine_energy_generation1': f'Hydropower 1 [${ENERGY_UNITS}$]',
#        'turbine_energy_generation2': f'Hydropower 2 [${ENERGY_UNITS}$]',
#        'thermal1'     : f'Thermal plant 1 [${ENERGY_UNITS}$]',
#        'thermal2'     : f'Thermal plant 2 [${ENERGY_UNITS}$]',
#        'thermal3'     : f'Thermal plant 3 [${ENERGY_UNITS}$]',
    }

# TODO check this and make automatic with the labels
#    dfkeys         = list(df.columns.levels[0])
#    dickeys        = list(Y_LABEL_MAP.keys())
#    keys_to_remove = [item for item in dfkeys if item not in dickeys]
#    df2 = df.drop(columns=keys_to_revove, axis=1)

    nrows = len(df.columns.levels[0])

    for node in df.columns.levels[0]:
        fig, ax = plt.subplots(nrows=1, figsize=(12,4))
        df[node].plot(ax=ax, color='grey', alpha=0.5, legend=False)
        
        try: 
            label = Y_LABEL_MAP[node]
        except KeyError:
            label = node
        ax.set_ylabel(label)
        plt.tight_layout()
        fig.savefig(os.path.join(output_directory, f'{node}.png'), dpi = 300)
        fig.savefig(os.path.join(output_directory, f'{node}.eps'))


@cli.command()
@click.argument('filename', type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.option('-s', '--seed', type=int, default=None)
@click.option('-p', '--num-cpus', type=int, default=None)
@click.option('-n', '--max-nfe', type=int, default=1000)
@click.option('--pop-size', type=int, default=50)
@click.option('-a', '--algorithm', type=click.Choice(['NSGAII', 'NSGAIII', 'EpsMOEA']), default='NSGAII')
@click.option('-e', '--epsilons', multiple=True, type=float, default=(0.05, ))
@click.option('--divisions-outer', type=int, default=12)
def search(filename, seed, num_cpus, max_nfe, pop_size, algorithm, epsilons, divisions_outer):
    import platypus
    logger.info('Loading model from file: "{}"'.format(filename))
    directory, _ = os.path.split(filename)
    output_directory = os.path.join(directory, 'outputs')

    if algorithm == 'NSGAII':
        algorithm_klass = platypus.NSGAII
        algorithm_kwargs = {'population_size': pop_size}
    elif algorithm == 'NSGAIII':
        algorithm_klass = platypus.NSGAIII
        algorithm_kwargs = {'divisions_outer': divisions_outer}
    elif algorithm == 'EpsMOEA':
        algorithm_klass = platypus.EpsMOEA
        algorithm_kwargs = {'population_size': pop_size, 'epsilons': epsilons}
    else:
        raise RuntimeError('Algorithm "{}" not supported.'.format(algorithm))

    if seed is None:
        seed = random.randrange(sys.maxsize)

    wrapper = PyretoJSONPlatypusWrapper(filename, search_data={'algorithm': algorithm, 'seed': seed,
                                                               'user_metadata':algorithm_kwargs},
                                        output_directory=output_directory)
    if seed is not None:
        random.seed(seed)

    logger.info('Starting model search.')
    if num_cpus is None:
        evaluator_klass = platypus.MapEvaluator
        evaluator_args = ()
    else:
        evaluator_klass = platypus.ProcessPoolEvaluator
        evaluator_args = (num_cpus, )
    
    with evaluator_klass(*evaluator_args) as evaluator:
        algorithm = algorithm_klass(wrapper.problem, evaluator=evaluator, **algorithm_kwargs)
        algorithm.run(max_nfe)


@cli.group('ukcp09')
def ukcp09_grp():
    pass


@ukcp09_grp.command('preprocess')
@click.argument('directory', type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.argument('out', type=click.Path(file_okay=True, dir_okay=False))
def ukcp09_preprocess(directory, out):
    logger.info('Starting archive preprocessing of folder: "{}"'.format(directory))

    names_to_load = {}
    for name in os.listdir(directory):
        if os.path.splitext(name)[-1] != '.zip':
            continue
        archive_num = int(os.path.splitext(name)[0].split('_')[-1])
        names_to_load[archive_num] = name
    logger.info('Archive contains {} files.'.format(len(names_to_load)))

    logger.info('Loading column headers for archive.')
    # Headers are always in the last file
    last_arhive_num = sorted(names_to_load.keys())[-1]
    headers = ukcp09.load_column_headers(os.path.join(directory, names_to_load[last_arhive_num]))

    logger.info('Creating empty HDF store: "{}"'.format(out))
    with pandas.HDFStore(out, mode='w', complib='zlib', complevel=9) as store:

        for archive_name, name in sorted(names_to_load.items()):
            for df in ukcp09.load_ukcp09_zipfile(os.path.join(directory, name), headers['headers']):
                # Save each dataframe to the store
                store[df.name] = df

    logger.info('Complete!')


@ukcp09_grp.command('subsample')
@click.argument('filename', type=click.Path(file_okay=True, dir_okay=False))
@click.argument('out', type=click.Path(file_okay=True, dir_okay=False))
def ukcp09_subsample(filename, out):

    logger.info('Computing annual rainfall and sunshine totals ...')
    rainfall, sunshine = ukcp09.get_annual_totals(filename)

    logger.info('Performing Monte Carlo sub-sampling ...')

    subsamples = ukcp09.subsample_selection(rainfall, sunshine, SAMPLE_SIZES)

    logger.info('Saving subsampled weather data ...')
    ukcp09.save_subamples(filename, rainfall, subsamples, out)

    logger.info('Complete!')


@cli.group('catchmod')
def catchmod_grp():
    pass


@catchmod_grp.command('run')
@click.argument('weather', type=click.Path(file_okay=True, dir_okay=False))
@click.argument('flows', type=click.Path(file_okay=True, dir_okay=False))
@click.argument('flows_by_catchment', type=click.Path(file_okay=True, dir_okay=False))
def catchmod_run(weather, flows, flows_by_catchment):

    models = list(catchmod.generate_catchmod_models())
    logger.info(f'Successfully loaded {len(models)} pycatchmod models.')
    # Change thise to run different sub-samples
    for size in SAMPLE_SIZES:
        logger.info(f'Starting flow generation for sub-sample of size {size}.')

        base, ext = os.path.splitext(weather)
        weather_ss = base + f'_sub{size:03d}' + ext

        base, ext = os.path.splitext(flows)
        flows_ss = base + f'_sub{size:03d}' + ext

        base, ext = os.path.splitext(flows_by_catchment)
        flows_by_catchment_ss = base + f'_sub{size:03d}' + ext

        catchmod.run_ukcp09_weather(models, weather_ss, flows_ss)

        logger.info('Concatenating flows by catchment.')
        catchmod.concat_flows_by_catchment(flows_ss, flows_by_catchment_ss)

        logger.info(f'Finished processing sub-sample "{size}"')

    logger.info('Complete!')


def start_cli():
    """ Start the command line interface. """
    from . import logger
    import sys
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(ch)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    # Also log pywr messages
    pywr_logger = logging.getLogger('pywr')
    pywr_logger.setLevel(logging.INFO)
    pywr_logger.addHandler(ch)
    cli(obj={})

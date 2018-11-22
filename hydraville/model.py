from jinja2 import Template
import numpy as np
import json
import os
from . import DATA_DIR, TEMPLATE_DIR
import logging
logger = logging.getLogger(__name__)


def make_model(template, networks, ensembles=None, seed=None, timestepper_kwargs=None, **kwargs):
    """ Create a model from a skeleton template and several other networks. """
    filename = os.path.join(DATA_DIR, template)

    with open(filename) as fh:
        data = json.load(fh)

    for network in networks:
        network_filename = f'{network}_network.json'
        network_data = make_network(network_filename, **kwargs)
        for key, value in network_data.items():
            if isinstance(data[key], list):
                data[key].extend(value)
            elif isinstance(data[key], dict):
                data[key].update(value)
            else:
                raise ValueError(f'Unable to update the template key "{key}" from network "{network}".')

    if ensembles is not None:
        apply_random_scenario_sample(data, ensembles, seed=seed)

    if timestepper_kwargs is not None:
        update_timestepper(data, **timestepper_kwargs)

    return data


def make_network(model, parameter_templates=None, parameter_template_options=None):
    """ Load the data for a particular sub-network. """
    filename = os.path.join(DATA_DIR, model)

    logger.info('Loading the model from: "{}"'.format(filename))

    if parameter_templates is not None:
        parameter_data = load_templates(parameter_templates, parameter_template_options)
    else:
        parameter_data = {}

    with open(filename) as fh:
        template = Template(fh.read())
        data = template.render(parameters=parameter_data)
        data = json.loads(data)

    logger.info('Finished modifying the model data.')
    return data


def apply_random_scenario_sample(model_data, ensembles, seed=None):

    if seed is not None:
        np.random.seed(seed)

    samples = []
    for scenario in model_data['scenarios']:
        samples.append(np.random.randint(0, scenario['size'], ensembles))

    samples = np.stack(samples).T
    model_data['scenario_combinations'] = samples.tolist()


def load_templates(templates, options):

    merged_data = {}
    for filename in templates:
        with open(os.path.join(TEMPLATE_DIR, filename)) as fh:
            template = Template(fh.read())
            data = template.render(**options)
            data = json.loads(data)
            common_keys = set(merged_data.keys()).intersection(set(data.keys()))
            if len(common_keys) > 0:
                raise ValueError('Conflicting keys in templates: {}'.format(', '.join(common_keys)))
            merged_data.update(data)
    return merged_data


def update_timestepper(data, **kwargs):
    """ Update the timestepper key values in Pywr JSON data. """
    for key, value in kwargs.items():
        if key not in ('start', 'end', 'timestep'):
            raise ValueError('Key "{}" not valid for "timestepper" section of JSON.'.format(key))
        logger.info('Setting timestepper {} to: "{}"'.format(key, value))
        data['timestepper'][key] = value


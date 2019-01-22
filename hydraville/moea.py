import uuid
from pywr.optimisation.platypus import PlatypusWrapper
import gzip
from bson import Binary, Code
from bson.json_util import dumps
import os
import datetime
import logging
logger = logging.getLogger(__name__)


class PyretoJSONPlatypusWrapper(PlatypusWrapper):
    def __init__(self, *args, **kwargs):
        self.output_directory = kwargs.pop('output_directory', 'outputs')
        search_data = kwargs.pop('search_data', {})
        super().__init__(*args, **kwargs)

        search_data.update({
            'name': self.model.metadata['title'],
        })
        search_data.update(**search_data)
        self._create_new_search_json(search_data)
        self.created_at = None

    def customise_model(self, model):
        model.solver.retry_solve = True

    @property
    def output_subdirectory(self):
        path = os.path.join(self.output_directory, self.uid)
        os.makedirs(path, exist_ok=True)
        return path

    def evaluate(self, solution):
        self.created_at = datetime.datetime.now()
        ret = super().evaluate(solution)
        self._create_new_individual_json()
        return ret

    def _create_new_search_json(self, search_data):

        if 'started_at' not in search_data:
            search_data['started_at'] = datetime.datetime.now().isoformat()

        fn = os.path.join(self.output_subdirectory, 'search.bson')

        with open(fn, mode='w') as fh:
            fh.write(dumps(search_data))

    def _create_new_individual_json(self):
        import time

        uid = uuid.uuid4().hex
        fn = os.path.join(self.output_subdirectory, 'i'+uid+'.bson.gz')

        logger.info('Saving individual to PyretoDB to JSON: {}'.format(fn))
        t0 = time.time()

        evaluated_at = datetime.datetime.now()
        # TODO runtime statistics
        individual = dict(
            variables=list(self._generate_variable_documents()),
            metrics=list(self._generate_metric_documents()),
            created_at=self.created_at.isoformat(),
            evaluated_at=evaluated_at.isoformat(),
        )

        with gzip.open(fn, mode='wt') as fh:
            fh.write(dumps(individual))
        logger.info('Save complete in {:.2f}s'.format(time.time() - t0))

    def _generate_variable_documents(self):

        for variable in self.model.variables:

            if variable.double_size > 0:
                upper = variable.get_double_upper_bounds()
                lower = variable.get_double_lower_bounds()
                values = variable.get_double_variables()
                for i in range(variable.double_size):
                    yield dict(name='{}[d{}]'.format(variable.name, i), value=float(values[i]),
                                             upper_bounds=float(upper[i]), lower_bounds=float(lower[i]))

            if variable.integer_size > 0:
                upper = variable.get_integer_upper_bounds()
                lower = variable.get_integer_lower_bounds()
                values = variable.get_integer_variables()
                for i in range(variable.integer_size):
                    yield dict(name='{}[i{}]'.format(variable.name, i),
                               value=int(values[i]), upper_bounds=int(upper[i]), lower_bounds=int(lower[i]))

    def _generate_metric_documents(self):

        for recorder in self.model.recorders:

            try:
                value = float(recorder.aggregated_value())
            except NotImplementedError:
                value = None

            try:
                df = recorder.to_dataframe()
                df = Binary(df.to_msgpack())
            except AttributeError:
                df = None

            if value is not None or df is not None:
                yield dict(name=recorder.name, value=value,
                                       dataframe=df,
                                       objective=recorder.is_objective is not None,
                                       constraint=recorder.is_constraint,
                                       minimise=recorder.is_objective == 'minimise')





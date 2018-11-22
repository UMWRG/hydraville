from pywr.parameters import Parameter, load_dataframe, load_parameter
from pywr.parameters._parameters import wrap_const
import numpy as np
from scipy.interpolate import interp1d
from scipy import stats


class DistributionParameter(Parameter):
    """ Parameter based on `scipy.stats` distributions.

    Parameters
    ==========
    distribution : instance of `scipy.stats` distribution.
        The distribution to use for generating the parameter values.
    scenario : `pywr.core.Scenario`
    sampling_method : str
    random_state : None or int
    """
    def __init__(self, model, distribution, scenario, *args, sampling_method='random', random_state=None,
                 lower_quantile=0.01, upper_quantile=0.99, **kwargs):

        super().__init__(model, *args, **kwargs)

        self.distribution = distribution
        self.scenario = scenario
        self.sampling_method = sampling_method
        self.random_state = random_state
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        # internal variable for storing the sampled variables for each scenario
        self._values = None
        self._scenario_index = None

    def setup(self):
        super().setup()

        if self.scenario is not None:
            self._scenario_index = self.model.scenarios.get_scenario_index(self.scenario)

    def reset(self):

        nscenarios = self.scenario.size

        if self.sampling_method == 'random':
            self._values = self.distribution.rvs(size=nscenarios, random_state=self.random_state)
        elif self.sampling_method == 'quantiles':
            # Linear distribution of quantiles
            quantiles = np.linspace(self.lower_quantile, self.upper_quantile, nscenarios)
            # Generate values
            self._values = self.distribution.ppf(quantiles)
        else:
            raise ValueError('Sampling method "{}" not recognised.'.format(self.sampling_method))

    def value(self, ts, scenario_index):
        return self._values[scenario_index.indices[self._scenario_index]]

    @classmethod
    def load(cls, model, data):
        scenario = data.pop('scenario', None)
        if scenario is not None:
            scenario = model.scenarios[scenario]

        distribution_data = data.pop('distribution')
        distribution_name = distribution_data.pop('name')
        distribution = getattr(stats, distribution_name)(**distribution_data)

        return cls(model, distribution=distribution, scenario=scenario, **data)
DistributionParameter.register()


class AnnualInterpolationParameter(Parameter):
    def __init__(self, model, parameters, *args, interp_kwargs=None, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.parameters = parameters
        for param in parameters.values():
            self.children.add(param)

        self.interp = None
        default_interp_kwargs = dict(kind='linear', bounds_error=True)
        if interp_kwargs is not None:
            # Overwrite or add to defaults with given values
            default_interp_kwargs.update(interp_kwargs)
        self.interp_kwargs = default_interp_kwargs

    def _value_to_interpolate(self, ts, scenario_index):
        raise NotImplementedError()

    def value(self, ts, scenario_index):

        x = []
        y = []
        for year, param in sorted(self.parameters.items()):
            x.append(year)
            y.append(param.get_value(scenario_index))

        interp = interp1d(x, y, **self.interp_kwargs)
        return interp(ts.year)
    @classmethod
    def load(cls, model, data):

        parameters_data = data.pop("parameters")
        parameters = {}
        for year, pdata in parameters_data.items():
            parameter = load_parameter(model, pdata)
            parameters[int(year)] = (wrap_const(model, parameter))

        return cls(model, parameters=parameters, **data)
AnnualInterpolationParameter.register()


class MonthlyDataFrameParameter(Parameter):
    def __init__(self, model, dataframe, scenarios, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

        self.dataframe = dataframe
        self.scenarios = scenarios
        self._df_index = 0
        self._scenario_indices = []

    def setup(self):
        super().setup()
        self._scenario_indices = []
        for i, scenario in enumerate(self.scenarios):
            self._scenario_indices.append(self.model.scenarios.get_scenario_index(scenario))

            level_length = len(self.dataframe.columns.levels[i])
            if level_length != scenario.size:
                raise ValueError('The size of multiindex level {} is not equal to the size of scenario "{}" ({}).'.format(
                    level_length, scenario.name, scenario.size
                ))

    def reset(self):
        super().reset()
        self._df_index = 0

    def value(self, ts, scenario_index):

        for i in range(self._df_index, len(self.dataframe)):
            df_ts = self.dataframe.index[i]
            if df_ts.year != ts.year and df_ts.month != ts.month:
                self._df_index = i
                break

        scenario_slice = []
        scenario_indices = scenario_index.indices
        for scenario, index in zip(self.scenarios, self._scenario_indices):
            j = scenario_indices[index]
            scenario_slice.append(j)

        return self.dataframe.loc[df_ts, tuple(scenario_slice)]

    @classmethod
    def load(cls, model, data):
        scenarios = data.pop('scenarios', [])
        scenario_instances = []
        for scenario in scenarios:
            scenario_instances.append(model.scenarios[scenario])
        df = load_dataframe(model, data)
        return cls(model, df, scenarios=scenario_instances, **data)
MonthlyDataFrameParameter.register()


class RollingCountOfIndexParameter(Parameter):
    """Compute the rolling count of an index parameter.
    """
    def __init__(self, model, index_parameter, window, **kwargs):
        super().__init__(model, **kwargs)
        self.index_parameter = index_parameter
        self.children.add(index_parameter)
        self.window = window

    def reset(self):
        super().reset()
        self._memory = [[] for si in self.model.scenarios.combinations]

    def value(self, ts, scenario_index):

        memory = self._memory[scenario_index.global_id]

        index = self.index_parameter.get_index(scenario_index)
        memory.append(index*ts.days)
        if len(memory) > self.window:
            memory.pop(0)

        return np.sum(memory)

    @classmethod
    def load(cls, model, data):
        index_parameter = load_parameter(model, data.pop('index_parameter'))
        return cls(model, index_parameter, **data)
RollingCountOfIndexParameter.register()
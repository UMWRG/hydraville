from pywr.recorders import Recorder, NumpyArrayStorageRecorder, NumpyArrayParameterRecorder
from scipy import stats
from scipy.interpolate import interp1d
import pandas


class VolumeRiskStorageRecorder(NumpyArrayStorageRecorder):
    """ Recorder for computing risk of storage volume across multiple scenarios.

    This recorder computes a custom set of values for each resampled time-slice of
    the underlying data. Note that this is in contrast to most recorder classes where
    the `.value()` method returns a value per scenario.

    """
    def __init__(self, *args, **kwargs):
        self.risk_value = kwargs.pop('risk_value')
        self.resample_freq = kwargs.pop('resample_freq', 'A')
        self.resample_agg_func = kwargs.pop('resample_agg_func', 'mean')
        self.rolling_window = kwargs.pop('rolling_window', None)
        self.rolling_func = kwargs.pop('rolling_func', 'mean')
        self.rolling_kwargs = kwargs.pop('rolling_kwargs', {})
        super().__init__(*args, **kwargs)

    def to_dataframe(self):
        # Resample the recorded data
        df = super().to_dataframe()
        return df.resample(self.resample_freq).agg(self.resample_agg_func)

    def values(self):
        df = self.to_dataframe()
        # Compute across all scenarios for each resampled timestamp
        risk = df.apply(stats.percentileofscore, axis=1, args=(self.risk_value, )) / 100

        if self.rolling_window is not None:
            risk = risk.rolling(self.rolling_window, **self.rolling_kwargs).agg(self.rolling_func)

        return risk.values
VolumeRiskStorageRecorder.register()


class FittedDistributionStorageRecorder(VolumeRiskStorageRecorder):
    """ Recorder for fitting a probability distribution to storage volume across multiple scenarios. """
    def __init__(self, *args, **kwargs):
        self.distribution = kwargs.pop('distribution')
        self.fit_kwargs = kwargs.pop('fit_kwargs', {})
        self.fit_rolling_window = kwargs.pop('fit_rolling_window', None)
        self.fit_rolling_func = kwargs.pop('fit_rolling_func', 'mean')
        self.fit_rolling_kwargs = kwargs.pop('fit_rolling_kwargs', {})
        super().__init__(*args, **kwargs)

    def to_dataframe(self):
        # Get the distribution from the stats module.
        dist = getattr(stats, self.distribution)
        columns = [v.strip() for v in stats.beta.shapes.split(',')] + ['loc', 'scale']
        df = super().to_dataframe()

        def fit(values):
            p = dist.fit(values, **self.fit_kwargs)
            return pandas.Series(p, index=columns)

        # Fit a distribution to each row (i.e. across each scenario)
        params = df.apply(fit, axis=1)
        if self.fit_rolling_window is not None:
            params = params.rolling(self.fit_rolling_window, **self.fit_rolling_kwargs)\
                .agg(self.fit_rolling_func)

        return params

    def values(self):
        df = self.to_dataframe()
        # Get the distribution from the stats module.
        dist = getattr(stats, self.distribution)

        def _calc_risk(params):
            return dist.cdf(self.risk_value, *params)

        risk = df.apply(_calc_risk, axis=1)
        if self.rolling_window is not None:
            risk = risk.rolling(self.rolling_window, **self.rolling_kwargs).agg(self.rolling_func)
        return risk.values

FittedDistributionStorageRecorder.register()


class InterpolatedParameterRecorder(NumpyArrayParameterRecorder):
    """ Recorder for computing the annual value of a parameter from an interpolation function. """
    def __init__(self, *args, **kwargs):
        self.resample_freq = kwargs.pop('resample_freq', 'A')
        self.resample_agg_func = kwargs.pop('resample_agg_func', 'mean')
        self.discount_rates = kwargs.pop('discount_rates', {})
        self.x = kwargs.pop('x')
        self.y = kwargs.pop('y')
        interp_kwargs = kwargs.pop('interp_kwargs', None)
        self.interp = None
        default_interp_kwargs = dict(kind='linear', bounds_error=True)

        if interp_kwargs is not None:
            # Overwrite or add to defaults with given values
            default_interp_kwargs.update(interp_kwargs)
        self.interp_kwargs = default_interp_kwargs
        super().__init__(*args, **kwargs)

    def setup(self):
        super().setup()
        self.interp = interp1d(self.x, self.y, **self.interp_kwargs)

    def to_dataframe(self):
        # Get the raw data
        df = super().to_dataframe()
        # Resample the recorded data
        df = df.resample(self.resample_freq).agg(self.resample_agg_func)

        # Apply interpolation
        df = df.apply(self.interp)
        return df

    def values(self):
        """Compute a value for each scenario using `temporal_agg_func`.
        """
        df = self.to_dataframe()
        return self._temporal_aggregator.aggregate_2d(df.values, axis=0, ignore_nan=self.ignore_nan)
InterpolatedParameterRecorder.register()

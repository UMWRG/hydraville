from pywr.parameters import Parameter, load_parameter
from pywr.parameter_property import parameter_property
from pywr.recorders import Recorder, NumpyArrayNodeRecorder, NumpyArrayParameterRecorder, Aggregator
import numpy as np


class IrrigationWaterRequirementParameter(Parameter):
    """Simple irrigation water requirement model. """
    def __init__(self, model, rainfall_parameter, et_parameter, crop_water_factor_parameter, area,
                 reference_et, yield_per_area, revenue_per_yield,
                 application_efficiency=1.0, conveyance_efficiency=1.0, **kwargs):
        super().__init__(model, **kwargs)

        self._rainfall_parameter = None
        self.rainfall_parameter = rainfall_parameter
        self._et_parameter = None
        self.et_parameter = et_parameter
        self._crop_water_factor_parameter = None
        self.crop_water_factor_parameter = crop_water_factor_parameter
        self.area = area
        self.yield_per_area = yield_per_area
        self.revenue_per_yield = revenue_per_yield
        self.reference_et = reference_et
        self.application_efficiency = application_efficiency
        self.conveyance_efficiency = conveyance_efficiency

    rainfall_parameter = parameter_property("_rainfall_parameter")
    et_parameter = parameter_property("_et_parameter")
    crop_water_factor_parameter = parameter_property("_crop_water_factor_parameter")

    def value(self, timestep, scenario_index):

        rainfall = self.rainfall_parameter.get_value(scenario_index)
        et = self.et_parameter.get_value(scenario_index)
        effective_rainfall = rainfall - et

        # Calculate crop water requirement
        if effective_rainfall > self.reference_et:
            # No crop water requirement if there is net rainfall
            crop_water_requirement = 0.0
        else:
            # Irrigation required to meet shortfall in rainfall
            crop_water_factor = self.crop_water_factor_parameter.get_value(scenario_index)
            crop_water_requirement = crop_water_factor * (self.reference_et - effective_rainfall) * self.area

        # Calculate overall efficiency
        efficiency = self.application_efficiency * self.conveyance_efficiency

        # TODO error checking on division by zero
        irrigation_water_requirement = crop_water_requirement / efficiency
        return irrigation_water_requirement

    def crop_yield(self, curtailment_ratio):
        return self.area * self.yield_per_area * curtailment_ratio

    def crop_revenue(self, curtailment_ratio):
        return self.revenue_per_yield * self.crop_yield(curtailment_ratio)

    @classmethod
    def load(cls, model, data):
        rainfall_parameter = load_parameter(model, data.pop('rainfall_parameter'))
        et_parameter = load_parameter(model, data.pop('et_parameter'))
        cwf_parameter = load_parameter(model, data.pop('crop_water_factor_parameter'))
        return cls(model, rainfall_parameter, et_parameter, cwf_parameter, **data)
IrrigationWaterRequirementParameter.register()


class RelativeCropYieldRecorder(Recorder):
    """Relative crop yield recorder.

    This recorder computes the relative crop yield based on a curtailment ratio between a node's
    actual flow and it's `max_flow` expected flow. It is assumed the `max_flow` parameter is an
    `AggregatedParameter` containing only `IrrigationWaterRequirementParameter` parameters.

    """
    def __init__(self, model, node, **kwargs):
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')
        super().__init__(model, **kwargs)

        self._node_recorder = NumpyArrayNodeRecorder(model, node)
        self.children.add(self._node_recorder)
        self._parameter_recorder = NumpyArrayParameterRecorder(model, node.max_flow)
        self.children.add(self._parameter_recorder)

        self._temporal_aggregator = Aggregator(temporal_agg_func)
        self.effective_yield = None

    def finish(self):

        actual = self._node_recorder.to_dataframe().resample('M').mean()
        requirement = self._parameter_recorder.to_dataframe().resample('M').mean()
        # TODO error checking on division by zero
        curtailment_ratio = actual / requirement
        # Annual minimum ratio
        curtailment_ratio = curtailment_ratio.resample('A').min()

        # Here we assume this max_flow is some sort of aggregated parameter
        crop_aggregated_parameter = self._node_recorder.node.max_flow
        # Get first crop_revenue for normalisation
        norm_crop_revenue = crop_aggregated_parameter.parameters[0].crop_revenue(curtailment_ratio)
        # Create an empty array contain the accumulate yields
        effective_yield = np.zeros_like(curtailment_ratio)

        # Loop through all the crops
        for parameter in crop_aggregated_parameter.parameters:
            crop_revenue = parameter.crop_revenue(curtailment_ratio)
            crop_yield = parameter.crop_yield(curtailment_ratio)
            # Increment effective yield, scaled by the first crop's revenue
            # TODO error checking on division by zero
            effective_yield += crop_yield * crop_revenue / norm_crop_revenue

        self.effective_yield = effective_yield

    def values(self):
        return self._temporal_aggregator.aggregate_2d(self.effective_yield.values, axis=0, ignore_nan=self.ignore_nan)
RelativeCropYieldRecorder.register()

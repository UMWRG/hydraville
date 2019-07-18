import logging
logger = logging.getLogger(__name__)

from .parameters import AnnualInterpolationParameter, DistributionParameter, MonthlyDataFrameParameter
from .crop_water import IrrigationWaterRequirementParameter
from .recorders import FittedDistributionStorageRecorder, InterpolatedParameterRecorder, VolumeRiskStorageRecorder

import os
DATA_DIR = os.path.join(os.path.dirname(__file__), 'json')
TEMPLATE_DIR = os.path.join(DATA_DIR, 'templates')
CATCHMOD_DIR = os.path.join(DATA_DIR, 'catchmod')

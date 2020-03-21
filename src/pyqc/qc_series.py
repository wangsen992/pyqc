import math
import copy
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import scipy.fftpack as fft
import ipdb as debugger

from .base import QualityControlBaseAccessor
from .utils import *

#=============================General Accessors==============================#
@pd.api.extensions.register_series_accessor("qc")
class QualityControlSeriesAccessor(QualityControlBaseAccessor):
    pass

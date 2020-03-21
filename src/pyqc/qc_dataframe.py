import math
import copy
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import scipy.fftpack as fft
import ipdb as debugger

from base import QualityControlBaseAccessor
from .utils import *

@pd.api.extensions.register_dataframe_accessor("qc")
class QualityControlDataFrameAccessor(QualityControlBaseAccessor):
    pass

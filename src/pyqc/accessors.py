import math
import copy
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import scipy.fftpack as fft

from .base import QualityControlBaseAccessor
from .utils import *

#=============================General Accessors==============================#
@pd.api.extensions.register_series_accessor("qc")
class QualityControlSeriesAccessor(QualityControlBaseAccessor):
    def describe(self):
        out_dict =  dict(mean = self._obj.mean(),
                        std = self._obj.std(),
                        skew= self._obj.skew(),
                        kurt= self._obj.kurtosis(),
                        pct_null = self._obj.isna().sum()/self._obj.size,
                        stationarity_measure = self.stationarity_measure,
                        pct_spike_flag = self.spike_mask.sum()/self._obj.size,
                        pct_hist_flag = self.hist_mask.sum()/self._obj.size)

        return pd.Series(out_dict, name=self._obj.name)

@pd.api.extensions.register_dataframe_accessor("qc")
class QualityControlDataFrameAccessor(QualityControlBaseAccessor):
    def describe(self):
        return pd.concat([self._obj[varname].qc.describe() \
                          for varname in self._obj.columns],
                         axis=1)

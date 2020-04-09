'''Base class for quality control accessor on dataframes and series'''

import copy
import numpy as np
import pandas as pd

from .utils import *

#=============================General Accessors==============================#
class QualityControlBaseAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self._init_qc_options()

    @staticmethod
    def _validate(obj):
        # Validate and enforce input index is equally spaced.
        if len(set(obj.index[1:] - obj.index[:-1])) != 1:
            raise AttributeError("Ensure index is equally spaced.")

    # Quality Control options
    # Internal states for qc operations. Provide update and reset options.
    def _init_qc_options(self):
        self._options = dict()
        # compute original data info
        duration = self._obj.index[-1] - self._obj.index[0]

        # assign init options
        self._options['spike_window'] = duration / 20
        self._options['spike_stride'] = self._options['spike_window'] / 20
        self._options['spike_factor'] = 3.5
        self._options['hist_window'] = self._options['spike_window']
        self._options['hist_bins'] = 200
        self._options['hist_pct_thres'] = 0.8
        self._options['stationarity_window'] = self._options['spike_window']

        self._default_options = copy.copy(self._options)
        self._old_options = copy.copy(self._options)

    def set_options(self, **kwargs):
        self._old_options.update(self._options)
        self._options.update(kwargs)

    def reset_default_options(self):
        self._old_options.update(self._options)
        self._options.update(self._default_options)

    @property
    def options(self):
        return self._options

    @property
    def option_is_updated(self):
        return self._options != self._old_options

    # Spike detection
    @property
    def spike_mask(self):
        if (not hasattr(self, '_spike_mask')) or self.option_is_updated:
            self._compute_spike_mask(inplace=True)
        return self._spike_mask

    def despike(self, inplace=False):
        if inplace == True:
            self._obj[self.spike_mask] = np.nan
        else:
            new_obj = self._obj.copy()
            new_obj[self.spike_mask] = np.nan
            return new_obj

    def _compute_spike_mask(self,
                            inplace=False):

        _spike_mask = compute_spike_mask(self._obj,
                                         window=self._options['spike_window'], 
                                         stride=self._options['spike_stride'], 
                                         factor=self._options['spike_factor'])

        if inplace == True:
            self._spike_mask = _spike_mask
        else:
            return _spike_mask

    # Amplitude resolution and dropouts detection
    @property
    def hist_mask(self):
        if not hasattr(self, '_hist_mask') or self.option_is_updated:
            self._compute_hist_mask(inplace=True)
        return self._hist_mask

    def _compute_hist_mask(self,
                          inplace=False):

        hist_mask = hist_based_mask_func[type(self._obj)]\
                                       (self._obj, 
                                       window=self._options['hist_window'], 
                                       bins=self._options['hist_bins'], 
                                       pct_thres=self._options['hist_pct_thres'])

        if inplace == True:
            self._hist_mask = hist_mask
        else:
            return hist_mask


    # Measure single variable nonstationarity by comparing normalized rolling
    # mean range over record mean
    @property
    def stationarity_measure(self):
        if not hasattr(self, '_stationarity_measure'):
            self._compute_stationarity_measure(inplace=True)
        return self._stationarity_measure

    def _compute_stationarity_measure(self, inplace=False):

        stationarity_measure = mean_ptp_ratio(self._obj,
                                              window=self._options['stationarity_window'])

        if inplace == True:
            self._stationarity_measure = stationarity_measure
        else:
            return stationarity_measure

    def describe(self):
        raise NotImplementedError("Should not call directly from base class")

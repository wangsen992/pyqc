import math
import numpy as np
import pandas as pd

def series_rolling(x, window, stride):

    if not isinstance(window, (int, float)) or not isinstance(stride, (int, float)):
        window_size = math.floor(window / x.index.freq)
        stride_size = math.floor(stride / x.index.freq) 
    end_index = x.shape[0] - window_size

    if stride_size == 0: 
        stride_size =1

    for i in np.arange(0, end_index, stride_size):
        yield x.iloc[i: i + window_size]

def pd_rolling(x, window, stride, resolution=None):

    # For DatetimeIndex
    if not isinstance(window, (int, float)) or not isinstance(stride, (int, float)):
        try:
            window = pd.Timedelta(window)
            stride = pd.Timedelta(stride)
        except ValueError:
            raise ValueError("Input window & stride must either be real or"\
                            + "offset strings")
        if not x.index.freq:
            freq = x.index[1] - x.index[0]
        else:
            freq = x.index.freq
        window_size = math.floor(window / freq)
        stride_size = math.floor(stride / freq) 

    # For real-valued index
    elif resolution:
        window_size = window // resolution
        stride_size = stride // resolution
    
    end_index = x.shape[0] - window_size

    if stride_size == 0: 
        stride_size =1

    for i in np.arange(0, end_index, stride_size):
        yield x.iloc[i: i + window_size]

def compute_spike_mask(x, window, stride, factor=3.5):
    '''Create a mask on x where spikes are True based on running mean and std. 

    This method should work datatype-agnostically on pd.DataFrame and
    pd.Series.'''

    # create a boolean mask for the entire dataset
    x_mask = x.astype(bool)
    x_mask.values[:] = False

    # processing rolling operations
    for x_sub in pd_rolling(x, window, stride):
        x_mean = x_sub.mean()
        x_std = x_sub.std()
        x_sub_mask = np.abs(x_sub-x_mean) > (factor *x_std)
        if x.ndim == 2:
            x_mask[x_sub_mask] = True
        elif x.ndim == 1:
            x_mask[x_sub_mask.index[x_sub_mask.values]] = True

    return x_mask

def hist_based_mask_series(x, window, bins, pct_thres=0.5):

    # create a boolean mask for the entire dataset
    x_mask = x.astype(bool) 
    x_mask.values[:] = False

    # compute stride from given window size
    window = pd.Timedelta(window)
    stride = window / 2

    # Processing rolling operations (This needs to be done seperately for
    # series and dataframe)
    for x_sub in pd_rolling(x, window, stride):
        x_sub_no_nan = x_sub.dropna()

        if x_sub_no_nan.size/x_sub.size > 0.2:
            hist, bins = np.histogram(x_sub.dropna().values,
                                     bins=bins,
                                     range=[x_sub.min(), x_sub.max()])
            if ((hist == 0).sum() / hist.size) >= pct_thres:
                x_sub_mask = x_sub_no_nan.astype(bool)
                x_sub_mask.values[:] = True
                x_mask[x_sub_mask.index[x_sub_mask.values]] = True

    return x_mask

def hist_based_mask_dataframe(x, window, bins, pct_thres=0.5):
     return pd.concat([hist_based_mask_series(x[varname],window,bins,pct_thres) \
                       for varname in x.columns], axis=1)

# Generalise the function usage
hist_based_mask_func = {pd.core.series.Series : hist_based_mask_series,
                         pd.core.frame.DataFrame : hist_based_mask_dataframe}

def mean_ptp_ratio(x, window='2T'):
    '''Compute the ratio of running mean range to record mean

    Parameters:
        x:      pd.Series with DatetimeIndex. Record to measure
        window: pd.DateOffset string. Running window size
    '''

    x_normed = (x - x.min()) / (x.max() - x.min())
    x_mean_rolling = x_normed.rolling(window).mean()
    return (x_mean_rolling.max() - x_mean_rolling.min())/x_normed.mean()

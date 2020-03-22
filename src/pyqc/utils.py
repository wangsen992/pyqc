import math
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import scipy.fftpack as fft


def spike_mask(x, window, stride, factor=3.5):
    '''Create a mask on x where spikes are True based on running mean and std. 

    This method should work datatype-agnostically on pd.DataFrame and
    pd.Series.'''

    pass

def series_rolling(x, window, stride):
    window_size = math.floor(window / x.index.freq)
    stride_size = math.floor(stride / x.index.freq) 
    end_index = x.shape[0] - window_size

    if stride_size == 0: 
        stride_size =1

    for i in np.arange(0, end_index, stride_size):
        yield x.iloc[i: i + window_size]

def spike_flags(x,
                window='2T',
                stride='10s',
                factor=3.5):
    '''Flag data entry indice identified with spikes using running mean and std
    method.'''

    flag_indice = set()
    for x_sub in series_rolling(x, window, stride):
        x_mean = x_sub.mean()
        x_std = x_sub.std()
        sub_flagged_indice =\
                x_sub[np.abs(x_sub-x_mean) > (factor *x_std)].index.to_list()
        flag_indice = flag_indice.union(sub_flagged_indice)

    return flag_indice

def hist_based_flags(x,
                     window='2T',
                     bins=100,
                     pct_thres=0.5):

    window = pd.Timedelta(window)
    stride = window / 2

    flag_indice = set()
    for x_sub in series_rolling(x, window, stride):
        x_sub_no_nan = x_sub.dropna() 
        
        if x_sub_no_nan.size/x_sub.size < 0.2:
            hist, bins = np.histogram(x_sub.dropna().values,
                                     bins=bins,
                                     range=[x_sub.min(), x_sub.max()])
            if ((hist == 0).sum() / hist.size) >= pct_thres:
               flag_indice = flag_indice.union(x_sub.index.to_list())

    return flag_indice

def mean_ptp_ratio(x, window='2T'):
    '''Compute the ratio of running mean range to record mean

    Parameters:
        x:      pd.Series with DatetimeIndex. Record to measure
        window: pd.DateOffset string. Running window size
    '''

    x_normed = (x - x.min()) / (x.max() - x.min())
    x_mean_rolling = x_normed.rolling(window).mean()
    return (x_mean_rolling.max() - x_mean_rolling.min())/x_normed.mean()

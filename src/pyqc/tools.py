import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

def summarize_qc_resamples(input_df, verbose=False, **resample_kwargs):

    time_list = list()
    data_list = list()

    for time, df in input_df.resample(**resample_kwargs):
        if verbose == True:
            print("Currently working on: {}".format(time))
        time_list.append(time)
        df_stats = df.qc.describe()
        data_list.append(df_stats.values)
    else:
        measures = df_stats.index.to_list()
        variables = df.columns.to_list()

    attrs = resample_kwargs

    return xr.DataArray(np.dstack(data_list),
                        coords = [measures, variables, time_list],
                        dims   = ['measure','variable','time'],
                        name   = "qc_summary",
                        attrs  = attrs)

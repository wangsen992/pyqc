# pyqc

A general quality control extension to pandas dataframe and series. Quality
control methods are designed to apply only to 1D array, likely to be timeseries
data. So, Pandas DatetimeIndex is also supported. 

## How to use
Load the data as you normally would with Pandas. After which use accessor
method with name `qc` to apply methods. 
```python3
import pyqc
import pandas as pd

df = pd.DataFrame(np.random.randn(100,3), names=list('abc'))
# apply method as such
spike_mask = df.qc.spike_indice
```
Right now there are detections of spikes and histogram-based measures such as
data resolution issues, or dropouts. 

# Common script to call when Python module is loaded
# For example, import packages that is needed to run the module.
# 
#
# import numpy

from .. import shared
from . import RAVERuntimeException

def pipeline_target_power_outputs(subset_analyzed, sample_rate, window_length, freq_range):
  try:
    # Parameters for preprocessing (you should adjust these based on your actual requirements)
    fs = sample_rate
    
    # Number of timepoints = sample rate * window size in sec
    nperseg = sample_rate * window_length
    
    freq_range = freq_range
    
    power_outputs = shared.preprocess_powers(subset_analyzed, fs=fs, nperseg=nperseg, freq_range=freq_range)
    power_outputs = shared.add_mean_and_std(power_outputs)
    return power_outputs
  except Exception as e:
    return RAVERuntimeException(e)



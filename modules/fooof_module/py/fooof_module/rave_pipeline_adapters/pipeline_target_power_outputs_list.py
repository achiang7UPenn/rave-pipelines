# Common script to call when Python module is loaded
# For example, import packages that is needed to run the module.
# 
#
# import numpy

from .. import shared
from . import RAVERuntimeException

def pipeline_target_power_outputs_list(subset_analyzed, sample_rate, window_length, freq_range):
  try:
    # Parameters for preprocessing (you should adjust these based on your actual requirements)
    fs = sample_rate
    
    # Number of timepoints = sample rate * window size in sec
    nperseg = sample_rate * window_length
    
    freq_range = freq_range
    power_outputs_list = []
    for i in range(0, len(subset_analyzed)):
      power_outputs = shared.preprocess_powers(subset_analyzed[i], fs=fs, nperseg=nperseg, freq_range=freq_range)
      power_outputs = shared.add_mean_and_std(power_outputs)
      power_outputs_list.append(power_outputs)
    return power_outputs_list
  except Exception as e:
    return RAVERuntimeException(e)



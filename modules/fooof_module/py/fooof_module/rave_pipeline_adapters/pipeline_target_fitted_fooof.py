# Common script to call when Python module is loaded
# For example, import packages that is needed to run the module.
# 
#
# import numpy

from .. import shared
from . import RAVERuntimeException

def pipeline_target_fitted_fooof(power_outputs, max_n_peaks, aperiodic_mode, freq_range):
  try:
    filtered_frequency = power_outputs['filtered_frequency']
    average_power = power_outputs['Average Power']
    # title = None
    # freq_range = freq_range
    # plt_log = False
    # aperiodic_mode = 'fixed'
    fitted_fooof = shared.fit_fooof(
      filtered_freqs = filtered_frequency, 
      filtered_powers = average_power, 
      freq_range = freq_range,
      max_n_peaks=max_n_peaks, 
      aperiodic_mode=aperiodic_mode
    )
    return fitted_fooof
  except Exception as e:
    return RAVERuntimeException(e)



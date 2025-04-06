# Common script to call when Python module is loaded
# For example, import packages that is needed to run the module.
# 
#
# import numpy

from .. import shared
from . import RAVERuntimeException

def pipeline_target_fitted_fooof(power_outputs_list, max_n_peaks, aperiodic_mode, freq_range):
  try:
    
    
    # title = None
    # freq_range = freq_range
    # plt_log = False
    # aperiodic_mode = 'fixed'
    fitted_fooof = shared.fit_fooof(
      power_outputs_list = power_outputs_list,
      freq_range = freq_range,
      max_n_peaks=max_n_peaks, 
      aperiodic_mode=aperiodic_mode
    )
    return fitted_fooof
  except Exception as e:
    return RAVERuntimeException(e)



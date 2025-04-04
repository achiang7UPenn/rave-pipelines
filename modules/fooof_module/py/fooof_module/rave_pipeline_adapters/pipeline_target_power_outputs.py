# Common script to call when Python module is loaded
# For example, import packages that is needed to run the module.
# 
#
# import numpy

from .. import shared
from . import RAVERuntimeException

def pipeline_target_power_outputs(power_outputs_list):
  try:
    power_outputs = power_outputs_list[0]
    return power_outputs
  except Exception as e:
    return RAVERuntimeException(e)



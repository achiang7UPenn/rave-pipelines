
class RAVERuntimeException(object):
  original_exception = None
  def __init__(self, e):
    if isinstance(e, Exception):
      self.original_exception = e
    elif isinstance(e, str):
      self.original_exception = Exception(e)
    else:
      self.original_exception = Exception('Unknown error')
  def __str__(self):
    return '{}: {}'.format(type(self.original_exception).__name__, self.original_exception)

from .serializers import rave_serialize
from .serializers import rave_unserialize

from .pipeline_target_power_outputs import pipeline_target_power_outputs as power_outputs
from .pipeline_target_fitted_fooof import pipeline_target_fitted_fooof as fitted_fooof

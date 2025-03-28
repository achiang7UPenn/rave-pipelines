# Common script to call when Python module is loaded
# For example, import packages that is needed to run the module.
# 
#
# import numpy

from .. import shared
from . import RAVERuntimeException

def pipeline_target_power_outputs(subset_analyzed):
  try:
    power_outputs = shared.pd.DataFrame()
    freqs = shared.pd.DataFrame()
    
    # Parameters for preprocessing (you should adjust these based on your actual requirements)
    fs = 2000
    nperseg = 4000
    freq_range = [1, 175]
    
    
    # Loop through each column in the original dataframe
    freq_in = False
    for column in subset_analyzed.columns[1:]:
        # Apply the preprocess function to each column
        # subset_analyzed[column].values to pass the column data as an array
        filtered_freqs, filtered_powers = shared.preprocess_data(subset_analyzed[column].values, frequency=fs, freq_range=freq_range, nperseg=nperseg)
        
        # Create a new column name based on the original column name
        new_column_name = f"Trial_{column}_power"
        
        # Add the processed power data to the new dataframe
        if not freq_in:
            power_outputs["filtered_frequency"] = filtered_freqs
            freq_in = True
        power_outputs[new_column_name] = filtered_powers
        freqs[new_column_name] = filtered_freqs
      
    # power_outputs.columns
    power_outputs = shared.add_mean_and_std(power_outputs)
    # power_outputs
    return power_outputs
  except Exception as e:
    return RAVERuntimeException(e)



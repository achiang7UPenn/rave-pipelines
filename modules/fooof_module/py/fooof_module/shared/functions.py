import h5py
from specparam import SpectralModel, SpectralGroupModel
from scipy.signal import welch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import base64
import os
from io import BytesIO


## DEFINE FUNCTIONS FOR DATA PROCESSING PIPELINE ##
def extract_data(electrode, block, frequency, timestamp, length, plot=False):
  """
    Extract raw data for the selected electrodes.

    Args:
        electrode: the number of the electrode to be extracted.
        block: the number of the experiment block to be extracted.
        frequency: the sampling frequency of the data.
        timestamp: the start time of the data to be extracted in seconds.
        length: the length of the data to be extracted in seconds.
    Returns:
        voltage_data: the voltage data from the selected electrode and block.
  """
  electrode = str(electrode)
  block = str(block)

  # Ensure block is in righ form
  if len(block) == 1:
    block = '00' + block
  elif len(block) == 2:
    block = '0' + block

  # Read data from electrode
  electrode_data = h5py.File(FILEPATH + electrode + '.h5', 'r')

  # Extract filtered data
  voltage_data = electrode_data['ref']['voltage'][block]

  # Extract the desired time split
  if length is not None:
    voltage_data = voltage_data[int(timestamp * frequency) : int((timestamp + length) * frequency)]

  if plot:
    plt.title(f"Electrode {electrode} block {block} voltage, time window: {timestamp} : {timestamp + length}s")
    plt.plot(np.arange(0, len(voltage_data[int(timestamp * frequency):int((timestamp + length) * frequency)])/frequency, 1/frequency), voltage_data[int(timestamp * frequency) : int((timestamp + length) * frequency)])
    plt.show()

  return voltage_data


def preprocess_data(data, frequency, freq_range, nperseg):
  """
    Transform data to frequency domain and filter out high frequencies.

    Args:
        data: neural voltage recording.
        frequency: the sampling frequency of the data.
        freq_range: the frequency range to be analyzed.
        nperseg: the number of samples per segment in Welch filter.
    Returns:
        filtered_freqs: the filtered frequencies.
        filtered_powers: the filtered powers.
  """

  # Apply the Welch filter
  freqs, powers = welch(data, fs=frequency, nperseg=nperseg)

  # Next, we'll filter out high frequencies
  freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])

  # Apply the mask to filter freqs and powers
  filtered_freqs = freqs[freq_mask]
  filtered_powers = powers[freq_mask]

  return filtered_freqs, filtered_powers


def generate_model_fit(filtered_freqs, filtered_powers, freq_range = [1,300], max_n_peaks = 10000,
                  aperiodic_mode = "fixed", plt_log = False, report = False):
  """
    Generate a plot of the spectral power for the selected electrodes.

    Args:
        electrode: the number of the electrode to be analyzed.
        block: the number of the ex[eriment block to be analyzed.
        frequency: the sampling frequency of the data.
        freq_range: the frequency range to be analyzed.
        nperseg: the number of samples per segment in Welch filter.
    Returns:
        None
  """

  # Initialize model object
  fm = SpectralModel(max_n_peaks = max_n_peaks, aperiodic_mode = aperiodic_mode)
  fm.fit(filtered_freqs, filtered_powers, freq_range)

  # Parameterize the power spectrum, and print out a plot
  if report:
    fm.report(filtered_freqs, filtered_powers, freq_range = freq_range)
    # plt.title(f"Electrode {electrode}, Block {block} at {timestamp}s Spectral Plot")
  else:
    plot = fm.plot(plt_log = plt_log)
    # plt.title(f"Electrode {electrode}, Block {block} at {timestamp}s Spectral Plot")
  plt.show()


def fit_fooof(filtered_freqs, filtered_powers, freq_range = [1, 175],
              max_n_peaks = 10000, aperiodic_mode = "fixed"):
  """
    Fits powers and frequencies using FOOOF and plots it.

    Args:
        electrode: the frequencies of the signal.
        block: the powers to fit.
        freq_range: the frequency range to be analyzed.
    Returns:
        None
  """

  
  if aperiodic_mode.startswith("f"):
    aperiodic_mode = 'fixed'
  else:
    aperiodic_mode = 'knee'
  
  filtered_freqs_np = np.array(filtered_freqs)
  filtered_powers_np = np.array(filtered_powers)
  
  fm = SpectralModel(peak_width_limits = freq_range, max_n_peaks = max_n_peaks, aperiodic_mode = aperiodic_mode)
  fm.fit(filtered_freqs_np, filtered_powers_np, freq_range)
  # fm.report(filtered_freqs_np, filtered_powers_np, freq_range)
  fooof_fit = {
    "model"       : fm,
    "frequencies" : filtered_freqs_np,
    "power"       : filtered_powers_np,
    "freq_range"  : freq_range,
    "max_n_peaks" : max_n_peaks,
    "aperiodic_mode": aperiodic_mode
  }
  
  return fooof_fit
  # fm.report(filtered_freqs_np, filtered_powers_np, freq_range)
  # return fm
  # 
  # # Parameterize the power spectrum, and print out a plot
  # # plot = fm.plot(plt_log = plt_log)
  # plt.title(f"Condition Analyzed: {title}")
  # plt.show()
  
  
## Defining

def get_electrode_data_during_experiment(experiment_info_df, electrodes, frequency, nperseg, freq_range, length):

  for electrode in electrodes:
    filt_freqs_list = []
    filt_powers_list = []

    for index, row in experiment_info_df.iterrows():
      voltage_data = extract_data(electrode, str(row['Block']), frequency, row['Time'], length)
      filtered_freqs, filtered_powers = preprocess_data(voltage_data, frequency, freq_range, nperseg)
      filt_freqs_list.append(filtered_freqs)
      filt_powers_list.append(filtered_powers)

    experiment_info_df[f'filt_freqs electrode_{electrode}'] = filt_freqs_list
    experiment_info_df[f'filt_powers electrode_{electrode}'] = filt_powers_list

  return experiment_info_df


# Dictionary: {"audio only": 1, "visual only": 2, "congruent": 3, "incongruent": 4}
# Define function to assign condition types using regex
def assign_condition_type(condition):
    if re.search(r'_a$', condition):   # Ends with "_a"
        return 1
    elif re.search(r'_v$', condition):  # Ends with "_v"
        return 2
    elif re.search(r'_AV$', condition): # Ends with "_AV"
        return 3
    else:
        return 4  # Everything else


def add_mean_and_std(df, mean_col_name="Average Power", std_col_name="StdDev"):
    """
    Adds two new columns to the DataFrame: one for the mean and one for the standard deviation
    of all columns except the first one.

    This function computes the mean and standard deviation across all columns except the first
    for each row, adding these as new columns to the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        mean_col_name (str): Name of the new column that will store the average.
        std_col_name (str): Name of the new column that will store the standard deviation.

    Returns:
        pd.DataFrame: The modified DataFrame with the new average and standard deviation columns.
    """
    # Check if DataFrame has more than one column
    if df.shape[1] > 1:
        # Calculate the mean and standard deviation of each row for all columns except the first
        mean_column = df.iloc[:, 1:].mean(axis=1)
        std_column = df.iloc[:, 1:].std(axis=1)

        # Add calculated mean and standard deviation as new columns in the DataFrame
        df[mean_col_name] = mean_column
        df[std_col_name] = std_column
    else:
        # Raise an error if there is not enough data to compute the statistics
        raise ValueError("DataFrame must have more than one column to compute statistics.")

    return df

def plot_trials(df, individual_trials=False):
    fig = go.Figure()
    for i in range(0, len(df)):
      dataframe = df[i]
      
      # Add the first trace (filt_powers electrode_14_avg)
      fig.add_trace(go.Scatter(
          x=dataframe['filtered_frequency'],
          y=np.log10(dataframe['Average Power']),
          mode='lines',
          name=f'Average Power Condition {i+1}'
      ))
  
      # Add the second trace (filt_powers electrode_14_std)
      fig.add_trace(go.Scatter(
          x=dataframe['filtered_frequency'],
          y=np.log10(dataframe['StdDev']),
          mode='lines',
          name=f'Standard Deviation Condition {i+1}'
      ))
  
      if individual_trials:
          # Plot each column except 'filtered_frequency', 'Average Power', and 'StdDev'
          for col in dataframe.columns:
              if col not in ['filtered_frequency', 'Average Power', 'StdDev']:
                  fig.add_trace(go.Scatter(
                      x=dataframe['filtered_frequency'],
                      y=np.log(dataframe[col]),
                      mode='lines',
                      name=col
                  ))
  
  
    # Update layout for grid and other settings
    fig.update_layout(
        title='log(Power) vs Frequency',
        xaxis_title='Frequency',
        yaxis_title='log(Power)',
        showlegend=True,
        template='plotly_white',
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
  
    # Show the figure
    return fig
    
    
def plot_raw_data(data, fs):
  plt.plot(np.arange(0, len(data)/fs, 1/fs), data)
  plt.show()


def tune_max_n_peaks(filtered_freqs, filtered_powers, freq_range, aperiodic_mode, peaks_range, show_errors=True):
  # Initialize storage
  r2s = []
  errors = []

  # Initialize the Plotly figure
  fig = go.Figure()
  
  filtered_freqs_np = np.array(filtered_freqs)
  filtered_powers_np = np.array(filtered_powers)

  # Loop through peak values and fit models
  for max_n_peaks in peaks_range:
      nm = SpectralModel(max_n_peaks=max_n_peaks, aperiodic_mode='fixed')
      nm.fit(filtered_freqs_np, filtered_powers_np, freq_range=freq_range)

      # Append metrics
      r2s.append(nm.r_squared_)
      errors.append(nm.error_)

      # Get the model fit and add it as a trace
      model_freqs, model_fit = nm.freqs, nm.get_model()
      fig.add_trace(go.Scatter(x=model_freqs, y=model_fit,
                              mode='lines',
                              name=f'{max_n_peaks} peaks (R²={nm.r_squared_:.2f})'))
      

  # Add the original data as a baseline (optional)
  fig.add_trace(go.Scatter(x=nm.freqs, y=nm.get_data(),
                          mode='lines', name='Original Spectrum',
                          line=dict(color='black', dash='dash')))

  # Customize the layout
  fig.update_layout(
      title='Spectral Model Fits Across Different Peak Numbers',
      xaxis_title='Frequency (Hz)',
      yaxis_title='Power',
      legend_title='Model',
      template='plotly_white'
  )
  fig.show()


  if show_errors:
    # Graphs of R2 and Errors
    fig, axs = plt.subplots(1, 2, figsize = (16, 5))
    axs[0].plot(peaks_range, r2s, "bo-")
    axs[0].set_title("R2 vs. Number of Peaks")
    axs[0].set_ylabel("R2")
    axs[0].set_xlabel("Number of Peaks")

    axs[1].plot(peaks_range, errors, "ro-")
    axs[1].set_title("MSE vs. Number of Peaks")
    axs[1].set_ylabel("MSE")
    axs[1].set_xlabel("Number of Peaks")
    
    plt.show()

def tune_aperiodic_mode(df, freq_range, max_n_peaks, show_errors=True):
    fig = go.Figure()
    aperiodic_modes = ['fixed', 'knee']
    error_figs_base64 = []  # list of base64-encoded matplotlib figs

    for i, dataframe in enumerate(df):
        filtered_freqs_np = np.array(dataframe['filtered_frequency'])
        filtered_powers_np = np.array(dataframe['Average Power'])

        r2s = []
        errors = []

        for aperiodic_mode in aperiodic_modes:
            nm = SpectralModel(peak_width_limits=freq_range, max_n_peaks=max_n_peaks, aperiodic_mode=aperiodic_mode)
            nm.fit(filtered_freqs_np, filtered_powers_np, freq_range)
            r2s.append(nm.r_squared_)
            errors.append(nm.error_)

            model_freqs, model_fit = nm.freqs, nm.get_model()
            fig.add_trace(go.Scatter(
                x=model_freqs, y=model_fit,
                mode='lines',
                name=f'{aperiodic_mode} (R²={nm.r_squared_:.2f}) - Condition {i+1}'
            ))

        fig.add_trace(go.Scatter(
            x=nm.freqs, y=nm.get_data(),
            mode='lines', name=f'Original Spectrum - Condition {i+1}',
            line=dict(color='black', dash='dash')
        ))

        if show_errors:
            # R² plot
            fig_r2, ax_r2 = plt.subplots()
            ax_r2.plot(aperiodic_modes, r2s, "bo-")
            ax_r2.set_title(f"R² - Condition {i+1}")
            ax_r2.set_ylabel("R²")
            ax_r2.set_xlabel("Aperiodic Mode")
            plt.tight_layout()

            buf_r2 = BytesIO()
            fig_r2.savefig(buf_r2, format='png')
            buf_r2.seek(0)
            error_figs_base64.append(base64.b64encode(buf_r2.read()).decode('utf-8'))
            buf_r2.close()
            plt.close(fig_r2)

            # MSE plot
            fig_err, ax_err = plt.subplots()
            ax_err.plot(aperiodic_modes, errors, "ro-")
            ax_err.set_title(f"MSE - Condition {i+1}")
            ax_err.set_ylabel("MSE")
            ax_err.set_xlabel("Aperiodic Mode")
            plt.tight_layout()

            buf_err = BytesIO()
            fig_err.savefig(buf_err, format='png')
            buf_err.seek(0)
            error_figs_base64.append(base64.b64encode(buf_err.read()).decode('utf-8'))
            buf_err.close()
            plt.close(fig_err)

    fig.update_layout(
        title='Spectral Model Fits for Different Aperiodic Modes',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Power',
        legend_title='Model Configurations',
        template='plotly_white'
    )

    return {
        'plotly': fig,
        'matplotlib': error_figs_base64  # list of base64-encoded images
    }


        
def tune_peak_threshold(filtered_freqs, filtered_powers, freq_range, max_n_peaks, aperiodic_mode, peak_threshold_range, show_errors=True):
    # Initialize storage
    r2s = []
    errors = []

    # Initialize the Plotly figure
    fig = go.Figure()
    
    filtered_freqs_np = np.array(filtered_freqs)
    filtered_powers_np = np.array(filtered_powers)

    # Loop through peak thresholds and fit models
    for peak_threshold in peak_threshold_range:
        nm = SpectralModel(max_n_peaks=max_n_peaks, aperiodic_mode=aperiodic_mode, peak_threshold=peak_threshold)
        nm.fit(filtered_freqs_np, filtered_powers_np, freq_range=freq_range)

        # Append metrics
        r2s.append(nm.r_squared_)
        errors.append(nm.error_)

        # Get the model fit and add it as a trace
        model_freqs, model_fit = nm.freqs, nm.get_model()  # Assuming the attributes are named like this
        fig.add_trace(go.Scatter(x=model_freqs, y=model_fit,
                                 mode='lines',
                                 name=f'Peak Threshold {peak_threshold} (R²={nm.r_squared_:.2f})'))

    # Add the original data as a baseline (optional)
    fig.add_trace(go.Scatter(x=nm.freqs, y=nm.get_data(),
                             mode='lines', name='Original Spectrum',
                             line=dict(color='black', dash='dash')))

    # Customize the layout
    fig.update_layout(
        title='Spectral Model Fits Across Different Peak Thresholds',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Power',
        legend_title='Model Configurations',
        template='plotly_white'
    )
    fig.show()

    if show_errors:
        # Graphs of R2 and Errors using Matplotlib
        fig, axs = plt.subplots(1, 2, figsize=(16, 5))
        
        axs[0].plot(peak_threshold_range, r2s, "b-")  # Using 'bo-' for blue circles connected by lines
        axs[0].set_title("R² vs. Peak Thresholds")
        axs[0].set_ylabel("R²")
        axs[0].set_xlabel("Peak Threshold")
        axs[0].set_xscale("log")  # Assuming peak_threshold_range needs a log scale

        axs[1].plot(peak_threshold_range, errors, "r-")  # Using 'ro-' for red circles connected by lines
        axs[1].set_title("MSE vs. Peak Thresholds")
        axs[1].set_ylabel("MSE")
        axs[1].set_xlabel("Peak Threshold")
        axs[1].set_xscale("log")

        plt.tight_layout()
        plt.show()



def preprocess_powers(subset_analyzed, fs=2000, nperseg=4000, freq_range=[1, 175]):
    """
    Processes each numeric column in the DataFrame using a spectral analysis and stores the results in new DataFrames.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        fs (int): Sampling frequency of the data.
        nperseg (int): Number of samples per segment.
        freq_range (list): Frequency range as [min_freq, max_freq].

    Returns:
        pd.DataFrame, pd.DataFrame: Two DataFrames containing the power outputs and frequencies.
    """
    power_outputs = pd.DataFrame()
    freq_in = False  # Flag to check if frequency column is added

    # Loop through each column from the second column onward
    for column in subset_analyzed.columns[1:]:  # Skip the first column
        filtered_freqs, filtered_powers = preprocess_data(subset_analyzed[column].values, frequency=fs, freq_range=freq_range, nperseg=nperseg)
        new_column_name = f"Trial_{column}_power"
        
        # Only add frequency data once
        if not freq_in:
            power_outputs["filtered_frequency"] = filtered_freqs
            freq_in = True
        
        power_outputs[new_column_name] = filtered_powers

    return power_outputs


# def plot_fooof_fits(df, freq_range, max_n_peaks, aperiodic_mode, plt_log=False):
#     # Create subplots: one for each condition
#     num_conditions = len(df)
#     fig = make_subplots(rows=num_conditions, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=[f'Condition {i+1}' for i in range(num_conditions)])
# 
#     for i in range(num_conditions):
#         # Initialize storage
#         dataframe = df[i]
# 
#         filtered_freqs_np = np.array(dataframe['filtered_frequency'])
#         filtered_powers_np = np.array(dataframe['Average Power'])
# 
#         fm = SpectralModel(peak_width_limits=freq_range, max_n_peaks=max_n_peaks, aperiodic_mode=aperiodic_mode)
#         fm.fit(filtered_freqs_np, filtered_powers_np, freq_range)
# 
#         # Get the model fit and add it as a trace
#         model_freqs, model_fit, aperiodic_fit = fm.freqs, fm.get_model(), fm._ap_fit
#         x_freqs = np.log10(model_freqs) if plt_log else model_freqs
# 
#         # Add traces to the corresponding subplot
#         fig.add_trace(go.Scatter(x=x_freqs, y=model_fit,
#                                  mode='lines',
#                                  name=f'Full Model Fit - Condition {i+1}',
#                                  line=dict(color='red')),
#                       row=i+1, col=1)
# 
#         fig.add_trace(go.Scatter(x=x_freqs, y=aperiodic_fit,
#                                  mode='lines',
#                                  name=f'Aperiodic Fit - Condition {i+1}',
#                                  line=dict(color='blue', dash='dash')),
#                       row=i+1, col=1)
# 
#         # Add the original data as a baseline
#         fig.add_trace(go.Scatter(x=x_freqs, y=fm.get_data(),
#                                  mode='lines',
#                                  name=f'Original Spectrum - Condition {i+1}',
#                                  line=dict(color='black')),
#                       row=i+1, col=1)
# 
#     # Customize the layout
#     fig.update_layout(
#         height=400 * num_conditions,  # Adjust the figure height based on number of conditions
#         template='plotly_white',
#         showlegend=True
#     )
#     for i in range(0, num_conditions):
#       fig.update_xaxes(title_text='log(Frequency)' if plt_log else 'Frequency', row=i+1, col=1)
#       fig.update_yaxes(title_text='Power', row=i+1, col=1)
# 
#     return fig


def new_fit_fooof(df, freq_range = [1,300],
               max_n_peaks = 10000, aperiodic_mode = "fixed", plt_log = False):

   # Initialize model object
   for i in range(0, len(df)):
     dataframe = df[i]
     filtered_freqs_np = np.array(dataframe['filtered_frequency'])
     filtered_powers_np = np.array(dataframe['Average Power'])
   
     fm = SpectralModel(peak_width_limits=freq_range, max_n_peaks = max_n_peaks, aperiodic_mode = aperiodic_mode)
     print(f"--- Report - Condition {i+1} ---")
     fm.report(filtered_freqs_np, filtered_powers_np, freq_range)


# Plot all fooof fits in one graph
def plot_fooof_fits(df, freq_range, max_n_peaks, aperiodic_mode, plt_log=False):
  figures=[]
  for i in range(0, len(df)):
      # Initialize storage
      dataframe = df[i]

      filtered_freqs_np = np.array(dataframe['filtered_frequency'])
      filtered_powers_np = np.array(dataframe['Average Power'])

      fm = SpectralModel(peak_width_limits=freq_range, max_n_peaks=max_n_peaks, aperiodic_mode=aperiodic_mode)
      fm.fit(filtered_freqs_np, filtered_powers_np, freq_range)
      
      # Get the model fit and add it as a trace
      model_freqs, model_fit, aperiodic_fit = fm.freqs, fm.get_model(), fm._ap_fit
      x_freqs = np.log10(model_freqs) if plt_log else model_freqs
      
      fig = go.Figure()

      fig.add_trace(go.Scatter(x=x_freqs, y=model_fit,
                               mode='lines',
                               name=f'Full Model Fit - Condition {i+1}',
                               line=dict(color='red')))

      fig.add_trace(go.Scatter(x=x_freqs, y=aperiodic_fit,
                               mode='lines',
                               name=f'Aperiodic Fit - Condition {i+1}',
                               line=dict(color='blue', dash='dash')))

      # Add the original data as a baseline
      fig.add_trace(go.Scatter(x=x_freqs, y=fm.get_data(),
                               mode='lines',
                               name=f'Original Spectrum - Condition {i+1}',
                               line=dict(color='black')))
      # Customize the layout
      fig.update_layout(
          title=f'FOOOF Model - Condition {i+1}',
          xaxis_title='log(Frequency)' if plt_log else 'Frequency',
          yaxis_title='Power',
          template='plotly_white'
      )
      figures.append(fig)
  return figures


# Tune aperiodic_mode
# def tune_aperiodic_mode(df, freq_range, max_n_peaks, show_errors=True):
#   fig = go.Figure()
#   for i in range(0, len(df)):
#       # Initialize storage
#       dataframe = df[i]
#       r2s = []
#       errors = []
#   
#       # Initialize the Plotly figure
#       # fig = go.Figure()
#       
#       filtered_freqs_np = np.array(dataframe['filtered_frequency'])
#       filtered_powers_np = np.array(dataframe['Average Power'])
#   
#       # Define the aperiodic modes to loop over
#       aperiodic_modes = ['fixed', 'knee']
#   
#       # Loop through aperiodic modes and fit models
#       for aperiodic_mode in aperiodic_modes:
#           nm = SpectralModel(peak_width_limits=freq_range, max_n_peaks=max_n_peaks, aperiodic_mode=aperiodic_mode)
#           nm.fit(filtered_freqs_np, filtered_powers_np, freq_range)
#   
#           # Append metrics
#           r2s.append(nm.r_squared_)
#           errors.append(nm.error_)
#   
#           # Get the model fit and add it as a trace
#           model_freqs, model_fit = nm.freqs, nm.get_model()
#           fig.add_trace(go.Scatter(x=model_freqs, y=model_fit,
#                                    mode='lines',
#                                    name=f'{aperiodic_mode} (R²={nm.r_squared_:.2f}) - Condition {i+1}'))
#   
#       # Add the original data as a baseline (optional)
#       fig.add_trace(go.Scatter(x=nm.freqs, y=nm.get_data(),
#                                mode='lines', name=f'Original Spectrum - Condition {i+1}',
#                                line=dict(color='black', dash='dash')))
#     # Customize the layout
#   fig.update_layout(
#       title='Spectral Model Fits for Different Aperiodic Modes',
#       xaxis_title='Frequency (Hz)',
#       yaxis_title='Power',
#       legend_title='Model Configurations',
#       template='plotly_white'
#   )
#   return fig
# 
#   if show_errors:
#       # Graphs of R2 and Errors using Matplotlib
#       fig, axs = plt.subplots(1, 2, figsize=(16, 5))
#       modes_labels = [f'{mode} mode' for mode in aperiodic_modes]
#       
#       axs[0].plot(modes_labels, r2s, "bo-")
#       axs[0].set_title("R² vs. Aperiodic Mode")
#       axs[0].set_ylabel("R²")
#       axs[0].set_xlabel("Aperiodic Mode")
# 
#       axs[1].plot(modes_labels, errors, "ro-")
#       axs[1].set_title("MSE vs. Aperiodic Mode")
#       axs[1].set_ylabel("MSE")
#       axs[1].set_xlabel("Aperiodic Mode")
# 
#       plt.tight_layout()
#       plt.show()

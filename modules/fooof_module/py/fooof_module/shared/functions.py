import h5py
from specparam import SpectralModel, SpectralGroupModel
from scipy.signal import welch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative
import tempfile
import base64
import os
from io import BytesIO


## DEFINE FUNCTIONS FOR DATA PROCESSING PIPELINE ##
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

def plot_trials(df, conditions, Std_count=1, individual_trials=False):
    fig = go.Figure()
    
    # Use Plotly's default qualitative color palette
    color_palette = qualitative.Plotly
    
    def to_rgba(color, alpha=0.2):
      """Convert 'rgb(...)' or '#...' to rgba(...) with transparency"""
      if isinstance(color, str):
        if color.startswith('#'):
          # Hex format → convert to rgba
          color = color.lstrip('#')
          r, g, b = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
          return f'rgba({r}, {g}, {b}, {alpha})'
      elif color.startswith('rgb'):
          # Already in rgb(...) format → just add alpha
          return color.replace('rgb', 'rgba').replace(')', f',{alpha})')
    # Fallback: transparent black
      return f'rgba(0, 0, 0, {alpha})'
    
    
    for i in range(len(df)):
      dataframe = df[i]
      freq = dataframe['filtered_frequency']
      mean_log = np.log10(dataframe['Average Power'])
      std_log = np.log10(dataframe['Average Power'] + dataframe['StdDev']) - mean_log
      upper = mean_log + (Std_count * std_log)
      lower = mean_log - (Std_count * std_log)
      
      # Select color for this condition
      base_color = color_palette[i % len(color_palette)]
      fill_color = to_rgba(base_color, alpha=0.2)


      # Add the first trace (filt_powers electrode_14_avg)
      fig.add_trace(go.Scatter(
          x=freq,
          y=mean_log,
          mode='lines',
          name=f'Average Power - {conditions[i]}',
          line=dict(width=2, color=base_color)
      ))

  
      # Plot the upper bound (invisible but used to define area)
      fig.add_trace(go.Scatter(
          x=freq,
          y=upper,
          mode='lines',
          line=dict(width=0),
          showlegend=False,
          hoverinfo='skip'
      ))
      
      
      # Plot the lower bound and fill to upper
      fig.add_trace(go.Scatter(
          x=freq,
          y=lower,
          mode='lines',
          fill='tonexty',
          fillcolor=fill_color,
          line=dict(width=0),
          name=f'±{Std_count} Std Dev - {conditions[i]}',
          hoverinfo='skip'
      ))
  
      if individual_trials:
          # Plot each column except 'filtered_frequency', 'Average Power', and 'StdDev'
          for col in dataframe.columns:
              if col not in ['filtered_frequency', 'Average Power', 'StdDev']:
                  fig.add_trace(go.Scatter(
                      x=freq,
                      y=np.log(dataframe[col]),
                      mode='lines',
                      name=col
                  ))
    
    # Update layout for grid and other settings
    fig.update_layout(
        title='log<sub>10</sub>(Power) vs Frequency (Individual Trials)' if individual_trials else 'log<sub>10</sub>(Power) vs Frequency (Average)',
        xaxis_title='Frequency',
        yaxis_title='log<sub>10</sub>(Power)',
        showlegend=True,
        template='plotly_white',
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
  
    # Show the figure
    return fig

def tune_max_n_peaks(df, freq_range, aperiodic_mode, peaks_range, conditions, show_errors=True):
  fig = go.Figure()
  error_figs_base64 = []
  failed_conditions = []
  
  for i in range(0, len(df)):
    dataframe = df[i]
    r2s = []
    errors = []
    filtered_freqs_np = np.array(dataframe['filtered_frequency'])
    filtered_powers_np = np.array(dataframe['Average Power'])
    #model_fail = None
    
    # Loop through peak values and fit models
    for max_n_peaks in peaks_range:
      nm = SpectralModel(peak_width_limits=freq_range, max_n_peaks=max_n_peaks, aperiodic_mode=aperiodic_mode)
      try:
        nm.fit(filtered_freqs_np, filtered_powers_np, freq_range=freq_range)
        #model_fail = None
  
        # Append metrics
        r2s.append(nm.r_squared_)
        errors.append(nm.error_)
  
        # Get the model fit and add it as a trace
        model_freqs, model_fit = nm.freqs, nm.get_model()
        fig.add_trace(go.Scatter(x=model_freqs, y=model_fit,
                                mode='lines',
                                name=f'{max_n_peaks} peaks (R²={nm.r_squared_:.2f}) - {conditions[i]}'))
      except Exception as e:
        #model_fail = "Model fitting failed. Please choose your settings again."
        if i + 1 not in failed_conditions:
          failed_conditions.append(i + 1)
      

    fig.add_trace(go.Scatter(x=nm.freqs, y=nm.get_data(),
                            mode='lines', name=f'Original Spectrum - {conditions[i]}',
                            line=dict(color='black', dash='dash')))
                            
    if show_errors:
      fig_r2, ax_r2 = plt.subplots()
      ax_r2.plot(peaks_range, r2s, "bo-")
      ax_r2.set_title(f"R² - {conditions[i]}")
      ax_r2.set_ylabel("R²")
      ax_r2.set_xlabel("Number of Peaks")
      plt.tight_layout()
      
      buf_r2 = BytesIO()
      fig_r2.savefig(buf_r2, format='png')
      buf_r2.seek(0)
      error_figs_base64.append(base64.b64encode(buf_r2.read()).decode('utf-8'))
      buf_r2.close()
      plt.close(fig_r2)
      
      fig_err, ax_err = plt.subplots()
      ax_err.plot(peaks_range, errors, "ro-")
      ax_err.set_title(f"MSE - {conditions[i]}")
      ax_err.set_ylabel("MSE")
      ax_err.set_xlabel("Number of Peaks")
      plt.tight_layout()

      buf_err = BytesIO()
      fig_err.savefig(buf_err, format='png')
      buf_err.seek(0)
      error_figs_base64.append(base64.b64encode(buf_err.read()).decode('utf-8'))
      buf_err.close()
      plt.close(fig_err)

  # Customize the layout
  fig.update_layout(
      title='Spectral Model Fits Across Different Peak Numbers',
      xaxis_title='Frequency (Hz)',
      yaxis_title='log<sub>10</sub>(Power)',
      legend_title='Model',
      template='plotly_white'
  )

  model_fail_message = None
  if failed_conditions:
      model_fail_message = f"Model fitting failed for condition(s): {', '.join(map(str, failed_conditions))}. Please adjust the max n peaks tuning parameter(s)."
  
  return {'plotly': fig, 'matplotlib': error_figs_base64, 'model_fail': model_fail_message}



def tune_aperiodic_mode(df, freq_range, max_n_peaks, conditions, show_errors=True):
    fig = go.Figure()
    aperiodic_modes = ['fixed', 'knee']
    error_figs_base64 = []
    failed_conditions = []

    for i, dataframe in enumerate(df):
        filtered_freqs_np = np.array(dataframe['filtered_frequency'])
        filtered_powers_np = np.array(dataframe['Average Power'])

        r2s = []
        errors = []
        #model_fail = None

        for aperiodic_mode in aperiodic_modes:
            nm = SpectralModel(peak_width_limits=freq_range, max_n_peaks=max_n_peaks, aperiodic_mode=aperiodic_mode)
            try:
              nm.fit(filtered_freqs_np, filtered_powers_np, freq_range)
              #model_fail = None
              r2s.append(nm.r_squared_)
              errors.append(nm.error_)
  
              model_freqs, model_fit = nm.freqs, nm.get_model()
              fig.add_trace(go.Scatter(
                  x=model_freqs, y=model_fit,
                  mode='lines',
                  name=f'{aperiodic_mode} (R²={nm.r_squared_:.2f}) - {conditions[i]}'
              ))
            except Exception as e:
              #model_fail = "Model fitting failed. Please choose your settings again."
              if i + 1 not in failed_conditions:
                failed_conditions.append(i + 1)

        fig.add_trace(go.Scatter(
            x=nm.freqs, y=nm.get_data(),
            mode='lines', name=f'Original Spectrum - {conditions[i]}',
            line=dict(color='black', dash='dash')
        ))

        if show_errors:
            # R² plot
            fig_r2, ax_r2 = plt.subplots()
            ax_r2.plot(aperiodic_modes, r2s, "bo-")
            ax_r2.set_title(f"R² - {conditions[i]}")
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
            ax_err.set_title(f"MSE - {conditions[i]}")
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
        yaxis_title='log<sub>10</sub>(Power)',
        legend_title='Model Configurations',
        template='plotly_white'
    )
    
    model_fail_message = None
    if failed_conditions:
        model_fail_message = f"Model fitting failed for condition(s): {', '.join(map(str, failed_conditions))}. Please adjust the aperiodic mode tuning parameter(s)."

    return {'plotly': fig, 'matplotlib': error_figs_base64, 'model_fail': model_fail_message}


        
def tune_peak_threshold(df, freq_range, max_n_peaks, aperiodic_mode, start1, stop1, num1, conditions, show_errors=True):
    fig = go.Figure()
    error_figs_base64 = []
    peak_threshold_range = np.logspace(start = np.log10(start1), stop = np.log10(stop1), num = num1)
    failed_conditions = []
    
    for i in range(0, len(df)):
      dataframe = df[i]
      r2s = []
      errors = []
      filtered_freqs_np = np.array(dataframe['filtered_frequency'])
      filtered_powers_np = np.array(dataframe['Average Power'])
      #model_fail = None

      # Loop through peak thresholds and fit models
      for peak_threshold in peak_threshold_range:
          nm = SpectralModel(peak_width_limits=freq_range, max_n_peaks=max_n_peaks, aperiodic_mode=aperiodic_mode, peak_threshold=peak_threshold)
          try:
            nm.fit(filtered_freqs_np, filtered_powers_np, freq_range=freq_range)
            #model_fail = None
  
            # Append metrics
            r2s.append(nm.r_squared_)
            errors.append(nm.error_)
  
            # Get the model fit and add it as a trace
            model_freqs, model_fit = nm.freqs, nm.get_model()  # Assuming the attributes are named like this
            fig.add_trace(go.Scatter(x=model_freqs, y=model_fit,
                                    mode='lines',
                                    name=f'PT {peak_threshold:.3g} (R²={nm.r_squared_:.2f}) - {conditions[i]}'))
          except Exception as e:
            #model_fail = "Model fitting failed. Please choose your settings again."
            if i + 1 not in failed_conditions:
              failed_conditions.append(i + 1)

      # Add the original data as a baseline (optional)
      fig.add_trace(go.Scatter(x=nm.freqs, y=nm.get_data(),
                              mode='lines', name=f'Original Spectrum - {conditions[i]}',
                              line=dict(color='black', dash='dash')))
                              
      if show_errors:
        fig_r2, ax_r2 = plt.subplots()
        ax_r2.plot(peak_threshold_range, r2s, "bo-")
        ax_r2.set_title(f"R² - {conditions[i]}")
        ax_r2.set_ylabel("R²")
        ax_r2.set_xlabel("Peak Threshold")
        plt.tight_layout()
        
        buf_r2 = BytesIO()
        fig_r2.savefig(buf_r2, format='png')
        buf_r2.seek(0)
        error_figs_base64.append(base64.b64encode(buf_r2.read()).decode('utf-8'))
        buf_r2.close()
        plt.close(fig_r2)
        
        fig_err, ax_err = plt.subplots()
        ax_err.plot(peak_threshold_range, errors, "ro-")
        ax_err.set_title(f"MSE - {conditions[i]}")
        ax_err.set_ylabel("MSE")
        ax_err.set_xlabel("Peak Threshold")
        plt.tight_layout()
        
        buf_err = BytesIO()
        fig_err.savefig(buf_err, format='png')
        buf_err.seek(0)
        error_figs_base64.append(base64.b64encode(buf_err.read()).decode('utf-8'))
        buf_err.close()
        plt.close(fig_err)
        
    # Customize the layout
    fig.update_layout(
        title='Spectral Model Fits Across Different Peak Thresholds (PTs)',
        xaxis_title='Frequency (Hz)',
        yaxis_title='log<sub>10</sub>(Power)',
        legend_title='Model Configurations',
        template='plotly_white'
    )
    
    model_fail_message = None
    if failed_conditions:
        model_fail_message = f"Model fitting failed for condition(s): {', '.join(map(str, failed_conditions))}. Please adjust the peak threshold tuning parameter(s)."
    
    return {'plotly': fig, 'matplotlib': error_figs_base64, 'model_fail': model_fail_message}

def new_fit_fooof(df, conditions, freq_range = [1,300],
               max_n_peaks = 10000, aperiodic_mode = "fixed", plt_log = False):

   # Initialize model object
   for i in range(0, len(df)):
     dataframe = df[i]
     filtered_freqs_np = np.array(dataframe['filtered_frequency'])
     filtered_powers_np = np.array(dataframe['Average Power'])
   
     fm = SpectralModel(peak_width_limits=freq_range, max_n_peaks = max_n_peaks, aperiodic_mode = aperiodic_mode)
     print(f"--- Report - {conditions[i]} ---")
     fm.report(filtered_freqs_np, filtered_powers_np, freq_range)


# Plot all fooof fits in one graph
def plot_fooof_fits(df, freq_range, max_n_peaks, aperiodic_mode, conditions, plt_log=False):#, condition_names=None):
  figures=[]
  for i in range(0, len(df)):
      # Initialize storage
      dataframe = df[i]
      center_frequencies = []
      power_width = []
      band_width = []

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
                               name=f'Full Model Fit - {conditions[i]}',
                               line=dict(color='red')))

      fig.add_trace(go.Scatter(x=x_freqs, y=aperiodic_fit,
                               mode='lines',
                               name=f'Aperiodic Fit - {conditions[i]}',
                               line=dict(color='blue', dash='dash')))

      # Add the original data as a baseline
      fig.add_trace(go.Scatter(x=x_freqs, y=fm.get_data(),
                               mode='lines',
                               name=f'Original Spectrum - {conditions[i]}',
                               line=dict(color='black')))
      
      model_peak_params = fm.get_params('peak_params')
      for j in range(0, len(model_peak_params)):
        center_frequencies.append(model_peak_params[j][0])
      for j in range(0, len(model_peak_params)):
        power_width.append(model_peak_params[j][1])
      for j in range(0, len(model_peak_params)):
        band_width.append(model_peak_params[j][2])
        
      for idx, peak in enumerate(center_frequencies):
            peak_x = np.log10(peak) if plt_log else peak
            bandwidth = band_width[idx]
            powerwidth = power_width[idx]
            fig.add_shape(
                type='line',
                x0=peak_x, x1=peak_x,
                y0=min(model_fit)-1, y1=max(model_fit)+0.5,
                line=dict(color="green", dash="dot", width=2)
            )
            
            fig.add_trace(go.Scatter(
                x=[peak_x], 
                y=[max(model_fit) + 1],
                text=[f'CF: {peak:.2f}'],
                mode='text',
                textposition='top center',
                textfont=dict(size=6),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=[peak_x], 
                y=[max(model_fit) + 0.8],
                text=[f'PW: {powerwidth:.2f}'],
                mode='text',
                textposition='top center',
                textfont=dict(size=6),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=[peak_x], 
                y=[max(model_fit) + 0.6],
                text=[f'BW: {bandwidth:.2f}'],
                mode='text',
                textposition='top center',
                textfont=dict(size=6),
                showlegend=False
            ))
        
      # Customize the layout
      fig.update_layout(
          title=f"FOOOF Model - {conditions[i]}",
          xaxis_title='log<sub>10</sub>(Frequency)' if plt_log else 'Frequency',
          yaxis_title='log<sub>10</sub>(Power)',
          template='plotly_white'
      )
      figures.append(fig)
  return figures

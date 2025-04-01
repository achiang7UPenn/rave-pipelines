# How to save key-value pairs to settings.yaml (pipeline inputs)
settings = list(
  project_name = "demo",
  subject_code = "DemoSubject",
  channels_to_load = 14:15,
  epoch_name = "auditory_onset",
  reference_name = "default",
  time_window_to_load = c(-1, 2),
  analyze_channel = 14
)
raveio::save_yaml(settings, "C:/Users/roger/OneDrive/Desktop/Ese 4510/Code/rave-pipelines/modules/fooof_module/settings.yaml")


# YAB
# raveio::install_subject("YAB")

# Start RAVE
# rave::start_rave()

# Load epoch
repository = raveio::prepare_subject_voltage_with_epoch(
  subject = "demo/DemoSubject",
  electrodes = 14,
  epoch_name = "auditory_onset",
  reference_name = "default",
  time_windows = c(-1, 2)
)
repository$subject$raw_sample_rates
repository$voltage$data_list$e_14[drop = FALSE]



# enter python mode
rpymat::repl_python()

# Exit Python back to R
# >>> exit



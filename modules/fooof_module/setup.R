# Make sure you install dependency first before start development
# rpymat::add_packages(c("specparam", "plotly", "h5py", "matplotlib", "pandas"), pip = TRUE)
# install.packages("plotly")

# Debug UI
ravedash::debug_modules()

# Run pipeline without UI
pipeline <- raveio::pipeline("fooof_module", paths = file.path(rstudioapi::getActiveProject(), "modules"))
pipeline$visualize()

power_outputs <- pipeline$run("power_outputs")


repository <- pipeline$run("repository")

# For debug purposes, run
# pipeline <- raveio::pipeline("fooof_module", paths = file.path(rstudioapi::getActiveProject(), "modules"))
# power_outputs <- pipeline$run("power_outputs")

# individual_trials = False
# shared.plot_trials(power_outputs, individual_trials=individual_trials)

# shared <- pipeline$python_module(type = "shared")
# shared$plot_trials(power_outputs, individual_trials = FALSE)

# Make sure you install dependency first before start development
# rpymat::add_packages(c("specparam", "plotly"), pip = TRUE)
# install.packages("plotly")

ravedash::debug_modules()

pipeline <- raveio::pipeline("fooof_module", paths = file.path(rstudioapi::getActiveProject(), "modules"))
pipeline$visualize()

power_outputs <- pipeline$run("power_outputs")


repository <- pipeline$run("repository")

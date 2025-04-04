module_server <- function(input, output, session, ...){


  # Local reactive values, used to store reactive event triggers
  local_reactives <- shiny::reactiveValues(
    update_outputs = NULL
  )

  # Local non-reactive values, used to store static variables
  local_data <- dipsaus::fastmap2()

  # get server tools to tweek
  server_tools <- get_default_handlers(session = session)

  # Run analysis once the following input IDs are changed
  # This is used by auto-recalculation feature
  # server_tools$run_analysis_onchange(
  #   component_container$get_input_ids(c(
  #     "electrode_text", "baseline_choices",
  #     "analysis_ranges", "condition_groups"
  #   ))
  # )

  # Register event: main pipeline need to run
  # When the users click on the "Run Analysis" button
  shiny::bindEvent(
    ravedash::safe_observe({

      # Step 1: collect user's inputs
      # Step 2: use pipeline$set_settings to save user inputs to settings.yaml
      # Step 3: use pipeline$run to get the results
      # Step 4: Tell outputs to update

      # Step 1: Collect input data
      settings <- component_container$collect_settings(ids = c(
        "electrode_text",
        # "baseline_choices",
        "condition_groups"
        # "analysis_ranges"
      ))

      # print(as.list(settings))

      # {
      #   "condition_groups": [
      #     {
      #       "group_name" : "All Conditions",
      #       "group_conditions": [
      #         "AdriveVlast",
      #         "AknownVmeant",
      #         ...
      #       ]
      #     }
      #   ],
      #   "analysis_electrodes": "13"
      # }
      #
      # -->
      #
      # {
      #   "condition_groupings": [
      #     {
      #       "label": ...
      #       "conditions",
      #     }
      #   ],
      #   "analyze_channel": 14
      # }

      pipeline_settings <- list(
        analyze_channel = dipsaus::parse_svec(settings$analysis_electrodes),

        # [fun(x) for x in li]
        # lapply(li, function(x) {
        #   ...
        # })
        condition_groupings = lapply(settings$condition_groups, function(group) {
          label <- group$group_name
          conditions <- group$group_conditions
          list(
            label = label,
            conditions = conditions
          )
        }),

        # Customized inputs
        window_length = input$fooof_winlen,
        freq_range = c(1, input$fooof_freq_range),
        max_n_peaks = input$fooof_max_n_peaks,
        aperiodic_mode = input$fooof_aperiodic_mode,
        plt_log = input$fooof_bool,
        freq_range_aperiodic_tuning = c(1, input$freq_range_tuning_aperiodic_mode),
        max_n_peaks_aperiodic_tuning = input$max_n_peaks_tuning_aperiodic_mode
      )

      # Step 2: save user inputs to settings.yaml
      pipeline$set_settings(.list = pipeline_settings)

      # Step 3: run analysis to get the results
      results <- pipeline$run(
        names = c('power_outputs', 'fitted_fooof', "power_outputs_list"),

        scheduler = "none",
        type = "vanilla"
      )
      # env <- pipeline$eval('fitted_fooof')

      # Step 4: Save the results and sell outputs to update
      local_data$power_outputs <- results$power_outputs
      local_data$fitted_fooof <- results$fitted_fooof
      local_data$power_outputs_list <- results$power_outputs_list
      local_reactives$update_outputs <- Sys.time()

      return()

    }),
    server_tools$run_analysis_flag(),
    ignoreNULL = TRUE, ignoreInit = TRUE
  )


  # Check whether the loaded data is valid
  shiny::bindEvent(
    ravedash::safe_observe({
      loaded_flag <- ravedash::watch_data_loaded()
      if(!loaded_flag){ return() }
      # Check whether the repository is properly loaded
      # No need to change
      new_repository <- pipeline$read("repository")
      if(!inherits(new_repository, "rave_prepare_subject_voltage_with_epoch")){
        ravedash::logger("Repository read from the pipeline, but it is not an instance of `rave_prepare_subject_voltage_with_epoch`. Abort initialization", level = "warning")
        return()
      }
      ravedash::logger("Repository read from the pipeline; initializing the module UI", level = "debug")

      # check if the repository has the same subject as current one
      old_repository <- component_container$data$repository
      if(inherits(old_repository, "rave_prepare_subject_voltage_with_epoch")){

        if( !attr(loaded_flag, "force") &&
            identical(old_repository$signature, new_repository$signature) ){
          ravedash::logger("The repository data remain unchanged ({new_repository$subject$subject_id}), skip initialization", level = "debug", use_glue = TRUE)
          return()
        }
      }

      # TODO: reset UIs to default
      condition_groupings <- pipeline$get_settings("condition_groupings")
      all_conditions <- unique(new_repository$epoch_table$Condition)
      condition_groupings <- dipsaus::drop_nulls(lapply(condition_groupings, function(group) {
        group_conditions = unlist(group$conditions)
        group_conditions <- group_conditions[group_conditions %in% all_conditions]
        if(!length(group_conditions)) {
          return(NULL)
        }
        list(
          group_name = group$label,
          group_conditions = group_conditions
        )
      }))
      if(length(condition_groupings)) {
        dipsaus::updateCompoundInput2(
          session = session,
          inputId = "condition_groups",
          value = condition_groupings,
          ncomp = length(condition_groupings)
        )
      }

      # Update window length
      window_length <- pipeline$get_settings("window_length")
      if(isTRUE(window_length > 0)) {
        shiny::updateSliderInput(session = session,
                                 inputId = "fooof_winlen",
                                 value = window_length)
      }

      # Reset preset UI & data
      component_container$reset_data()
      component_container$data$repository <- new_repository
      component_container$initialize_with_new_data()

      # Reset outputs
      # shidashi::reset_output("collapse_over_trial")

    }, priority = 1001),
    ravedash::watch_data_loaded(),
    ignoreNULL = FALSE,
    ignoreInit = FALSE
  )

  # Register outputs
  # output$collapse_over_trial <- shiny::renderUI({
  output$collapse_over_trial <- shiny::renderUI({
    shiny::validate(
      shiny::need(
        length(local_reactives$update_outputs) &&
          !isFALSE(local_reactives$update_outputs),
        message = "Please run the module first"
      )
    )

    # retrieve the `power_outputs` from `local_data`
    power_outputs <- local_data$power_outputs
    power_outputs_list <- local_data$power_outputs_list

    # For debug purposes, run
    # pipeline <- raveio::pipeline("fooof_module", paths = "/Users/dipterix/Dropbox (Personal)/projects/rave-pipeline-ese2025/modules/")
    # power_outputs <- pipeline$read("power_outputs")

    # individual_trials = False
    # shared.plot_trials(power_outputs, individual_trials=individual_trials)

    shared <- pipeline$python_module(type = "shared")
    plot <- shared$plot_trials(power_outputs_list, individual_trials = FALSE)

    return(shiny::HTML(rpymat::py_to_r(plot$to_html())))
  })

  output$individual_trials_plot <- shiny::renderUI({
    shiny::validate(
      shiny::need(
        length(local_reactives$update_outputs) &&
          !isFALSE(local_reactives$update_outputs),
        message = "Please run the module first"
      )
    )

    # retrieve the `power_outputs` from `local_data`
    power_outputs <- local_data$power_outputs
    power_outputs_list <- local_data$power_outputs_list

    # For debug purposes, run
    # pipeline <- raveio::pipeline("fooof_module", paths = "/Users/dipterix/Dropbox (Personal)/projects/rave-pipeline-ese2025/modules/")
    # power_outputs <- pipeline$read("power_outputs")

    # individual_trials = False
    # shared.plot_trials(power_outputs, individual_trials=individual_trials)

    shared <- pipeline$python_module(type = "shared")
    plot <- shared$plot_trials(power_outputs_list, individual_trials = TRUE)

    return(shiny::HTML(rpymat::py_to_r(plot$to_html())))
  })


  output$fooof_print_results <- shiny::renderPrint({
    shiny::validate(
      shiny::need(
        length(local_reactives$update_outputs) &&
          !isFALSE(local_reactives$update_outputs),
        message = "Please run the module first"
      )
    )

    # retrieve the `power_outputs` from `local_data`
    fitted_fooof <- local_data$fitted_fooof

    # For debug purposes, run
    # pipeline <- raveio::pipeline("fooof_module", paths = file.path(rstudioapi::getActiveProject(), "modules/"))
    # fitted_fooof <- pipeline$read("fitted_fooof")

    model <- fitted_fooof$model
    frequencies <- fitted_fooof$frequencies
    power <- fitted_fooof$power
    freq_range <- fitted_fooof$freq_range
    # max_n_peaks <- fitted_fooof$max_n_peaks
    # aperiodic_mode <- fitted_fooof$aperiodic_mode
    # plt_log <- pipeline_settings$plt_log
    # plt_log <- input$fooof_bool

    # shared <- pipeline$python_module(type = "shared")
    report <- reticulate::py_capture_output({
      model$report(frequencies, power, freq_range = freq_range)
    })
    cat(report)
  })

  output$fooof_plot_results <- shiny::renderImage({
    shiny::validate(
      shiny::need(
        length(local_reactives$update_outputs) &&
          !isFALSE(local_reactives$update_outputs),
        message = "Please run the module first"
      )
    )

    # retrieve the `power_outputs` from `local_data`
    fitted_fooof <- local_data$fitted_fooof
    plt_log <- !isFALSE(input$fooof_bool)
    plt <- rpymat::import("matplotlib.pyplot", as = "plt")

    # For debug purposes, run
    # pipeline <- raveio::pipeline("fooof_module", paths = file.path(rstudioapi::getActiveProject(), "modules/"))
    # fitted_fooof <- pipeline$read("fitted_fooof")

    model <- fitted_fooof$model
    model$plot(plt_log = plt_log)
    plt$title(sprintf("Condition Analyzed: %s", "my condition"))

    # Create a temporary file to save the plot
    outfile <- normalizePath(tempfile(fileext = ".png"), winslash = "/", mustWork = FALSE)
    plt$savefig(outfile)
    plt$close()
    list(src = outfile,
         contentType = "image/png",
         width = "80%",
         height = "100%",
         alt = "Placeholder for fooof fit plot")
  }, deleteFile = TRUE)

  output$aperiodic_tuning_part <- shiny::renderUI({
    shiny::validate(
      shiny::need(
        length(local_reactives$update_outputs) &&
          !isFALSE(local_reactives$update_outputs),
        message = "Please run the module first"
      )
    )

    # retrieve the `power_outputs` from `local_data`
    power_outputs <- local_data$power_outputs
    power_outputs_list <- local_data$power_outputs_list

    pipeline_settings <- pipeline$get_settings()

    freq_range_aperiodic_tuning <- pipeline_settings$freq_range_aperiodic_tuning
    max_n_peaks_aperiodic_tuning <- pipeline_settings$max_n_peaks_aperiodic_tuning

    shared <- pipeline$python_module(type = "shared")
    plot <- shared$tune_aperiodic_mode(power_outputs_list, freq_range_aperiodic_tuning, max_n_peaks_aperiodic_tuning, show_errors=TRUE)

    return(shiny::HTML(rpymat::py_to_r(plot$to_html())))
  })

}


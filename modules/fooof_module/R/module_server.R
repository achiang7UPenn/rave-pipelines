module_server <- function(input, output, session, ...){


  # Local reactive values, used to store reactive event triggers
  local_reactives <- shiny::reactiveValues(
    update_outputs = NULL
  )

  # Local non-reactive values, used to store static variables
  local_data <- dipsaus::fastmap2()

  # get server tools to tweek
  server_tools <- get_default_handlers(session = session)



  observe({
    input_range <- input$threshold_value_range_tuning_peak_threshold
    step <- 0.1
    if (is.null(input_range)) return()

    lower <- input_range[1]
    upper <- input_range[2]

    # If lower bound is >= upper, adjust accordingly
    if (lower == upper) {
      if ((upper + step) <= 10) {
        # Upper bound approaches lower bound: increment upper bound
        new_upper <- max(upper + step, lower + step)
        updateSliderInput(session,
                          inputId = "threshold_value_range_tuning_peak_threshold",
                          value = c(lower, new_upper))
      } else if ((upper + step) > 10) {
        new_lower <- min(upper - step, lower - step)
        updateSliderInput(session,
                          inputId = "threshold_value_range_tuning_peak_threshold",
                          value = c(new_lower, upper))
      }
    }
  })

  observe({
    input_range <- input$fooof_freq_range
    step <- 5
    if (is.null(input_range)) return()

    lower <- input_range[1]
    upper <- input_range[2]

    # If lower bound is >= upper, adjust accordingly
    if (upper == lower) {
      if ((upper + step) <= 300) {
        # Upper bound approaches lower bound: increment upper bound
        new_upper <- max(upper + step, lower + step)
        updateSliderInput(session,
                          inputId = "fooof_freq_range",
                          value = c(lower, new_upper))
      } else if ((upper + step) > 300) {
        new_lower <- min(upper - step, lower - step)
        updateSliderInput(session,
                          inputId = "fooof_freq_range",
                          value = c(new_lower, upper))
      }
    }
  })

  observe({
    input_range <- input$freq_range_tuning_max_n_peaks
    step <- 5
    if (is.null(input_range)) return()

    lower <- input_range[1]
    upper <- input_range[2]

    # If lower bound is >= upper, adjust accordingly
    if (upper == lower) {
      if ((upper + step) <= 300) {
        # Upper bound approaches lower bound: increment upper bound
        new_upper <- max(upper + step, lower + step)
        updateSliderInput(session,
                          inputId = "freq_range_tuning_max_n_peaks",
                          value = c(lower, new_upper))
      } else if ((upper + step) > 300) {
        new_lower <- min(upper - step, lower - step)
        updateSliderInput(session,
                          inputId = "freq_range_tuning_max_n_peaks",
                          value = c(new_lower, upper))
      }
    }
  })

  observe({
    input_range <- input$freq_range_tuning_aperiodic_mode
    step <- 5
    if (is.null(input_range)) return()

    lower <- input_range[1]
    upper <- input_range[2]

    # If lower bound is >= upper, adjust accordingly
    if (upper == lower) {
      if ((upper + step) <= 300) {
        # Upper bound approaches lower bound: increment upper bound
        new_upper <- max(upper + step, lower + step)
        updateSliderInput(session,
                          inputId = "freq_range_tuning_aperiodic_mode",
                          value = c(lower, new_upper))
      } else if ((upper + step) > 300) {
        new_lower <- min(upper - step, lower - step)
        updateSliderInput(session,
                          inputId = "freq_range_tuning_aperiodic_mode",
                          value = c(new_lower, upper))
      }
    }
  })

  observe({
    input_range <- input$freq_range_tuning_peak_threshold
    step <- 5
    if (is.null(input_range)) return()

    lower <- input_range[1]
    upper <- input_range[2]

    # If lower bound is >= upper, adjust accordingly
    if (upper == lower) {
      if ((upper + step) <= 300) {
        # Upper bound approaches lower bound: increment upper bound
        new_upper <- max(upper + step, lower + step)
        updateSliderInput(session,
                          inputId = "freq_range_tuning_peak_threshold",
                          value = c(lower, new_upper))
      } else if ((upper + step) > 300) {
        new_lower <- min(upper - step, lower - step)
        updateSliderInput(session,
                          inputId = "freq_range_tuning_peak_threshold",
                          value = c(new_lower, upper))
      }
    }
  })

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
        individual_trials = input$fooof_ind_trials,
        window_length = input$fooof_winlen,
        freq_range = input$fooof_freq_range,
        max_n_peaks = input$fooof_max_n_peaks,
        aperiodic_mode = input$fooof_aperiodic_mode,
        plt_log = input$fooof_bool,
        freq_range_aperiodic_tuning = input$freq_range_tuning_aperiodic_mode,
        max_n_peaks_aperiodic_tuning = input$max_n_peaks_tuning_aperiodic_mode,
        freq_range_tuning_max_n_peaks = input$freq_range_tuning_max_n_peaks,
        aperiodic_mode_tuning_max_n_peaks = input$aperiodic_mode_tuning_max_n_peaks,
        peaks_range_tuning_max_n_peaks = seq(input$peaks_range_tuning_max_n_peaks[1], input$peaks_range_tuning_max_n_peaks[2]),
        freq_range_tuning_peak_threshold = input$freq_range_tuning_peak_threshold,
        max_n_peaks_tuning_peak_threshold = input$max_n_peaks_tuning_peak_threshold,
        aperiodic_mode_tuning_peak_threshold = input$aperiodic_mode_tuning_peak_threshold,
        threshold_value_range_tuning_peak_threshold = input$threshold_value_range_tuning_peak_threshold,
        number_of_threshold_value_tuning_peak_threshold = input$number_of_threshold_value_tuning_peak_threshold
        # start = input$threshold_value_range_tuning_peak_threshold[1],
        # stop = input$threshold_value_range_tuning_peak_threshold[2],
        # num = input$number_of_threshold_value_tuning_peak_threshold,
        # value_list = numeric(num),
        # for (i in 1:num) {
        #   value_list[i] = 10 * ((start + i * (stop - start)) / (num - 1))
        # }

        # # Initialize an empty list to store the values
        # value_list <- numeric(num)
        #
        # # Calculate the values based on the formula
        # for (i in 1:num) {
        #   value_list[i] <- 10 * ((start + i * (stop - start)) / (num - 1))
        # }
      )

      # Step 2: save user inputs to settings.yaml
      pipeline$set_settings(.list = pipeline_settings)

      dipsaus::shiny_alert2(
        title = "Running",
        text = "Applying fooof... Please wait. The results will be displayed momentarily.",
        auto_close = FALSE,
        buttons = FALSE,
        icon = "info"
      )

      on.exit({
        Sys.sleep(0.5)
        dipsaus::close_alert2()
      }, add = TRUE, after = FALSE)

      # Step 3: run analysis to get the results
      results <- pipeline$run(
        names = c('power_outputs', 'fitted_fooof', 'power_outputs_list', 'conditions_analyzed_1'),

        scheduler = "none",
        type = "vanilla"
      )
      # env <- pipeline$eval('fitted_fooof')

      # Step 4: Save the results and sell outputs to update
      local_data$power_outputs <- results$power_outputs
      local_data$fitted_fooof <- results$fitted_fooof
      local_data$power_outputs_list <- results$power_outputs_list
      local_data$conditions_analyzed_1 <- results$conditions_analyzed_1
      local_reactives$update_outputs <- Sys.time()

      Sys.sleep(0.5)
      dipsaus::close_alert2()
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

      individual_trials <- pipeline$get_settings("individual_trials")
      if(is.logical(individual_trials)) {
        shiny::updateCheckboxInput(session = session,
                                   inputId = "fooof_ind_trials",
                                   value = individual_trials)
      }

      # Update window length
      window_length <- pipeline$get_settings("window_length")
      if(isTRUE(window_length > 0)) {
        shiny::updateSliderInput(session = session,
                                 inputId = "fooof_winlen",
                                 value = window_length)
      }

      freq_range <- pipeline$get_settings("freq_range")
      if(isTRUE(freq_range[1] > 0)) {
        shiny::updateSliderInput(session = session,
                                 inputId = "fooof_freq_range",
                                 value = c(freq_range[1], freq_range[2]))
      }

      max_n_peaks <- pipeline$get_settings("max_n_peaks")
      if(isTRUE(max_n_peaks > 0)) {
        shiny::updateSliderInput(session = session,
                                 inputId = "fooof_max_n_peaks",
                                 value = max_n_peaks)
      }

      aperiodic_mode <- pipeline$get_settings("aperiodic_mode")
      if(aperiodic_mode == 'fixed' || aperiodic_mode == 'knee') {
        shiny::updateSelectInput(session = session,
                                 inputId = "fooof_aperiodic_mode",
                                 selected = aperiodic_mode)
      }

      plt_log <- pipeline$get_settings("plt_log")
      if(is.logical(plt_log)) {
        shiny::updateCheckboxInput(session = session,
                                   inputId = "fooof_bool",
                                   value = plt_log)
      }

      freq_range_tuning_max_n_peaks <- pipeline$get_settings("freq_range_tuning_max_n_peaks")
      if(isTRUE(freq_range_tuning_max_n_peaks[1] > 0)) {
        shiny::updateSliderInput(session = session,
                                 inputId = "freq_range_tuning_max_n_peaks",
                                 value = c(freq_range_tuning_max_n_peaks[1], freq_range_tuning_max_n_peaks[2]))
      }

      freq_range_aperiodic_tuning <- pipeline$get_settings("freq_range_aperiodic_tuning")
      if(isTRUE(freq_range_aperiodic_tuning[1] > 0)) {
        shiny::updateSliderInput(session = session,
                                 inputId = "freq_range_tuning_aperiodic_mode",
                                 value = c(freq_range_aperiodic_tuning[1], freq_range_aperiodic_tuning[2]))
      }

      max_n_peaks_aperiodic_tuning <- pipeline$get_settings("max_n_peaks_aperiodic_tuning")
      if(isTRUE(max_n_peaks_aperiodic_tuning > 0)) {
        shiny::updateSliderInput(session = session,
                                 inputId = "max_n_peaks_tuning_aperiodic_mode",
                                 value = max_n_peaks_aperiodic_tuning)
      }

      freq_range_tuning_peak_threshold <- pipeline$get_settings("freq_range_tuning_peak_threshold")
      if(isTRUE(freq_range_tuning_peak_threshold[1] > 0)) {
        shiny::updateSliderInput(session = session,
                                 inputId = "freq_range_tuning_peak_threshold",
                                 value = c(freq_range_tuning_peak_threshold[1], freq_range_tuning_peak_threshold[2]))
      }

      threshold_value_range_tuning_peak_threshold <- pipeline$get_settings("threshold_value_range_tuning_peak_threshold")
      if(isTRUE(threshold_value_range_tuning_peak_threshold[1] > 0)) {
        shiny::updateSliderInput(session = session,
                                 inputId = "threshold_value_range_tuning_peak_threshold",
                                 value = c(threshold_value_range_tuning_peak_threshold[1], threshold_value_range_tuning_peak_threshold[2]))
      }

      number_of_threshold_value_tuning_peak_threshold <- pipeline$get_settings("number_of_threshold_value_tuning_peak_threshold")
      if(isTRUE(number_of_threshold_value_tuning_peak_threshold > 0)) {
        shiny::updateSliderInput(session = session,
                                 inputId = "number_of_threshold_value_tuning_peak_threshold",
                                 value = number_of_threshold_value_tuning_peak_threshold)
      }

      aperiodic_mode_tuning_peak_threshold <- pipeline$get_settings("aperiodic_mode_tuning_peak_threshold")
      if(aperiodic_mode_tuning_peak_threshold == 'fixed' || aperiodic_mode_tuning_peak_threshold == 'knee') {
        shiny::updateSelectInput(session = session,
                                 inputId = "aperiodic_mode_tuning_peak_threshold",
                                 selected = aperiodic_mode_tuning_peak_threshold)
      }

      max_n_peaks_tuning_peak_threshold <- pipeline$get_settings("max_n_peaks_tuning_peak_threshold")
      if(isTRUE(max_n_peaks_tuning_peak_threshold > 0)) {
        shiny::updateSliderInput(session = session,
                                 inputId = "max_n_peaks_tuning_peak_threshold",
                                 value = max_n_peaks_tuning_peak_threshold)
      }

      peaks_range_tuning_max_n_peaks <- pipeline$get_settings("peaks_range_tuning_max_n_peaks")
      if(isTRUE(peaks_range_tuning_max_n_peaks[1] > 0)) {
        shiny::updateSliderInput(session = session,
                                 inputId = "peaks_range_tuning_max_n_peaks",
                                 value = c(peaks_range_tuning_max_n_peaks[1], peaks_range_tuning_max_n_peaks[length(peaks_range_tuning_max_n_peaks)]))
      }

      aperiodic_mode_tuning_max_n_peaks <- pipeline$get_settings("aperiodic_mode_tuning_max_n_peaks")
      if(aperiodic_mode_tuning_max_n_peaks == 'fixed' || aperiodic_mode_tuning_max_n_peaks == 'knee') {
        shiny::updateSelectInput(session = session,
                                 inputId = "aperiodic_mode_tuning_max_n_peaks",
                                 selected = aperiodic_mode_tuning_max_n_peaks)
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


  # Output for plotting average spectral power
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
    conditions_analyzed_1 <- local_data$conditions_analyzed_1

    pipeline_settings <- pipeline$get_settings()

    individual_trials <- !isFALSE(input$fooof_ind_trials)

    # For debug purposes, run
    # pipeline <- raveio::pipeline("fooof_module", paths = "/Users/dipterix/Dropbox (Personal)/projects/rave-pipeline-ese2025/modules/")
    # power_outputs <- pipeline$read("power_outputs")

    # individual_trials = False
    # shared.plot_trials(power_outputs, individual_trials=individual_trials)

    shared <- pipeline$python_module(type = "shared")
    plot <- shared$plot_trials(power_outputs_list, conditions_analyzed_1, individual_trials = individual_trials)

    return(shiny::HTML(rpymat::py_to_r(plot$to_html())))
  })

  # # Output for plotting individual trials
  # output$individual_trials_plot <- shiny::renderUI({
  #   shiny::validate(
  #     shiny::need(
  #       length(local_reactives$update_outputs) &&
  #         !isFALSE(local_reactives$update_outputs),
  #       message = "Please run the module first"
  #     )
  #   )

  #   # retrieve the `power_outputs` from `local_data`
  #   power_outputs <- local_data$power_outputs
  #   power_outputs_list <- local_data$power_outputs_list
  #
  #   # For debug purposes, run
  #   # pipeline <- raveio::pipeline("fooof_module", paths = "/Users/dipterix/Dropbox (Personal)/projects/rave-pipeline-ese2025/modules/")
  #   # power_outputs <- pipeline$read("power_outputs")
  #
  #   # individual_trials = False
  #   # shared.plot_trials(power_outputs, individual_trials=individual_trials)
  #
  #   shared <- pipeline$python_module(type = "shared")
  #   plot <- shared$plot_trials(power_outputs_list, individual_trials = TRUE)
  #
  #   return(shiny::HTML(rpymat::py_to_r(plot$to_html())))
  # })

  # Output for reports
  output$fooof_print_results <- shiny::renderPrint({
    shiny::validate(
      shiny::need(
        length(local_reactives$update_outputs) &&
          !isFALSE(local_reactives$update_outputs),
        message = "Please run the module first"
      )
    )

    # retrieve the `power_outputs` from `local_data`
    # fitted_fooof <- local_data$fitted_fooof
    #
    # model <- fitted_fooof$model
    # frequencies <- fitted_fooof$frequencies
    # power <- fitted_fooof$power
    # freq_range <- fitted_fooof$freq_range
    power_outputs_list <- local_data$power_outputs_list
    conditions_analyzed_1 <- local_data$conditions_analyzed_1

    pipeline_settings <- pipeline$get_settings()

    freq_range <- pipeline_settings$freq_range
    max_n_peaks <- pipeline_settings$max_n_peaks
    aperiodic_mode <- pipeline_settings$aperiodic_mode
    plt_log <- !isFALSE(input$fooof_bool)

    shared <- pipeline$python_module(type = "shared")
    report <- reticulate::py_capture_output({
      shared$new_fit_fooof(power_outputs_list, conditions_analyzed_1, freq_range = freq_range, max_n_peaks=max_n_peaks, aperiodic_mode=aperiodic_mode, plt_log=plt_log)
    })
    cat(report)
  })

  # output$fooof_plot_results <- shiny::renderImage({
  #   shiny::validate(
  #     shiny::need(
  #       length(local_reactives$update_outputs) &&
  #         !isFALSE(local_reactives$update_outputs),
  #       message = "Please run the module first"
  #     )
  #   )
  #
  #   # retrieve the `power_outputs` from `local_data`
  #   fitted_fooof <- local_data$fitted_fooof
  #   plt_log <- !isFALSE(input$fooof_bool)
  #   plt <- rpymat::import("matplotlib.pyplot", as = "plt")
  #
  #   # For debug purposes, run
  #   # pipeline <- raveio::pipeline("fooof_module", paths = file.path(rstudioapi::getActiveProject(), "modules/"))
  #   # fitted_fooof <- pipeline$read("fitted_fooof")
  #
  #   model <- fitted_fooof$model
  #   model$plot(plt_log = plt_log)
  #   plt$title(sprintf("Condition Analyzed: %s", "my condition"))
  #
  #   # Create a temporary file to save the plot
  #   outfile <- normalizePath(tempfile(fileext = ".png"), winslash = "/", mustWork = FALSE)
  #   plt$savefig(outfile)
  #   plt$close()
  #   list(src = outfile,
  #        contentType = "image/png",
  #        width = "80%",
  #        height = "100%",
  #        alt = "Placeholder for fooof fit plot")
  # }, deleteFile = TRUE)

  # output$aperiodic_tuning_part <- shiny::renderUI({
  #   shiny::validate(
  #     shiny::need(
  #       length(local_reactives$update_outputs) &&
  #         !isFALSE(local_reactives$update_outputs),
  #       message = "Please run the module first"
  #     )
  #   )
  #
  #   # retrieve the `power_outputs` from `local_data`
  #   power_outputs <- local_data$power_outputs
  #   power_outputs_list <- local_data$power_outputs_list
  #
  #   pipeline_settings <- pipeline$get_settings()
  #
  #   freq_range_aperiodic_tuning <- pipeline_settings$freq_range_aperiodic_tuning
  #   max_n_peaks_aperiodic_tuning <- pipeline_settings$max_n_peaks_aperiodic_tuning
  #
  #   shared <- pipeline$python_module(type = "shared")
  #   plot <- shared$tune_aperiodic_mode(power_outputs_list, freq_range_aperiodic_tuning, max_n_peaks_aperiodic_tuning, show_errors=TRUE)
  #
  #   return(shiny::HTML(rpymat::py_to_r(plot$to_html())))
  # })


  # Output for aperiodic tuning
  output$aperiodic_tuning_part <- shiny::renderUI({
    shiny::validate(
      shiny::need(
        length(local_reactives$update_outputs) &&
          !isFALSE(local_reactives$update_outputs),
        message = "Please run the module first"
      )
    )

    power_outputs_list <- local_data$power_outputs_list
    conditions_analyzed_1 <- local_data$conditions_analyzed_1

    pipeline_settings <- pipeline$get_settings()

    freq_range_aperiodic_tuning <- pipeline_settings$freq_range_aperiodic_tuning
    max_n_peaks_aperiodic_tuning <- pipeline_settings$max_n_peaks_aperiodic_tuning
    freq_range <- pipeline_settings$freq_range
    max_n_peaks <- pipeline_settings$max_n_peaks

    lower_bound <- max(freq_range[1], freq_range_aperiodic_tuning[1])
    upper_bound <- min(freq_range[2], freq_range_aperiodic_tuning[2])

    # If no overlap, return error message UI
    if (lower_bound >= upper_bound) {
      return(
        htmltools::tags$p(
          style = "color: red; font-weight: bold; margin-left: 20px; margin-top: 15px;",
          "The frequency range for aperiodic tuning does not overlap with the frequency range of the data. Please check your frequency settings."
        )
      )
    }

    freq_range_msg <- paste("Frequency range:", lower_bound, "Hz to", upper_bound, "Hz")


    shared <- pipeline$python_module(type = "shared")
    result <- rpymat::py_to_r(shared$tune_aperiodic_mode(
      power_outputs_list,
      freq_range_aperiodic_tuning,
      max_n_peaks_aperiodic_tuning,
      conditions_analyzed_1,
      show_errors = TRUE
    ))

    if (!is.null(result$model_fail)) {
      return(
        htmltools::tags$p(
          style = "color: red; font-weight: bold; margin-left: 20px; margin-top: 15px;",
          result$model_fail
        )
      )
    }

    plotly_html <- htmltools::HTML(rpymat::py_to_r(result$plotly$to_html()))
    matplotlib_imgs <- result$matplotlib

    # Group into condition boxes (2 images per condition)
    condition_boxes <- lapply(seq(1, length(matplotlib_imgs), by = 2), function(i) {
      row_imgs <- matplotlib_imgs[i:min(i+1, length(matplotlib_imgs))]
      htmltools::tags$div(
        style = "border: 2px solid #ddd; border-radius: 8px; padding: 15px; margin-top: 20px; background-color: #fafafa;",
        htmltools::tags$div(
          style = "font-weight: bold; margin-bottom: 11px;",
          paste("Error Plots -", conditions_analyzed_1[(i + 1) %/% 2])
        ),
        htmltools::tags$div(
          style = "display: flex; justify-content: space-around; flex-wrap: wrap;",
          lapply(row_imgs, function(base64_img) {
            htmltools::tags$img(
              src = paste0("data:image/png;base64,", base64_img),
              style = "width: 48%; height: auto;"
            )
          })
        )
      )
    })

    htmltools::tagList(
      htmltools::tags$p(style = "font-weight: bold; font-size: 16px; margin-left: 20px; margin-top: 15px;", freq_range_msg),
      plotly_html,
      condition_boxes
    )
  })

  # Output for max_n_peaks tuning
  output$max_n_peaks_tuning_part <- shiny::renderUI({
    shiny::validate(
      shiny::need(
        length(local_reactives$update_outputs) &&
          !isFALSE(local_reactives$update_outputs),
        message = "Please run the module first"
      )
    )

    power_outputs_list <- local_data$power_outputs_list
    conditions_analyzed_1 <- local_data$conditions_analyzed_1

    pipeline_settings <- pipeline$get_settings()

    freq_range_tuning_max_n_peaks <- pipeline_settings$freq_range_tuning_max_n_peaks
    aperiodic_mode_tuning_max_n_peaks <- pipeline_settings$aperiodic_mode_tuning_max_n_peaks
    peaks_range_tuning_max_n_peaks <- pipeline_settings$peaks_range_tuning_max_n_peaks
    freq_range <- pipeline_settings$freq_range
    max_n_peaks <- pipeline_settings$max_n_peaks

    lower_bound <- max(freq_range[1], freq_range_tuning_max_n_peaks[1])
    upper_bound <- min(freq_range[2], freq_range_tuning_max_n_peaks[2])

    # If no overlap, return error message UI
    if (lower_bound >= upper_bound) {
      return(
        htmltools::tags$p(
          style = "color: red; font-weight: bold; margin-left: 20px; margin-top: 15px;",
          "The frequency range for max-n peaks tuning does not overlap with the frequency range of the data. Please check your frequency settings."
        )
      )
    }

    freq_range_msg <- paste("Frequency range:", lower_bound, "Hz to", upper_bound, "Hz")


    shared <- pipeline$python_module(type = "shared")
    result <- rpymat::py_to_r(shared$tune_max_n_peaks(
      power_outputs_list,
      freq_range_tuning_max_n_peaks,
      aperiodic_mode_tuning_max_n_peaks,
      peaks_range_tuning_max_n_peaks,
      conditions_analyzed_1,
      show_errors = TRUE
    ))

    if (!is.null(result$model_fail)) {
      return(
        htmltools::tags$p(
          style = "color: red; font-weight: bold; margin-left: 20px; margin-top: 15px;",
          result$model_fail
        )
      )
    }

    plotly_html <- htmltools::HTML(rpymat::py_to_r(result$plotly$to_html()))
    matplotlib_imgs <- result$matplotlib

    # Group into condition boxes (2 images per condition)
    condition_boxes <- lapply(seq(1, length(matplotlib_imgs), by = 2), function(i) {
      row_imgs <- matplotlib_imgs[i:min(i+1, length(matplotlib_imgs))]

      htmltools::tags$div(
        style = "border: 2px solid #ddd; border-radius: 8px; padding: 15px; margin-top: 20px; background-color: #fafafa;",
        htmltools::tags$div(
          style = "font-weight: bold; margin-bottom: 11px;",
          paste("Error Plots -", conditions_analyzed_1[(i + 1) %/% 2])
        ),
        htmltools::tags$div(
          style = "display: flex; justify-content: space-around; flex-wrap: wrap;",
          lapply(row_imgs, function(base64_img) {
            htmltools::tags$img(
              src = paste0("data:image/png;base64,", base64_img),
              style = "width: 48%; height: auto;"
            )
          })
        )
      )
    })

    htmltools::tagList(
      htmltools::tags$p(style = "font-weight: bold; font-size: 16px; margin-left: 20px; margin-top: 15px;;", freq_range_msg),
      plotly_html,
      condition_boxes
    )
  })

  # Output for peak threshold tuning
  output$peak_threshold_tuning_part <- shiny::renderUI({
    shiny::validate(
      shiny::need(
        length(local_reactives$update_outputs) &&
          !isFALSE(local_reactives$update_outputs),
        message = "Please run the module first"
      )
    )

    power_outputs_list <- local_data$power_outputs_list
    conditions_analyzed_1 <- local_data$conditions_analyzed_1

    pipeline_settings <- pipeline$get_settings()

    freq_range_tuning_peak_threshold <- pipeline_settings$freq_range_tuning_peak_threshold
    max_n_peaks_tuning_peak_threshold <- pipeline_settings$max_n_peaks_tuning_peak_threshold
    aperiodic_mode_tuning_peak_threshold <- pipeline_settings$aperiodic_mode_tuning_peak_threshold
    threshold_value_range_tuning_peak_threshold <- pipeline_settings$threshold_value_range_tuning_peak_threshold
    number_of_threshold_value_tuning_peak_threshold <- pipeline_settings$number_of_threshold_value_tuning_peak_threshold
    freq_range <- pipeline_settings$freq_range
    max_n_peaks <- pipeline_settings$max_n_peaks

    lower_bound <- max(freq_range[1], freq_range_tuning_peak_threshold[1])
    upper_bound <- min(freq_range[2], freq_range_tuning_peak_threshold[2])

    # If no overlap, return error message UI
    if (lower_bound >= upper_bound) {
      return(
        htmltools::tags$p(
          style = "color: red; font-weight: bold; margin-left: 20px; margin-top: 15px;",
          "The frequency range for max-n peaks tuning does not overlap with the frequency range of the data. Please check your frequency settings."
        )
      )
    }

    freq_range_msg <- paste("Frequency range:", lower_bound, "Hz to", upper_bound, "Hz")

    shared <- pipeline$python_module(type = "shared")
    result <- rpymat::py_to_r(shared$tune_peak_threshold(
      power_outputs_list,
      freq_range_tuning_peak_threshold,
      max_n_peaks_tuning_peak_threshold,
      aperiodic_mode_tuning_peak_threshold,
      threshold_value_range_tuning_peak_threshold[1],
      threshold_value_range_tuning_peak_threshold[2],
      number_of_threshold_value_tuning_peak_threshold,
      conditions_analyzed_1,
      show_errors = TRUE
    ))

    if (!is.null(result$model_fail)) {
      return(
        htmltools::tags$p(
          style = "color: red; font-weight: bold; margin-left: 20px; margin-top: 15px;",
          result$model_fail
        )
      )
    }

    plotly_html <- htmltools::HTML(rpymat::py_to_r(result$plotly$to_html()))
    matplotlib_imgs <- result$matplotlib

    # Group into condition boxes (2 images per condition)
    condition_boxes <- lapply(seq(1, length(matplotlib_imgs), by = 2), function(i) {
      row_imgs <- matplotlib_imgs[i:min(i+1, length(matplotlib_imgs))]
      htmltools::tags$div(
        style = "border: 2px solid #ddd; border-radius: 8px; padding: 15px; margin-top: 20px; background-color: #fafafa;",
        htmltools::tags$div(
          style = "font-weight: bold; margin-bottom: 11px;",
          paste("Error Plots -", conditions_analyzed_1[(i + 1) %/% 2])
        ),
        htmltools::tags$div(
          style = "display: flex; justify-content: space-around; flex-wrap: wrap;",
          lapply(row_imgs, function(base64_img) {
            htmltools::tags$img(
              src = paste0("data:image/png;base64,", base64_img),
              style = "width: 48%; height: auto;"
            )
          })
        )
      )
    })

    htmltools::tagList(
      htmltools::tags$p(style = "font-weight: bold; font-size: 16px; margin-left: 20px; margin-top: 15px;", freq_range_msg),
      plotly_html,
      condition_boxes
    )
  })


  # Output for fooof model fits
  output$fooof_plot_results_testing <- shiny::renderUI({
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
    conditions_analyzed_1 <- local_data$conditions_analyzed_1

    pipeline_settings <- pipeline$get_settings()

    freq_range <- pipeline_settings$freq_range
    max_n_peaks <- pipeline_settings$max_n_peaks
    aperiodic_mode <- pipeline_settings$aperiodic_mode
    plt_log <- !isFALSE(input$fooof_bool)

    shared <- pipeline$python_module(type = "shared")
    plot_list <- rpymat::py_to_r(shared$plot_fooof_fits(power_outputs_list, freq_range, max_n_peaks, aperiodic_mode, conditions_analyzed_1, plt_log = plt_log))

    # Convert each Python plot to HTML and combine
    html_output <- lapply(plot_list, function(p) {
      html_string <- rpymat::py_to_r(p$to_html())
      htmltools::HTML(html_string)
    })

    return(htmltools::tagList(html_output))
  })

}


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
        freq_range = c(1, input$fooof_freq_range)
      )

      # Step 2: save user inputs to settings.yaml
      pipeline$set_settings(.list = pipeline_settings)

      # Step 3: run analysis to get the results
      power_outputs <- pipeline$run(
        names = 'power_outputs',

        scheduler = "none",
        type = "vanilla"
      )

      # Step 4: Save the results and sell outputs to update
      local_data$power_outputs <- power_outputs
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

    # For debug purposes, run
    # pipeline <- raveio::pipeline("fooof_module", paths = "/Users/dipterix/Dropbox (Personal)/projects/rave-pipeline-ese2025/modules/")
    # power_outputs <- pipeline$read("power_outputs")

    # individual_trials = False
    # shared.plot_trials(power_outputs, individual_trials=individual_trials)

    shared <- pipeline$python_module(type = "shared")
    plot <- shared$plot_trials(power_outputs, individual_trials = FALSE)

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

    # For debug purposes, run
    # pipeline <- raveio::pipeline("fooof_module", paths = "/Users/dipterix/Dropbox (Personal)/projects/rave-pipeline-ese2025/modules/")
    # power_outputs <- pipeline$read("power_outputs")

    # individual_trials = False
    # shared.plot_trials(power_outputs, individual_trials=individual_trials)

    shared <- pipeline$python_module(type = "shared")
    plot <- shared$plot_trials(power_outputs, individual_trials = TRUE)

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
    power_outputs <- local_data$power_outputs

    # For debug purposes, run
    # pipeline <- raveio::pipeline("fooof_module", paths = "/Users/dipterix/Dropbox (Personal)/projects/rave-pipeline-ese2025/modules/")
    # power_outputs <- pipeline$read("power_outputs")

    shared <- pipeline$python_module(type = "shared")
    report <- reticulate::py_capture_output({
      shared$fit_fooof(
        power_outputs['filtered_frequency'],
        power_outputs['Average Power']
      )
    })
    cat(report)
  })

}


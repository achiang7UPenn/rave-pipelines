

module_html <- function(){

  shiny::fluidPage(
    shiny::fluidRow(

      shiny::column(
        width = 3L,
        shiny::div(
          # class = "row fancy-scroll-y stretch-inner-height",
          class = "row screen-height overflow-y-scroll",
          shiny::column(
            width = 12L,

            electrode_selector$ui_func(),

            comp_condition_groups$ui_func(),

            # baseline_choices$ui_func(),

            # defines a panel
            ravedash::input_card(
              title = "Configure Analysis",

              ravedash::flex_group_box(
                title = "Power Spectrum Settings",

                shidashi::flex_item(
                  shiny::checkboxInput(
                    inputId = ns("fooof_ind_trials"),
                    label = "Plot individual trials",
                    value = FALSE
                  )
                ),

                shidashi::flex_break(),

                shidashi::flex_item(
                  shiny::sliderInput(
                    inputId = ns("fooof_standard_deviation"),
                    label = "Standard Deviation",
                    min = 0,
                    max = 3,
                    value = 1,
                    step = 1,
                  )
                )

              ),

              # a collection of inputs
              ravedash::flex_group_box(
                title = "FOOOF Parameters",

                # first item
                shidashi::flex_item(
                  shiny::sliderInput(
                    inputId = ns("fooof_winlen"),
                    label = "Window length",
                    min = 0,
                    max = 4,
                    value = 1,
                    step = 0.1,
                    post = " s"
                  )
                ),

                # break the elements into two rows
                shidashi::flex_break(),

                # second item is the frequency range
                shidashi::flex_item(
                  shiny::sliderInput(
                    inputId = ns("fooof_freq_range"),
                    label = "Frequency range",
                    min = 1,
                    max = 300,
                    value = c(1, 200),
                    step = 1,
                    post = " Hz"
                  )
                ),

                shidashi::flex_break(),

                shidashi::flex_item(
                  shiny::sliderInput(
                    inputId = ns("fooof_max_n_peaks"),
                    label = "Maximum n peaks",
                    min = 1,
                    max = 15,
                    value = 5,
                    step = 1,
                  )
                ),

                shidashi::flex_break(),

                shidashi::flex_item(
                  shiny::selectInput(
                    inputId = ns("fooof_aperiodic_mode"),
                    label = "Aperiodic mode",
                    choices = c("fixed", "knee"),
                    selected = "knee"
                  )
                ),

                shidashi::flex_break(),

                shidashi::flex_item(
                  shiny::checkboxInput(
                    inputId = ns("fooof_bool"),
                    label = "Model fit in logarithmic scale",
                    value = FALSE
                  )
                )

              ),

              ravedash::flex_group_box(
                title = "Max n Peaks Tuning",

                shidashi::flex_item(
                  shiny::sliderInput(
                    inputId = ns("freq_range_tuning_max_n_peaks"),
                    label = "Frequency range",
                    min = 1,
                    max = 300,
                    value = c(1, 200),
                    step = 1,
                    post = "Hz"
                  )
                ),

                shidashi::flex_break(),

                shidashi::flex_item(
                  shiny::selectInput(
                    inputId = ns("aperiodic_mode_tuning_max_n_peaks"),
                    label = "Aperiodic mode",
                    choices = c("fixed", "knee"),
                    selected = "knee"
                  )
                ),

                shidashi::flex_break(),

                shidashi::flex_item(
                  shiny::sliderInput(
                    inputId = ns("peaks_range_tuning_max_n_peaks"),
                    label = "Peaks range",
                    min = 1,
                    max = 50,
                    value = c(1, 15),
                    step = 1,
                  )
                )
              ),

              ravedash::flex_group_box(
                title = "Aperiodic Mode Tuning",

                shidashi::flex_item(
                  shiny::sliderInput(
                    inputId = ns("freq_range_tuning_aperiodic_mode"),
                    label = "Frequency range",
                    min = 1,
                    max = 300,
                    value = c(1, 200),
                    step = 1,
                    post = "Hz"
                  )
                ),

                shidashi::flex_break(),

                shidashi::flex_item(
                  shiny::sliderInput(
                    inputId = ns("max_n_peaks_tuning_aperiodic_mode"),
                    label = "Maximum n peaks",
                    min = 1,
                    max = 15,
                    value = 5,
                    step = 1,
                  )
                )

              ),

              ravedash::flex_group_box(
                title = "Peak Threshold Tuning",

                shidashi::flex_item(
                  shiny::sliderInput(
                    inputId = ns("freq_range_tuning_peak_threshold"),
                    label = "Frequency range",
                    min = 1,
                    max = 300,
                    value = c(1, 200),
                    step = 1,
                    post = "Hz"
                  )
                ),

                shidashi::flex_break(),

                shidashi::flex_item(
                  shiny::sliderInput(
                    inputId = ns("max_n_peaks_tuning_peak_threshold"),
                    label = "Maximum n peaks",
                    min = 1,
                    max = 15,
                    value = 5,
                    step = 1,
                  )
                ),

                shidashi::flex_break(),

                shidashi::flex_item(
                  shiny::selectInput(
                    inputId = ns("aperiodic_mode_tuning_peak_threshold"),
                    label = "Aperiodic mode",
                    choices = c("fixed", "knee"),
                    selected = "knee"
                  )
                ),

                shidashi::flex_break(),

                shidashi::flex_item(
                  shiny::sliderInput(
                    inputId = ns("threshold_value_range_tuning_peak_threshold"),
                    label = "Peak threshold value",
                    min = 0.1,
                    max = 10,
                    value = c(1,10),
                    step = 0.1,
                  )
                ),

                shidashi::flex_break(),

                shidashi::flex_item(
                  shiny::sliderInput(
                    inputId = ns("number_of_threshold_value_tuning_peak_threshold"),
                    label = "Number of peak thresholds to be generated",
                    min = 1,
                    max = 100,
                    value = 100,
                    step = 1
                  )
                )

              )
            )

          )
        )
      ),

      shiny::column(
        width = 9L,
        shiny::div(
          class = "row screen-height overflow-y-scroll output-wrapper",
          shiny::column(
            width = 12L,


            ravedash::output_card(
              'Power Spectrum',
              class_body = "no-padding fill-width height-450 min-height-450 resize-vertical",
              shiny::div(
                class = 'position-relative fill',
                shiny::uiOutput(ns("collapse_over_trial"), width = '100%', height = "100%")
                # plotly::plotlyOutput(ns("collapse_over_trial"), width = '100%', height = "100%")
              )
            ),

            # ravedash::output_card(
            #   'Individual Trials',
            #   class_body = "no-padding fill-width height-450 min-height-450 resize-vertical",
            #   shiny::div(
            #     class = 'position-relative fill',
            #     shiny::uiOutput(ns("individual_trials_plot"), width = '100%', height = "100%")
            #     # plotly::plotlyOutput(ns("collapse_over_trial"), width = '100%', height = "100%")
            #   )
            # ),


            # ravedash::output_card(
            #   "FOOOF Spectral Model Plot",
            #   class_body = "no-padding fill-width height-450 min-height-450 resize-vertical",
            #   shiny::div(
            #     class = 'position-relative fill',
            #
            #     shiny::plotOutput(
            #       outputId = ns("fooof_plot_results")
            #     )
            #   )
            # ),

            ravedash::output_card(
              "FOOOF Fits Plot",
              class_body = "no-padding fill-width height-450 min-height-450 resize-vertical",
              shiny::div(
                class = 'position-relative fill',

                shiny::uiOutput(ns("fooof_plot_results_testing"), width="100%'", height="100%")
              )
            ),

            ravedash::output_card(
              "Fooof Spectral Model Fit",
              class_body = "no-padding fill-width height-650 min-height-450 resize-vertical",
              shiny::div(
                class = 'position-relative fill',

                shiny::verbatimTextOutput(
                  outputId = ns("fooof_print_results")
                )
              )
            ),

            ravedash::output_card(
              'Max n Peaks Tuning',
              class_body = "no-padding fill-width height-450 min-height-450 resize-vertical",
              shiny::div(
                class = 'position-relative fill',
                shiny::uiOutput(ns("max_n_peaks_tuning_part"), width = '100%', height = "100%")
                # plotly::plotlyOutput(ns("collapse_over_trial"), width = '100%', height = "100%")
              )
            ),

            ravedash::output_card(
              'Aperiodic Mode Tuning',
              class_body = "no-padding fill-width height-450 min-height-450 resize-vertical",
              shiny::div(
                class = 'position-relative fill',
                shiny::uiOutput(ns("aperiodic_tuning_part"), width = '100%', height = "100%")
                # plotly::plotlyOutput(ns("collapse_over_trial"), width = '100%', height = "100%")
              )
            ),

            ravedash::output_card(
              'Peak Threshold Tuning',
              class_body = "no-padding fill-width height-450 min-height-450 resize-vertical",
              shiny::div(
                class = 'position-relative fill',
                shiny::uiOutput(ns("peak_threshold_tuning_part"), width = '100%', height = "100%")
                # plotly::plotlyOutput(ns("collapse_over_trial"), width = '100%', height = "100%")
              )
            )


          )
        )
      )

    )
  )
}

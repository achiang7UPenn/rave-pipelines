

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

              # a collection of inputs
              ravedash::flex_group_box(
                title = "Fooof Parameters",

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
                    min = 2,
                    max = 300,
                    value = 200,
                    step = 0.1,
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

              # ravedash::flex_group_box(
              #   title = "Peak Tuning",
              #
              #   shidashi::flex_item(
              #     shiny::sliderInput(
              #       inputId = ns("freq_range_tuning_peak"),
              #       label = "Frequency range",
              #       min = 1,
              #       max = 300,
              #       value = 200,
              #       step = 0.1,
              #       post = " Hz"
              #     )
              #   ),
              #
              #   shidashi::flex_break(),
              #
              #   shidashi::flex_item(
              #     shiny::textInput(
              #       inputId = ns("aperiodic_mode_tuning_peak"),
              #       label = "Aperiodic mode"
              #     )
              #   )
              #
              # ),

              ravedash::flex_group_box(
                title = "Aperiodic Mode Tuning",

                shidashi::flex_item(
                  shiny::sliderInput(
                    inputId = ns("freq_range_tuning_aperiodic_mode"),
                    label = "Frequency range (Tune anything below the frequency range under Fooof Parameters)",
                    min = 2,
                    max = 300,
                    value = 200,
                    step = 0.1,
                    post = " Hz"
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
              'Average',
              class_body = "no-padding fill-width height-450 min-height-450 resize-vertical",
              shiny::div(
                class = 'position-relative fill',
                shiny::uiOutput(ns("collapse_over_trial"), width = '100%', height = "100%")
                # plotly::plotlyOutput(ns("collapse_over_trial"), width = '100%', height = "100%")
              )
            ),

            ravedash::output_card(
              'Individual Trials',
              class_body = "no-padding fill-width height-450 min-height-450 resize-vertical",
              shiny::div(
                class = 'position-relative fill',
                shiny::uiOutput(ns("individual_trials_plot"), width = '100%', height = "100%")
                # plotly::plotlyOutput(ns("collapse_over_trial"), width = '100%', height = "100%")
              )
            ),


            ravedash::output_card(
              "FOOOF Spectral Model Plot",
              class_body = "no-padding fill-width height-450 min-height-450 resize-vertical",
              shiny::div(
                class = 'position-relative fill',

                shiny::plotOutput(
                  outputId = ns("fooof_plot_results")
                )
              )
            ),
            ravedash::output_card(
              "FOOOF Spectral Model Fit",
              class_body = "no-padding fill-width height-650 min-height-450 resize-vertical",
              shiny::div(
                class = 'position-relative fill',

                shiny::verbatimTextOutput(
                  outputId = ns("fooof_print_results")
                )
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
            )


          )
        )
      )

    )
  )
}

import logging

import numpy as np
import pandas as pd
import panel as pn

logger = logging.getLogger(__name__)


class InputParametersGUI:
    def __init__(self, folder_path):
        read_progress = pn.indicators.Progress(name="Progress", value=100, max=100, width=300)
        extract_progress = pn.indicators.Progress(name="Progress", value=100, max=100, width=300)
        psth_progress = pn.indicators.Progress(name="Progress", value=100, max=100, width=300)

        template = pn.template.BootstrapTemplate(title="Input Parameters GUI")

        mark_down_1 = pn.pane.Markdown(
            """**Select folders for the analysis from the file selector below**""", width=600
        )

        files_1 = pn.widgets.FileSelector(folder_path, name="folderNames", width=950)

        explain_modality = pn.pane.Markdown(
            """
            **Data Modality:** Select the type of data acquisition system used for your recordings:
            - **tdt**: Tucker-Davis Technologies system
            - **csv**: Generic CSV format
            - **doric**: Doric Photometry system
            - **npm**: Neurophotometrics system
            """,
            width=600,
        )

        modality_selector = pn.widgets.Select(
            name="Data Modality", value="tdt", options=["tdt", "csv", "doric", "npm"], width=320
        )

        explain_time_artifacts = pn.pane.Markdown(
            """
                                                    - ***Number of cores :*** Number of cores used for analysis. Try to
                                                    keep it less than the number of cores in your machine.
                                                    - ***Combine Data? :*** Make this parameter ``` True ``` if user wants to combine
                                                    the data, especially when there is two different
                                                    data files for the same recording session.<br>
                                                    - ***Isosbestic Control Channel? :*** Make this parameter ``` False ``` if user
                                                    does not want to use isosbestic control channel in the analysis.<br>
                                                    - ***Eliminate first few seconds :*** It is the parameter to cut out first x seconds
                                                    from the data. Default is 1 seconds.<br>
                                                    - ***Window for Moving Average filter :*** The filtering of signals
                                                    is done using moving average filter. Default window used for moving
                                                    average filter is 100 datapoints. Change it based on the requirement.<br>
                                                    - ***Moving Window (transients detection) :*** Transients in the z-score
                                                    and/or \u0394F/F are detected using this moving window.
                                                    Default is 15 seconds. Change it based on the requirement.<br>
                                                    - ***High Amplitude filtering threshold (HAFT) (transients detection) :*** High amplitude
                                                    events greater than x times the MAD above the median are filtered out. Here, x is
                                                    high amplitude filtering threshold. Default is 2.
                                                    - ***Transients detection threshold (TD Thresh):*** Peaks with local maxima greater than x times
                                                    the MAD above the median of the trace (after filtering high amplitude events) are detected
                                                    as transients. Here, x is transients detection threshold. Default is 3.
                                                    - ***Number of channels (Neurophotometrics only) :*** Number of
                                                    channels used while recording, when data files has no column names mentioning "Flags"
                                                    or "LedState".
                                                    - ***removeArtifacts? :*** Make this parameter ``` True``` if there are
                                                    artifacts and user wants to remove the artifacts.
                                                    - ***removeArtifacts method :*** Selecting ```concatenate``` will remove bad
                                                    chunks and concatenate the selected good chunks together.
                                                    Selecting ```replace with NaN``` will replace bad chunks with NaN
                                                    values.
                                                    """,
            width=350,
        )

        timeForLightsTurnOn = pn.widgets.LiteralInput(
            name="Eliminate first few seconds (int)", value=1, type=int, width=320
        )

        isosbestic_control = pn.widgets.Select(
            name="Isosbestic Control Channel? (bool)", value=True, options=[True, False], width=320
        )

        numberOfCores = pn.widgets.LiteralInput(name="# of cores (int)", value=2, type=int, width=150)

        combine_data = pn.widgets.Select(name="Combine Data? (bool)", value=False, options=[True, False], width=150)

        computePsth = pn.widgets.Select(
            name="z_score and/or \u0394F/F? (psth)", options=["z_score", "dff", "Both"], width=320
        )

        transients = pn.widgets.Select(
            name="z_score and/or \u0394F/F? (transients)", options=["z_score", "dff", "Both"], width=320
        )

        plot_zScore_dff = pn.widgets.Select(
            name="z-score plot and/or \u0394F/F plot?",
            options=["z_score", "dff", "Both", "None"],
            value="None",
            width=320,
        )

        moving_wd = pn.widgets.LiteralInput(
            name="Moving Window for transients detection (s) (int)", value=15, type=int, width=320
        )

        highAmpFilt = pn.widgets.LiteralInput(name="HAFT (int)", value=2, type=int, width=150)

        transientsThresh = pn.widgets.LiteralInput(name="TD Thresh (int)", value=3, type=int, width=150)

        moving_avg_filter = pn.widgets.LiteralInput(
            name="Window for Moving Average filter (int)", value=100, type=int, width=320
        )

        removeArtifacts = pn.widgets.Select(
            name="removeArtifacts? (bool)", value=False, options=[True, False], width=150
        )

        artifactsRemovalMethod = pn.widgets.Select(
            name="removeArtifacts method", value="concatenate", options=["concatenate", "replace with NaN"], width=150
        )

        no_channels_np = pn.widgets.LiteralInput(
            name="Number of channels (Neurophotometrics only)", value=2, type=int, width=320
        )

        z_score_computation = pn.widgets.Select(
            name="z-score computation Method",
            options=["standard z-score", "baseline z-score", "modified z-score"],
            value="standard z-score",
            width=200,
        )

        baseline_wd_strt = pn.widgets.LiteralInput(
            name="Baseline Window Start Time (s) (int)", value=0, type=int, width=200
        )
        baseline_wd_end = pn.widgets.LiteralInput(
            name="Baseline Window End Time (s) (int)", value=0, type=int, width=200
        )

        explain_z_score = pn.pane.Markdown(
            """
                                        ***Note :***<br>
                                        - Details about z-score computation methods are explained in Github wiki.<br>
                                        - The details will make user understand what computation method to use for
                                        their data.<br>
                                        - Baseline Window Parameters should be kept 0 unless you are using baseline<br>
                                        z-score computation method. The parameters are in seconds.
                                        """,
            width=580,
        )

        explain_nsec = pn.pane.Markdown(
            """
                                        - ***Time Interval :*** To omit bursts of event timestamps, user defined time interval
                                        is set so that if the time difference between two timestamps is less than this defined time
                                        interval, it will be deleted for the calculation of PSTH.
                                        - ***Compute Cross-correlation :*** Make this parameter ```True```, when user wants
                                        to compute cross-correlation between PSTHs of two different signals or signals
                                        recorded from different brain regions.
                                        """,
            width=580,
        )

        nSecPrev = pn.widgets.LiteralInput(name="Seconds before 0 (int)", value=-10, type=int, width=120)

        nSecPost = pn.widgets.LiteralInput(name="Seconds after 0 (int)", value=20, type=int, width=120)

        computeCorr = pn.widgets.Select(
            name="Compute Cross-correlation (bool)", options=[True, False], value=False, width=200
        )

        timeInterval = pn.widgets.LiteralInput(name="Time Interval (s)", value=2, type=int, width=120)

        use_time_or_trials = pn.widgets.Select(
            name="Bin PSTH trials (str)", options=["Time (min)", "# of trials"], value="Time (min)", width=120
        )

        bin_psth_trials = pn.widgets.LiteralInput(
            name="Time(min) / # of trials \n for binning? (int)", value=0, type=int, width=200
        )

        explain_baseline = pn.pane.Markdown(
            """
                                            ***Note :***<br>
                                            - If user does not want to do baseline correction,
                                            put both parameters 0.<br>
                                            - If the first event timestamp is less than the length of baseline
                                            window, it will be rejected in the PSTH computation step.<br>
                                            - Baseline parameters must be within the PSTH parameters
                                            set in the PSTH parameters section.
                                            """,
            width=580,
        )

        baselineCorrectionStart = pn.widgets.LiteralInput(
            name="Baseline Correction Start time(int)", value=-5, type=int, width=200
        )

        baselineCorrectionEnd = pn.widgets.LiteralInput(
            name="Baseline Correction End time(int)", value=0, type=int, width=200
        )

        zscore_param_wd = pn.WidgetBox(
            "### Z-score Parameters",
            explain_z_score,
            z_score_computation,
            pn.Row(baseline_wd_strt, baseline_wd_end),
            width=600,
        )

        psth_param_wd = pn.WidgetBox(
            "### PSTH Parameters",
            explain_nsec,
            pn.Row(nSecPrev, nSecPost, computeCorr),
            pn.Row(timeInterval, use_time_or_trials, bin_psth_trials),
            width=600,
        )

        baseline_param_wd = pn.WidgetBox(
            "### Baseline Parameters",
            explain_baseline,
            pn.Row(baselineCorrectionStart, baselineCorrectionEnd),
            width=600,
        )
        peak_explain = pn.pane.Markdown(
            """
                                        ***Note :***<br>
                                        - Peak and area are computed between the window set below.<br>
                                        - Peak and AUC parameters must be within the PSTH parameters set in the PSTH parameters section.<br>
                                        - Please make sure when user changes the parameters in the table below, click on any other cell after
                                        changing a value in a particular cell.
                                        """,
            width=580,
        )

        start_end_point_df = pd.DataFrame(
            {
                "Peak Start time": [-5, 0, 5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                "Peak End time": [0, 3, 10, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )

        df_widget = pn.widgets.Tabulator(start_end_point_df, name="DataFrame", show_index=False, widths=280)

        peak_param_wd = pn.WidgetBox("### Peak and AUC Parameters", peak_explain, df_widget, width=600)

        mark_down_2 = pn.pane.Markdown(
            """**Select folders for the average analysis from the file selector below**""", width=600
        )

        files_2 = pn.widgets.FileSelector(folder_path, name="folderNamesForAvg", width=950)

        averageForGroup = pn.widgets.Select(name="Average Group? (bool)", value=False, options=[True, False], width=435)

        visualizeAverageResults = pn.widgets.Select(
            name="Visualize Average Results? (bool)", value=False, options=[True, False], width=435
        )

        visualize_zscore_or_dff = pn.widgets.Select(
            name="z-score or \u0394F/F? (for visualization)", options=["z_score", "dff"], width=435
        )

        individual_analysis_wd_2 = pn.Column(
            explain_time_artifacts,
            pn.Row(numberOfCores, combine_data),
            isosbestic_control,
            timeForLightsTurnOn,
            moving_avg_filter,
            computePsth,
            transients,
            plot_zScore_dff,
            moving_wd,
            pn.Row(highAmpFilt, transientsThresh),
            no_channels_np,
            pn.Row(removeArtifacts, artifactsRemovalMethod),
        )

        group_analysis_wd_1 = pn.Column(mark_down_2, files_2, averageForGroup, width=800)

        visualization_wd = pn.Row(visualize_zscore_or_dff, pn.Spacer(width=60), visualizeAverageResults)

    def getInputParameters(self):
        abspath = getAbsPath()
        inputParameters = {
            "abspath": abspath[0],
            "folderNames": files_1.value,
            "modality": modality_selector.value,
            "numberOfCores": numberOfCores.value,
            "combine_data": combine_data.value,
            "isosbestic_control": isosbestic_control.value,
            "timeForLightsTurnOn": timeForLightsTurnOn.value,
            "filter_window": moving_avg_filter.value,
            "removeArtifacts": removeArtifacts.value,
            "artifactsRemovalMethod": artifactsRemovalMethod.value,
            "noChannels": no_channels_np.value,
            "zscore_method": z_score_computation.value,
            "baselineWindowStart": baseline_wd_strt.value,
            "baselineWindowEnd": baseline_wd_end.value,
            "nSecPrev": nSecPrev.value,
            "nSecPost": nSecPost.value,
            "computeCorr": computeCorr.value,
            "timeInterval": timeInterval.value,
            "bin_psth_trials": bin_psth_trials.value,
            "use_time_or_trials": use_time_or_trials.value,
            "baselineCorrectionStart": baselineCorrectionStart.value,
            "baselineCorrectionEnd": baselineCorrectionEnd.value,
            "peak_startPoint": list(df_widget.value["Peak Start time"]),  # startPoint.value,
            "peak_endPoint": list(df_widget.value["Peak End time"]),  # endPoint.value,
            "selectForComputePsth": computePsth.value,
            "selectForTransientsComputation": transients.value,
            "moving_window": moving_wd.value,
            "highAmpFilt": highAmpFilt.value,
            "transientsThresh": transientsThresh.value,
            "plot_zScore_dff": plot_zScore_dff.value,
            "visualize_zscore_or_dff": visualize_zscore_or_dff.value,
            "folderNamesForAvg": files_2.value,
            "averageForGroup": averageForGroup.value,
            "visualizeAverageResults": visualizeAverageResults.value,
        }
        return inputParameters

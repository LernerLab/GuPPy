import logging
import os

import numpy as np
import pandas as pd
import panel as pn

logger = logging.getLogger(__name__)


def _default_root_path():
    # Respect GUPPY_BASE_DIR env var for headless/test mode; otherwise use the home directory.
    base_dir_env = os.environ.get("GUPPY_BASE_DIR")
    if base_dir_env and os.path.isdir(base_dir_env):
        return base_dir_env
    return os.path.expanduser("~")


def checkSameLocation(arr, abspath):
    # abspath = []
    for i in range(len(arr)):
        abspath.append(os.path.dirname(arr[i]))
    abspath = np.asarray(abspath)
    abspath = np.unique(abspath)
    if len(abspath) > 1:
        logger.error("All the folders selected should be at the same location")
        raise Exception("All the folders selected should be at the same location")

    return abspath


def getAbsPath(files_1, files_2):
    arr_1, arr_2 = files_1.value, files_2.value
    if len(arr_1) == 0 and len(arr_2) == 0:
        logger.error("No folder is selected for analysis")
        raise Exception("No folder is selected for analysis")

    abspath = []
    if len(arr_1) > 0:
        abspath = checkSameLocation(arr_1, abspath)
    else:
        abspath = checkSameLocation(arr_2, abspath)

    abspath = np.unique(abspath)
    if len(abspath) > 1:
        logger.error("All the folders selected should be at the same location")
        raise Exception("All the folders selected should be at the same location")
    return abspath


class ParameterForm:
    def __init__(self, *, template):
        self.template = template
        self.folder_path = _default_root_path()
        self.styles = dict(background="WhiteSmoke")
        self.setup_individual_parameters()
        self.setup_group_parameters()
        self.setup_visualization_parameters()
        self.add_to_template()

    def setup_individual_parameters(self):
        # Individual analysis components
        self.mark_down_1 = pn.pane.Markdown(
            """**Select folders for the analysis from the file selector below**""", width=600
        )

        self.files_1 = pn.widgets.FileSelector(self.folder_path, root_directory="/", name="folderNames", width=950)

        self.explain_modality = pn.pane.Markdown(
            """
            **Data Modality:** Select the type of data acquisition system used for your recordings:
            - **auto**: Automatically detect from folder contents (recommended)
            - **tdt**: Tucker-Davis Technologies system
            - **csv**: Generic CSV format
            - **doric**: Doric Photometry system
            - **npm**: Neurophotometrics system
            """,
            width=600,
        )

        self.modality_selector = pn.widgets.Select(
            name="Data Modality", value="auto", options=["auto", "tdt", "csv", "doric", "npm"], width=320
        )

        self.explain_time_artifacts = pn.pane.Markdown(
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

        self.timeForLightsTurnOn = pn.widgets.LiteralInput(
            name="Eliminate first few seconds (int)", value=1, type=int, width=320
        )

        self.isosbestic_control = pn.widgets.Select(
            name="Isosbestic Control Channel? (bool)", value=True, options=[True, False], width=320
        )

        self.numberOfCores = pn.widgets.LiteralInput(name="# of cores (int)", value=2, type=int, width=150)

        self.combine_data = pn.widgets.Select(
            name="Combine Data? (bool)", value=False, options=[True, False], width=150
        )

        self.computePsth = pn.widgets.Select(
            name="z_score and/or \u0394F/F? (psth)", options=["z_score", "dff", "Both"], width=320
        )

        self.transients = pn.widgets.Select(
            name="z_score and/or \u0394F/F? (transients)", options=["z_score", "dff", "Both"], width=320
        )

        self.plot_zScore_dff = pn.widgets.Select(
            name="z-score plot and/or \u0394F/F plot?",
            options=["z_score", "dff", "Both", "None"],
            value="None",
            width=320,
        )

        self.moving_wd = pn.widgets.LiteralInput(
            name="Moving Window for transients detection (s) (int)", value=15, type=int, width=320
        )

        self.highAmpFilt = pn.widgets.LiteralInput(name="HAFT (int)", value=2, type=int, width=150)

        self.transientsThresh = pn.widgets.LiteralInput(name="TD Thresh (int)", value=3, type=int, width=150)

        self.moving_avg_filter = pn.widgets.LiteralInput(
            name="Window for Moving Average filter (int)", value=100, type=int, width=320
        )

        self.removeArtifacts = pn.widgets.Select(
            name="removeArtifacts? (bool)", value=False, options=[True, False], width=150
        )

        self.artifactsRemovalMethod = pn.widgets.Select(
            name="removeArtifacts method", value="concatenate", options=["concatenate", "replace with NaN"], width=150
        )

        self.no_channels_np = pn.widgets.LiteralInput(
            name="Number of channels (Neurophotometrics only)", value=2, type=int, width=320
        )

        self.z_score_computation = pn.widgets.Select(
            name="z-score computation Method",
            options=["standard z-score", "baseline z-score", "modified z-score"],
            value="standard z-score",
            width=200,
        )

        self.baseline_wd_strt = pn.widgets.LiteralInput(
            name="Baseline Window Start Time (s) (int)", value=0, type=int, width=200
        )
        self.baseline_wd_end = pn.widgets.LiteralInput(
            name="Baseline Window End Time (s) (int)", value=0, type=int, width=200
        )

        self.explain_z_score = pn.pane.Markdown(
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

        self.explain_nsec = pn.pane.Markdown(
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

        self.nSecPrev = pn.widgets.LiteralInput(name="Seconds before 0 (int)", value=-10, type=int, width=120)

        self.nSecPost = pn.widgets.LiteralInput(name="Seconds after 0 (int)", value=20, type=int, width=120)

        self.computeCorr = pn.widgets.Select(
            name="Compute Cross-correlation (bool)", options=[True, False], value=False, width=200
        )

        self.timeInterval = pn.widgets.LiteralInput(name="Time Interval (s)", value=2, type=int, width=120)

        self.use_time_or_trials = pn.widgets.Select(
            name="Bin PSTH trials (str)", options=["Time (min)", "# of trials"], value="Time (min)", width=120
        )

        self.bin_psth_trials = pn.widgets.LiteralInput(
            name="Time(min) / # of trials \n for binning? (int)", value=0, type=int, width=200
        )

        self.explain_baseline = pn.pane.Markdown(
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

        self.baselineCorrectionStart = pn.widgets.LiteralInput(
            name="Baseline Correction Start time(int)", value=-5, type=int, width=200
        )

        self.baselineCorrectionEnd = pn.widgets.LiteralInput(
            name="Baseline Correction End time(int)", value=0, type=int, width=200
        )

        self.zscore_param_wd = pn.WidgetBox(
            "### Z-score Parameters",
            self.explain_z_score,
            self.z_score_computation,
            pn.Row(self.baseline_wd_strt, self.baseline_wd_end),
            width=600,
        )

        self.psth_param_wd = pn.WidgetBox(
            "### PSTH Parameters",
            self.explain_nsec,
            pn.Row(self.nSecPrev, self.nSecPost, self.computeCorr),
            pn.Row(self.timeInterval, self.use_time_or_trials, self.bin_psth_trials),
            width=600,
        )

        self.baseline_param_wd = pn.WidgetBox(
            "### Baseline Parameters",
            self.explain_baseline,
            pn.Row(self.baselineCorrectionStart, self.baselineCorrectionEnd),
            width=600,
        )
        self.peak_explain = pn.pane.Markdown(
            """
                        ***Note :***<br>
                        - Peak and area are computed between the window set below.<br>
                        - Peak and AUC parameters must be within the PSTH parameters set in the PSTH parameters section.<br>
                        - Please make sure when user changes the parameters in the table below, click on any other cell after
                        changing a value in a particular cell.
                        """,
            width=580,
        )

        self.start_end_point_df = pd.DataFrame(
            {
                "Peak Start time": [-5, 0, 5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                "Peak End time": [0, 3, 10, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )

        self.df_widget = pn.widgets.Tabulator(self.start_end_point_df, name="DataFrame", show_index=False, widths=280)

        self.peak_param_wd = pn.WidgetBox("### Peak and AUC Parameters", self.peak_explain, self.df_widget, width=600)

        self.individual_analysis_wd_2 = pn.Column(
            self.explain_time_artifacts,
            pn.Row(self.numberOfCores, self.combine_data),
            self.isosbestic_control,
            self.timeForLightsTurnOn,
            self.moving_avg_filter,
            self.computePsth,
            self.transients,
            self.plot_zScore_dff,
            self.moving_wd,
            pn.Row(self.highAmpFilt, self.transientsThresh),
            self.no_channels_np,
            pn.Row(self.removeArtifacts, self.artifactsRemovalMethod),
        )

        self.psth_baseline_param = pn.Column(
            self.zscore_param_wd, self.psth_param_wd, self.baseline_param_wd, self.peak_param_wd
        )

        self.widget = pn.Column(
            self.mark_down_1,
            self.files_1,
            self.explain_modality,
            self.modality_selector,
            pn.Row(self.individual_analysis_wd_2, self.psth_baseline_param),
        )
        self.individual = pn.Card(self.widget, title="Individual Analysis", styles=self.styles, width=1000)

    def setup_group_parameters(self):
        self.mark_down_2 = pn.pane.Markdown(
            """**Select folders for the average analysis from the file selector below**""", width=600
        )

        self.files_2 = pn.widgets.FileSelector(
            self.folder_path, root_directory="/", name="folderNamesForAvg", width=950
        )

        self.averageForGroup = pn.widgets.Select(
            name="Average Group? (bool)", value=False, options=[True, False], width=435
        )

        self.group_analysis_wd_1 = pn.Column(self.mark_down_2, self.files_2, self.averageForGroup, width=800)
        self.group = pn.Card(self.group_analysis_wd_1, title="Group Analysis", styles=self.styles, width=1000)

    def setup_visualization_parameters(self):
        self.visualizeAverageResults = pn.widgets.Select(
            name="Visualize Average Results? (bool)", value=False, options=[True, False], width=435
        )

        self.visualize_zscore_or_dff = pn.widgets.Select(
            name="z-score or \u0394F/F? (for visualization)", options=["z_score", "dff"], width=435
        )

        self.visualization_wd = pn.Row(self.visualize_zscore_or_dff, pn.Spacer(width=60), self.visualizeAverageResults)
        self.visualize = pn.Card(
            self.visualization_wd, title="Visualization Parameters", styles=self.styles, width=1000
        )

    def add_to_template(self):
        self.template.main.append(self.individual)
        self.template.main.append(self.group)
        self.template.main.append(self.visualize)

    def getInputParameters(self):
        abspath = getAbsPath(self.files_1, self.files_2)
        inputParameters = {
            "abspath": abspath[0],
            "folderNames": self.files_1.value,
            "modality": self.modality_selector.value,
            "numberOfCores": self.numberOfCores.value,
            "combine_data": self.combine_data.value,
            "isosbestic_control": self.isosbestic_control.value,
            "timeForLightsTurnOn": self.timeForLightsTurnOn.value,
            "filter_window": self.moving_avg_filter.value,
            "removeArtifacts": self.removeArtifacts.value,
            "artifactsRemovalMethod": self.artifactsRemovalMethod.value,
            "noChannels": self.no_channels_np.value,
            "zscore_method": self.z_score_computation.value,
            "baselineWindowStart": self.baseline_wd_strt.value,
            "baselineWindowEnd": self.baseline_wd_end.value,
            "nSecPrev": self.nSecPrev.value,
            "nSecPost": self.nSecPost.value,
            "computeCorr": self.computeCorr.value,
            "timeInterval": self.timeInterval.value,
            "bin_psth_trials": self.bin_psth_trials.value,
            "use_time_or_trials": self.use_time_or_trials.value,
            "baselineCorrectionStart": self.baselineCorrectionStart.value,
            "baselineCorrectionEnd": self.baselineCorrectionEnd.value,
            "peak_startPoint": list(self.df_widget.value["Peak Start time"]),  # startPoint.value,
            "peak_endPoint": list(self.df_widget.value["Peak End time"]),  # endPoint.value,
            "selectForComputePsth": self.computePsth.value,
            "selectForTransientsComputation": self.transients.value,
            "moving_window": self.moving_wd.value,
            "highAmpFilt": self.highAmpFilt.value,
            "transientsThresh": self.transientsThresh.value,
            "plot_zScore_dff": self.plot_zScore_dff.value,
            "visualize_zscore_or_dff": self.visualize_zscore_or_dff.value,
            "folderNamesForAvg": self.files_2.value,
            "averageForGroup": self.averageForGroup.value,
            "visualizeAverageResults": self.visualizeAverageResults.value,
        }
        return inputParameters

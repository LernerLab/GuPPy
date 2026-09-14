import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn

from .dandi_selector import DandiSelector
from .frontend_utils import default_root_path
from ..utils.utils import (
    common_parent_directory,
    discover_run_folders,
    is_group_folder,
    parse_run_name,
    run_directory_root,
    run_folder_for_run,
    session_is_under_data_root,
)
from ..utils.validation import (
    validate_non_negative,
    validate_output_base_directory,
    validate_positive,
    validate_required_folder_selection,
    validate_significance_level,
)

logger = logging.getLogger(__name__)

# The two output layouts the form offers, as they read on the radio buttons.
SEPARATE_OUTPUT_DIRECTORY = "Mirror the data root into a separate output directory"
OUTPUTS_INSIDE_SESSION = "Write each run inside the session folder it came from"

# Width of each parameter section inside the 1000px Individual Analysis card.
SECTION_WIDTH = 960
# White against the card's WhiteSmoke, with a border, so the sections read as
# distinct blocks rather than merging into the card behind them.
SECTION_STYLES = {
    "background": "White",
    "border": "1px solid #C8CCD0",
    "border-radius": "6px",
    "margin-bottom": "12px",
}


def _reject_group_folder_selected_as_run(*, path: str) -> None:
    """Raise when a group output directory is selected where a session run is expected.

    Parameters
    ----------
    path : str
        A path selected in the individual-analysis session or output selectors.

    Raises
    ------
    ValueError
        If ``path`` names a ``<group_name>_group`` directory.
    """
    if is_group_folder(path):
        message = (
            f"'{path}' is a group output directory, not a session run. Groups are created in the "
            "Group Analysis card and can only be opened by Step 5 (Visualization); Steps 2-4 need "
            "the raw traces a group directory does not contain."
        )
        logger.error(message)
        raise ValueError(message)


def _blank_comparison_rows(count: int) -> pd.DataFrame:
    """Build an empty PSTH comparison table of ``count`` rows.

    Parameters
    ----------
    count : int
        Number of blank rows.

    Returns
    -------
    pd.DataFrame
        Table with the ``Event A`` and ``Event B`` columns the form reads.
    """
    return pd.DataFrame({"Event A": [""] * count, "Event B": [""] * count})


def _titled_box(*, title: str, read_by: str, contents: list, width: int) -> pn.WidgetBox:
    """Build one parameter section, headed by its title and the steps that consume it.

    Parameters
    ----------
    title : str
        Heading for the section.
    read_by : str
        The pipeline steps whose workers read the section's parameters, phrased to
        follow "Read by" (e.g. ``"Step 3"``).
    contents : list
        Panel objects to lay out under the heading.
    width : int
        Fixed width of the returned box, in pixels.

    Returns
    -------
    panel.WidgetBox
        The assembled section.
    """
    heading = pn.pane.Markdown(f"### {title}", width=width - 40, styles={"margin-bottom": "0"})
    read_by_note = pn.pane.Markdown(
        f"*Read by {read_by}*",
        width=width - 40,
        styles={"color": "#6C757D", "font-size": "0.85em", "margin-top": "0"},
    )
    return pn.WidgetBox(heading, read_by_note, *contents, width=width, styles=SECTION_STYLES)


def _table_heading(*, label: str, description: str, width: int) -> pn.Row:
    """Label a Tabulator and give it the help icon Tabulator itself cannot carry.

    Parameters
    ----------
    label : str
        Name shown above the table.
    description : str
        Help text the icon reveals.
    width : int
        Width of the label pane, in pixels.

    Returns
    -------
    panel.Row
        The label beside its help icon.
    """
    return pn.Row(
        pn.pane.Markdown(f"**{label}**", width=width, styles={"margin-bottom": "0"}),
        pn.widgets.TooltipIcon(value=description),
    )


def _preselect(selector: pn.widgets.FileSelector, path: str) -> None:
    """Select ``path`` in a file browser and show it in the Selected files pane.

    Assigning ``value`` alone sets the parameter without redrawing the browser, so the
    choice would not appear until something else made the widget re-list its directory.

    Parameters
    ----------
    selector : pn.widgets.FileSelector
        The browser to select in.
    path : str
        Absolute path to select.
    """
    selector.value = [path]
    selector._update_files()


class ParameterForm:
    """Panel form collecting all GuPPy analysis parameters.

    Builds and owns every input widget for the individual-analysis, group-
    analysis, and visualization sections, then appends them to the provided
    Panel template's main area.

    Parameters
    ----------
    template : panel.template.base.BasicTemplate
        The Panel template whose ``main`` area will receive the form cards.
    start_path : str, optional
        Initial directory shown in the file selectors. Falls back to the
        value returned by ``default_root_path()`` when not supplied or when
        the path does not exist.
    """

    def __init__(
        self,
        *,
        template: object,
        start_path: str | None = None,
        data_root: str | None = None,
        output_base_directory: str | None = None,
    ) -> None:
        self.template = template
        self.folder_path = start_path if start_path and Path(start_path).is_dir() else default_root_path()
        self.styles = dict(background="WhiteSmoke")
        # Sessions the run selection was last synced against, so a change can tell which
        # sessions are new and should inherit the bulk run-name choice.
        self._sessions_in_run_selection: list[str] = []
        # Set while the bulk picker is written to programmatically, so a value dropped for
        # being gone from disk is not mistaken for the user deselecting it.
        self._suppressing_run_name_propagation = False
        # Run selections stashed per source mode, so switching to DANDI and back does not
        # discard the choices made for local sessions (and vice versa).
        self._run_selection_by_source_mode: dict[str, tuple[list[str], list[str]]] = {}

        self.setup_individual_parameters()
        self.setup_group_parameters()
        self.add_to_template()
        self.files_1.param.watch(self._on_sessions_changed, "value")
        self.run_names_for_all_sessions.param.watch(self._on_run_names_for_all_sessions_change, "value")
        self.outputs_selector.param.watch(self._load_parameters_from_selected_runs, "value")
        self.output_location_mode.param.watch(self._on_output_location_changed, "value")
        self.output_base_selector.param.watch(self._on_output_location_changed, "value")
        self.data_root_selector.param.watch(self._on_data_root_changed, "value")
        self.dandi_selector.output_root_selector.param.watch(self._on_sessions_changed, "value")
        self.dandi_selector.attach_asset_selection_watcher(callback=self._on_sessions_changed)

        # Pre-select the directories named on the command line, which are the ones a user
        # keeps across sessions, so only the session folders are left to pick each time.
        if data_root and Path(data_root).is_dir():
            _preselect(self.data_root_selector, str(Path(data_root)))
        if output_base_directory and Path(output_base_directory).is_dir():
            _preselect(self.output_base_selector, str(Path(output_base_directory)))

    def setup_individual_parameters(self) -> None:
        """Build all widgets for the individual-analysis card and store them as instance attributes."""
        # Individual analysis components
        # Color the "dandi" button muted pink (matches the DANDI brain-logo palette) so
        # the two options are visually distinguishable at a glance.
        dandi_button_stylesheet = """
        :host .bk-btn-group > button:nth-child(2) {
            background-color: #E8B4BC !important;
            border-color: #D89AA3 !important;
            color: #3A2A2F !important;
        }
        :host .bk-btn-group > button:nth-child(2).bk-active {
            background-color: #C98A94 !important;
            border-color: #B56E78 !important;
            color: #FFFFFF !important;
        }
        """
        self.source_mode = pn.widgets.RadioButtonGroup(
            name="Data Source",
            options=["local", "dandi"],
            value="local",
            button_type="primary",
            width=300,
            stylesheets=[dandi_button_stylesheet],
        )
        self.source_mode.param.watch(self._on_source_mode_change, "value")

        self.data_root_header = pn.pane.Markdown(
            "**Data root.** Pick the directory your session folders live under. GuPPy mirrors "
            "each session's path below it into the output directory, so "
            "`<data root>/subject1/session1` writes its runs to "
            "`<output directory>/subject1/session1`.",
            width=950,
        )
        self.data_root_selector = pn.widgets.FileSelector(
            self.folder_path,
            root_directory="/",
            name="Data root",
            width=950,
        )

        self.session_selector_header = pn.pane.Markdown(
            "**Session folders.** Pick the sessions to analyze. Each one must sit under the "
            "data root you chose above.",
            width=950,
        )
        self.files_1 = pn.widgets.FileSelector(self.folder_path, root_directory="/", name="session_folders", width=950)

        self.dandi_selector = DandiSelector(styles=self.styles, start_path=self.folder_path)
        # Hidden by default; shown when source_mode == "dandi"
        self.dandi_selector.panel.visible = False

        self.timeForLightsTurnOn = pn.widgets.IntInput(
            name="Eliminate first few seconds (int)",
            value=1,
            width=320,
            description="Seconds dropped from the start of every recording, discarding the bright transient from when the LED first turns on. Applies to every session in the batch; to cut deeper into one recording, mark its opening as an artifact period instead.",
        )

        self.isosbestic_control = pn.widgets.Select(
            name="Isosbestic Control Channel? (bool)",
            value=True,
            options=[True, False],
            width=310,
            description="Whether the recording includes an isosbestic control channel. When False, GuPPy fits an exponential decay to the signal itself and uses that as a stand-in control, which removes the photobleaching trend but not motion artifacts.",
        )

        self.control_fit_method = pn.widgets.Select(
            name="Control Channel Fitting Method",
            options=["IRWLS", "OLS"],
            value="IRWLS",
            width=310,
            description="How the control channel is rescaled onto the signal before subtraction. IRWLS down-weights outlier samples so transients do not distort the fit; OLS is a plain least-squares fit.",
        )

        self.control_fit_window_mode = pn.widgets.Select(
            name="Control Fit Window",
            options=["full trace", "baseline epoch"],
            value="full trace",
            width=310,
            description="Which part of the recording the fit is estimated from. 'full trace' uses the whole recording; 'baseline epoch' uses only the window set beside it and applies those coefficients throughout, for sessions where a sustained step change such as a drug injection would otherwise distort the fit.",
        )
        self.control_fit_window_strt = pn.widgets.IntInput(
            name="Control Fit Window Start Time (s) (int)",
            value=0,
            width=310,
            description="Start of the baseline epoch the fit is estimated from, in seconds. Must be less than the end and fall inside the recording. Ignored when the fit window is 'full trace'.",
        )
        self.control_fit_window_end = pn.widgets.IntInput(
            name="Control Fit Window End Time (s) (int)",
            value=0,
            width=310,
            description="End of the baseline epoch the fit is estimated from, in seconds. Must be greater than the start and fall inside the recording. Ignored when the fit window is 'full trace'.",
        )

        self.photobleaching_detrend = pn.widgets.Select(
            name="Photobleaching Detrend? (bool)",
            value=False,
            options=[True, False],
            width=310,
            description="Adds an exponential decay term to the control fit, removing the residual photobleaching the control channel does not see. Useful for long recordings. Requires an isosbestic control channel and the OLS fitting method.",
        )

        self.numberOfCores = pn.widgets.IntInput(
            name="# of cores (int)",
            value=2,
            width=150,
            description="Number of CPU workers used for the per-channel steps. Keep it at or below the number of cores in your machine; setting it higher does not help.",
        )

        self.combine_data = pn.widgets.Select(
            name="Combine Data? (bool)",
            value=False,
            options=[True, False],
            width=150,
            description="Set to True when one recording session was written as two separate data files; the matching channels are concatenated into a single trace before preprocessing.",
        )

        self.output_location_mode = pn.widgets.RadioBoxGroup(
            name="Output location",
            options=[SEPARATE_OUTPUT_DIRECTORY, OUTPUTS_INSIDE_SESSION],
            value=SEPARATE_OUTPUT_DIRECTORY,
            width=620,
        )
        self.output_location_header = pn.pane.Markdown(
            "**Output directory.** Pick the directory the mirrored output tree is written into. "
            "A session's runs land in `<output directory>/<session path under the data root>/"
            "output_<run name>`, so nothing is written into your session folders.",
            width=950,
        )
        self.output_base_selector = pn.widgets.FileSelector(
            self.folder_path,
            root_directory="/",
            name="Output directory",
            width=950,
        )

        self.outputs_selector_header = pn.pane.Markdown(
            "**Existing runs (steps 2–5):** Pick at least one existing output directory per "
            "selected session. Naming a run below selects it in every session that has it; the "
            "tree underneath shows the result and can be adjusted session by session. To create "
            "a new run, use the Label Stores GUI in step 1.",
            width=950,
        )
        self.run_names_for_all_sessions = pn.widgets.MultiChoice(
            name="Run name(s) for all sessions",
            value=[],
            options=[],
            placeholder="Select runs by name across all sessions",
            width=620,
        )
        self.outputs_selector = pn.widgets.FileSelector(
            self.folder_path,
            root_directory="/",
            file_pattern="*output_*",
            name="Existing runs (steps 2–5)",
            width=950,
        )

        self.computePsth = pn.widgets.Select(
            name="z_score and/or \u0394F/F? (psth)",
            options=["z_score", "dff", "Both"],
            width=320,
            description="Which metric Step 4 aligns events on. 'Both' writes a complete set of PSTH outputs for each metric.",
        )

        self.transients = pn.widgets.Select(
            name="z_score and/or \u0394F/F? (transients)",
            options=["z_score", "dff", "Both"],
            width=320,
            description="Which metric the transient detector runs on. 'Both' runs it on each metric in turn.",
        )

        self.moving_wd = pn.widgets.IntInput(
            name="Moving Window for transients detection (s) (int)",
            value=15,
            width=380,
            description="Width of the moving window transients are detected in, in seconds.",
        )

        self.highAmpFilt = pn.widgets.IntInput(
            name="HAFT (int)",
            value=2,
            width=150,
            description="High-amplitude filtering threshold. Events greater than this many MADs above the median are filtered out before transients are detected.",
        )

        self.transientsThresh = pn.widgets.IntInput(
            name="TD Thresh (int)",
            value=3,
            width=160,
            description="Transient detection threshold. Peaks with local maxima greater than this many MADs above the median of the filtered trace are detected as transients.",
        )

        self.computeBinnedMetrics = pn.widgets.Select(
            name="Compute Binned Metrics? (bool)",
            options=[True, False],
            value=False,
            width=250,
            description="Divides the session into equal time bins and reports the mean z-score, mean dF/F and transient count in each. Useful for correlating the signal against a behavioral measure scored at a fixed cadence.",
        )

        self.binnedMetricsWidth = pn.widgets.IntInput(
            name="Bin Width (s) (int)",
            value=120,
            width=170,
            description="Width of those bins in seconds. The last bin is kept even when the session does not divide evenly, so it may be shorter than the rest.",
        )

        self.moving_avg_filter = pn.widgets.IntInput(
            name="Window for Moving Average filter (int)",
            value=100,
            width=320,
            description="Width of the moving-average smoothing kernel applied to the control and signal traces, in samples rather than seconds. The default suits recordings around 1 kHz; lower it proportionally for slower acquisition rates.",
        )

        self.z_score_computation = pn.widgets.Select(
            name="z-score computation Method",
            options=["standard z-score", "baseline z-score", "modified z-score"],
            value="standard z-score",
            width=260,
            description="How each trace is normalized. The z-score explainer in the documentation covers what the three methods do and which one suits which recording.",
        )

        self.baseline_wd_strt = pn.widgets.IntInput(
            name="Baseline Window Start Time (s) (int)",
            value=0,
            width=290,
            description="Start of the epoch the baseline z-score method normalizes against, in seconds. Leave at 0 for the other two methods. Must be less than the end and fall inside the recording.",
        )
        self.baseline_wd_end = pn.widgets.IntInput(
            name="Baseline Window End Time (s) (int)",
            value=0,
            width=280,
            description="End of the epoch the baseline z-score method normalizes against, in seconds. Leave at 0 for the other two methods. Must be greater than the start and fall inside the recording.",
        )

        self.nSecPrev = pn.widgets.IntInput(
            name="Seconds before 0 (int)",
            value=-10,
            width=190,
            description="Start of the peri-event window, in seconds relative to each event timestamp. Negative values reach back before the event, so the default of -10 opens the window 10 seconds before it; a positive value would start the window after the event instead.",
        )

        self.nSecPost = pn.widgets.IntInput(
            name="Seconds after 0 (int)",
            value=20,
            width=180,
            description="End of the peri-event window, in seconds relative to each event timestamp.",
        )

        self.computeCorr = pn.widgets.Select(
            name="Compute Cross-correlation (bool)",
            options=[True, False],
            value=False,
            width=260,
            description="Cross-correlates the PSTHs of two distinct signal recording sites, for detecting coordinated activity between areas. Requires at least two signal recording sites.",
        )

        self.computePsthSignificance = pn.widgets.Select(
            name="Compute PSTH Significance? (bool)",
            options=[True, False],
            value=False,
            width=270,
            description="Whether bootstrap confidence intervals and the comparison tests below are computed for each PSTH.",
        )

        self.psthSignificanceAlpha = pn.widgets.FloatInput(
            name="Significance Level (alpha) (float)",
            value=0.05,
            step=0.01,
            width=280,
            description="The two-sided threshold the confidence interval is computed at. 0.05 gives a 95% interval.",
        )

        self.psthBootstrapResamples = pn.widgets.IntInput(
            name="Bootstrap Resamples (int)",
            value=1000,
            step=100,
            width=210,
            description="How many times the trials are resampled to build each interval. More resamples means less run-to-run variation and a longer run.",
        )

        self.useTransientsAsEvents = pn.widgets.Select(
            name="Use Transients as Events? (bool)",
            options=[True, False],
            value=False,
            width=260,
            description="Uses each recording site's detected transients as its own event timestamps, for spontaneous activity with no external event TTLs. The PSTH, peak and area are then computed against them exactly as against a TTL train.",
        )

        self.timeInterval = pn.widgets.IntInput(
            name="Time Interval (s)",
            value=2,
            width=150,
            description="Minimum spacing between accepted event timestamps, in seconds. When two timestamps fall closer than this the second is dropped, so bursts do not produce double-counted overlapping windows.",
        )

        self.use_time_or_trials = pn.widgets.Select(
            name="Bin PSTH trials (str)",
            options=["Time (min)", "# of trials"],
            value="Time (min)",
            width=180,
            description="Whether PSTH trials are binned by elapsed time or by trial count.",
        )

        self.bin_psth_trials = pn.widgets.IntInput(
            name="Time(min) / # of trials for binning? (int)",
            value=0,
            width=330,
            description="Size of each bin, in the unit chosen beside it. Set to 0 to leave the trials unbinned.",
        )

        self.baselineCorrectionStart = pn.widgets.IntInput(
            name="Baseline Correction Start time(int)",
            value=-5,
            width=280,
            description="Start of the window each trial is baselined against, in seconds relative to the event. Set both bounds to 0 to skip baseline correction. Must lie inside the PSTH window.",
        )

        self.baselineCorrectionEnd = pn.widgets.IntInput(
            name="Baseline Correction End time(int)",
            value=0,
            width=280,
            description="End of the window each trial is baselined against, in seconds relative to the event. Set both bounds to 0 to skip baseline correction. Must lie inside the PSTH window.",
        )

        self.zscore_param_wd = _titled_box(
            title="Z-score Normalization",
            read_by="Step 3",
            contents=[
                pn.Row(self.z_score_computation, self.baseline_wd_strt, self.baseline_wd_end),
            ],
            width=SECTION_WIDTH,
        )

        self.psth_param_wd = _titled_box(
            title="PSTH Computation",
            read_by="Step 4 and Group Analysis",
            contents=[
                pn.Row(self.computePsth, self.nSecPrev, self.nSecPost),
                pn.Row(self.computeCorr),
                pn.Row(self.timeInterval, self.use_time_or_trials, self.bin_psth_trials),
                pn.Row(self.baselineCorrectionStart, self.baselineCorrectionEnd),
            ],
            width=SECTION_WIDTH,
        )

        self.start_end_point_df = pd.DataFrame(
            {
                "Peak Start time": [-5, 0, 5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                "Peak End time": [0, 3, 10, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )

        self.df_widget = pn.widgets.Tabulator(self.start_end_point_df, name="DataFrame", show_index=False, widths=280)

        self.auc_units = pn.widgets.Select(
            name="AUC Units (str)",
            options=["samples", "seconds"],
            value="samples",
            width=260,
            description="'seconds' reports the area in z-score (or dF/F) times seconds, the unit commonly reported in the literature. 'samples' integrates with one-sample spacing instead, so the area also scales with the recording's sampling rate.",
        )

        self.peak_param_wd = _titled_box(
            title="Peak and AUC Measurement",
            read_by="Step 4",
            contents=[
                _table_heading(
                    label="Peak and area windows",
                    description=(
                        "Each row is one window that peak amplitude and area are measured over, in "
                        "seconds relative to the event. Every window must lie inside the PSTH window. "
                        "The table commits an edit only once you click another cell."
                    ),
                    width=260,
                ),
                self.df_widget,
                self.auc_units,
            ],
            width=SECTION_WIDTH,
        )

        # One blank row to start, grown by the Add button rather than a fixed block of
        # slots: the number of worthwhile pairs scales with the square of the event count,
        # so any fixed size is both too many rows to look at and too few to hold.
        self.comparison_df = _blank_comparison_rows(1)

        self.comparison_df_widget = pn.widgets.Tabulator(
            self.comparison_df,
            name="Comparisons",
            show_index=False,
            widths=250,
            buttons={"remove": '<div title="Remove this comparison">\u2715</div>'},
        )
        self.comparison_df_widget.on_click(self._remove_comparison_row, column="remove")

        self.add_comparison_button = pn.widgets.Button(
            name="+ Add comparison", button_type="default", width=180, align="start"
        )
        self.add_comparison_button.on_click(self._add_comparison_row)

        self.significance_param_wd = _titled_box(
            title="Significance Testing",
            read_by="Step 4 and Group Analysis",
            contents=[
                pn.Row(self.computePsthSignificance, self.psthSignificanceAlpha, self.psthBootstrapResamples),
                _table_heading(
                    label="Event comparisons",
                    description=(
                        "Each row names a pair of events to compare against each other, for example "
                        "rewarded versus unrewarded nose pokes. Every event is tested against zero "
                        "automatically, so leave this blank to run only those tests. Each pair is "
                        "compared within every recording site and metric, using the labels assigned in "
                        "Step 1. Comparisons run inside one output folder: in a session run folder the "
                        "trials are resampled, in a group folder the session averages are."
                    ),
                    width=200,
                ),
                self.comparison_df_widget,
                self.add_comparison_button,
            ],
            width=SECTION_WIDTH,
        )

        self.execution_param_wd = _titled_box(
            title="Parallel Execution",
            read_by="Steps 2 and 4 and Group Analysis",
            contents=[self.numberOfCores],
            width=SECTION_WIDTH,
        )

        self.control_fit_param_wd = _titled_box(
            title="Control Channel Fitting",
            read_by="Step 3",
            contents=[
                pn.Row(self.isosbestic_control, self.control_fit_method),
                pn.Row(self.photobleaching_detrend, self.control_fit_window_mode),
                pn.Row(self.control_fit_window_strt, self.control_fit_window_end),
            ],
            width=SECTION_WIDTH,
        )

        self.filtering_param_wd = _titled_box(
            title="Signal Filtering",
            read_by="Step 3 and Group Analysis",
            contents=[pn.Row(self.timeForLightsTurnOn, self.moving_avg_filter)],
            width=SECTION_WIDTH,
        )

        self.transients_param_wd = _titled_box(
            title="Transient Detection",
            read_by="Steps 4 and 5 and Group Analysis",
            contents=[
                pn.Row(self.transients, self.useTransientsAsEvents),
                pn.Row(self.moving_wd, self.highAmpFilt, self.transientsThresh),
            ],
            width=SECTION_WIDTH,
        )

        self.binned_metrics_param_wd = _titled_box(
            title="Metric Binning",
            read_by="Step 4",
            contents=[
                pn.Row(self.computeBinnedMetrics, self.binnedMetricsWidth),
            ],
            width=SECTION_WIDTH,
        )

        # One column, ordered by the step that reads each section, so the card reads
        # straight down rather than leaving the reader to guess a column order.
        self.individual_parameters = pn.Column(
            self.execution_param_wd,
            self.control_fit_param_wd,
            self.filtering_param_wd,
            self.zscore_param_wd,
            self.psth_param_wd,
            self.peak_param_wd,
            self.transients_param_wd,
            self.binned_metrics_param_wd,
            self.significance_param_wd,
        )

        self.input_folder_selection_widget = pn.Column(
            pn.Row(pn.pane.Markdown("**Data Source:**"), self.source_mode),
            self.data_root_header,
            self.data_root_selector,
            pn.layout.Divider(),
            self.session_selector_header,
            self.files_1,
            self.dandi_selector.panel,
            self.combine_data,
        )
        self.input_folder_selection = pn.Card(
            self.input_folder_selection_widget,
            title="Input Folder Selection",
            styles=self.styles,
            width=1000,
        )

        self.output_folder_selection_widget = pn.Column(
            self.output_location_header,
            self.output_location_mode,
            self.output_base_selector,
            pn.layout.Divider(),
            self.outputs_selector_header,
            self.run_names_for_all_sessions,
            self.outputs_selector,
        )
        self.output_folder_selection = pn.Card(
            self.output_folder_selection_widget,
            title="Output Folder Selection",
            styles=self.styles,
            width=1000,
            collapsed=True,
        )

        self.widget = pn.Column(self.individual_parameters)
        self.individual = pn.Card(
            self.widget, title="Parameter Selection", styles=self.styles, width=1000, collapsed=True
        )

    def _on_source_mode_change(self, event: object) -> None:
        is_dandi = event.new == "dandi"
        self.files_1.visible = not is_dandi
        self.dandi_selector.panel.visible = is_dandi
        self._run_selection_by_source_mode[event.old] = (
            list(self.outputs_selector.value or []),
            list(self.run_names_for_all_sessions.value),
        )
        run_folders, run_names = self._run_selection_by_source_mode.get(event.new, ([], []))
        self._restore_run_selection(run_folders=run_folders, run_names=run_names)

    @property
    def data_root(self) -> str | None:
        """Return the directory the session folders are selected under.

        Returns
        -------
        str or None
            The chosen data root, the DANDI output root in DANDI mode, or ``None``
            when neither has been chosen.
        """
        if self.source_mode.value == "dandi":
            return self.dandi_selector.output_root
        selected = list(self.data_root_selector.value or [])
        return str(selected[0]) if selected else None

    @property
    def output_base_directory(self) -> str | None:
        """Return the directory the mirrored output tree is written into.

        Returns
        -------
        str or None
            The chosen output directory, or ``None`` for the inside-the-session
            layout and while no directory has been chosen yet.
        """
        if self.output_location_mode.value == OUTPUTS_INSIDE_SESSION:
            return None
        selected = list(self.output_base_selector.value or [])
        return str(selected[0]) if selected else None

    def _on_data_root_changed(self, event: object = None) -> None:
        """Start the session browser inside the newly chosen data root."""
        data_root = self.data_root
        if data_root:
            self.files_1.directory = data_root
        self._on_output_location_changed()

    def _session_by_run_folder(self) -> dict[str, str]:
        """Map each run folder on disk back to the selected session it belongs to."""
        output_base_directory = self.output_base_directory
        data_root = self.data_root
        return {
            str(Path(run_folder)): session
            for session in self._sessions_for_run_selection()
            for run_folder in discover_run_folders(
                session, output_base_directory=output_base_directory, data_root=data_root
            )
        }

    def _collect_selected_runs(self) -> dict[str, list[str]]:
        """Group the FileSelector's selected output dirs by the session they belong to.

        Raises
        ------
        ValueError
            If a group output directory was selected as an individual run, or if a
            selected run folder belongs to no selected session.
        """
        session_by_run_folder = self._session_by_run_folder()
        grouped: dict[str, list[str]] = {}
        for path in self.outputs_selector.value or []:
            _reject_group_folder_selected_as_run(path=path)
            session = session_by_run_folder.get(str(Path(path)))
            if session is None:
                raise ValueError(
                    f"Output directory {path!r} does not belong to any selected session. Select the "
                    "session it was written for in the Input Folder Selection card, or check that "
                    "the Output Location setting points at the directory holding it."
                )
            grouped.setdefault(session, []).append(parse_run_name(path))
        return grouped

    def validate_selected_runs_for_consumers(self) -> None:
        """Ensure every selected session that has output dirs on disk also has at least one selected.

        Run this from the click handlers for steps 2–5 (which consume existing
        output directories). Skips sessions with no ``_output_<run>`` dirs
        yet — those are typically pre-step-1 states.
        """
        for session in self.files_1.value or []:
            _reject_group_folder_selected_as_run(path=session)
        grouped = self._collect_selected_runs()
        output_base_directory = self.output_base_directory
        data_root = self.data_root
        missing = [
            session
            for session in (self.files_1.value or [])
            if discover_run_folders(session, output_base_directory=output_base_directory, data_root=data_root)
            and not grouped.get(session)
        ]
        if missing:
            raise ValueError(
                f"No output directory selected for session(s) {missing!r}. "
                "Open the Output Folder Selection panel and pick at least one run "
                "directory per selected session."
            )
        self._warn_about_runs_only_in_session_folders()

    def _warn_about_runs_only_in_session_folders(self) -> None:
        """Warn when a session's only runs sit inside the session folder."""
        if self.output_base_directory is None:
            return
        stranded = [
            session
            for session in (self.files_1.value or [])
            if discover_run_folders(session)
            and not discover_run_folders(
                session, output_base_directory=self.output_base_directory, data_root=self.data_root
            )
        ]
        if stranded:
            self._notify(
                "warning",
                f"Session(s) {stranded!r} hold output directories inside the session folder, which "
                f"the output directory {self.output_base_directory!r} does not list. Set Output "
                f"Location to '{OUTPUTS_INSIDE_SESSION}' to work with them.",
            )

    def _prospective_sessions(self) -> list[str]:
        """Return the session directories the form is pointed at, created or not."""
        if self.source_mode.value == "dandi":
            return self._prospective_dandi_sessions()
        return list(self.files_1.value or [])

    def _sessions_for_run_selection(self) -> list[str]:
        """Return the existing session directories whose runs steps 2-5 will read.

        A session the form cannot place in the output tree yet — because no data root has
        been chosen, or because it sits outside the one that has — contributes no runs to
        browse. ``getInputParameters`` is what reports it, when an analysis is started.
        """
        sessions = [session for session in self._prospective_sessions() if Path(session).is_dir()]
        if self.output_base_directory is None:
            return sessions
        data_root = self.data_root
        return [
            session for session in sessions if session_is_under_data_root(session_path=session, data_root=data_root)
        ]

    def _run_names_for_sessions(self, sessions: list[str]) -> list[str]:
        """Return every run name present in at least one of ``sessions``, in run-name order.

        A name need not exist in every session to be offered: applying it selects the runs
        that do exist and leaves the remaining sessions to the folder tree.
        """
        output_base_directory = self.output_base_directory
        data_root = self.data_root
        run_names = []
        for session in sessions:
            for directory in discover_run_folders(
                session, output_base_directory=output_base_directory, data_root=data_root
            ):
                run_name = parse_run_name(directory)
                if run_name not in run_names:
                    run_names.append(run_name)
        return run_names

    def _apply_selected_run_folders(self, run_folders: list[str]) -> None:
        """Write ``run_folders`` into the outputs FileSelector as its selection.

        Setting ``value`` updates the parameter but not the widget's "Selected files" pane;
        ``_update_files`` re-enumerates so programmatic picks appear there — including runs
        outside the directory currently shown, which it lists by their relative path.
        """
        deduplicated = list(dict.fromkeys(run_folders))
        if deduplicated != list(self.outputs_selector.value or []):
            self.outputs_selector.value = deduplicated
        self.outputs_selector._update_files()

    def _retarget_outputs_selector(self, sessions: list[str]) -> None:
        """Root the existing-runs FileSelector so all selected sessions' run dirs are reachable.

        - An output base directory holds the mirrored tree for every session, so it is both
          root and starting directory. The cases below are the inside-the-session layout,
          where each session holds its own runs:
        - Zero sessions: fall back to the form's starting directory.
        - One session: root and starting directory both set to that session so its `_output_*`
          children show directly (no extra click).
        - Multiple sessions: root set to their common parent so every session is navigable;
          starting directory set to the first session so the user lands on one session's
          outputs and can navigate up to switch between sessions.
        - DANDI mode: root set to the chosen output root, which holds every mirrored session.
        """
        output_base_directory = self.output_base_directory
        dandi_output_root = self.dandi_selector.output_root if self.source_mode.value == "dandi" else None
        if output_base_directory is not None:
            root_target = output_base_directory
            directory_target = output_base_directory
        elif dandi_output_root:
            root_target = dandi_output_root
            directory_target = sessions[0] if sessions else dandi_output_root
        elif not sessions:
            root_target = self.folder_path
            directory_target = self.folder_path
        elif len(sessions) == 1:
            root_target = sessions[0]
            directory_target = sessions[0]
        else:
            root_target = os.path.commonpath(sessions)
            directory_target = sessions[0]
        # Set root_directory before directory so Panel's `path.startswith(self._root_directory)`
        # check in FileSelector._dir_change can't silently revert (Windows-specific failure mode
        # when the constructor's root_directory="/" resolves to a drive root that isn't shared
        # with tmp_path or the user's session folder).
        self.outputs_selector.root_directory = root_target
        self.outputs_selector.directory = directory_target
        # Sync the FileSelector's internal _cwd and re-enumerate. Without this, _cwd remains
        # at the construction-time path; clicking a sub-dir uses the stale _cwd to compute
        # the navigated path, that path doesn't exist, and the FileSelector silently snaps
        # back to the stale _cwd — visible to the user as "selection resets the directory".
        self.outputs_selector._update_files()

    def _refresh_run_name_options(self, sessions: list[str]) -> None:
        """Re-read the run names on disk into the bulk picker, dropping any that no longer exist."""
        run_names = self._run_names_for_sessions(sessions)
        self.run_names_for_all_sessions.options = run_names
        surviving = [run_name for run_name in self.run_names_for_all_sessions.value if run_name in run_names]
        if surviving != list(self.run_names_for_all_sessions.value):
            # A name drops out only when the sessions holding it are gone, so the paths it
            # would deselect have already left the selection with them.
            self._suppressing_run_name_propagation = True
            self.run_names_for_all_sessions.value = surviving
            self._suppressing_run_name_propagation = False

    def _on_sessions_changed(self, event: object = None) -> None:
        """Retarget the folder tree and carry the run selection across a change of sessions.

        Sessions that survive the change keep the runs already picked for them, so dropping
        one session does not discard the choices made for the others. A session new to the
        selection starts out with the runs the bulk picker currently names.
        """
        sessions = self._sessions_for_run_selection()
        new_sessions = [session for session in sessions if session not in self._sessions_in_run_selection]
        self._sessions_in_run_selection = sessions
        self._retarget_outputs_selector(sessions)

        session_by_run_folder = self._session_by_run_folder()
        selected = [path for path in (self.outputs_selector.value or []) if str(Path(path)) in session_by_run_folder]
        selected += self._run_folders_on_disk(sessions=new_sessions, run_names=self.run_names_for_all_sessions.value)
        self._apply_selected_run_folders(selected)
        self._refresh_run_name_options(sessions)

    def _restore_run_selection(self, *, run_folders: list[str], run_names: list[str]) -> None:
        """Point the card at the current source mode's sessions and put back its stashed selection."""
        sessions = self._sessions_for_run_selection()
        self._sessions_in_run_selection = sessions
        self._retarget_outputs_selector(sessions)
        self._refresh_run_name_options(sessions)
        self._suppressing_run_name_propagation = True
        self.run_names_for_all_sessions.value = [
            run_name for run_name in run_names if run_name in self.run_names_for_all_sessions.options
        ]
        self._suppressing_run_name_propagation = False
        session_by_run_folder = self._session_by_run_folder()
        self._apply_selected_run_folders([path for path in run_folders if str(Path(path)) in session_by_run_folder])

    def _on_run_names_for_all_sessions_change(self, event: object) -> None:
        """Select or deselect the runs matching the bulk choice, leaving hand-picked ones alone."""
        if self._suppressing_run_name_propagation:
            return
        added = [run_name for run_name in event.new if run_name not in event.old]
        removed = [run_name for run_name in event.old if run_name not in event.new]
        kept = [path for path in (self.outputs_selector.value or []) if parse_run_name(path) not in removed]
        sessions = self._sessions_for_run_selection()
        self._apply_selected_run_folders(kept + self._run_folders_on_disk(sessions=sessions, run_names=added))

    def _run_folders_on_disk(self, *, sessions: list[str], run_names: list[str]) -> list[str]:
        """Return the existing run directories named by ``run_names`` across ``sessions``."""
        output_base_directory = self.output_base_directory
        data_root = self.data_root
        run_folders = []
        for session in sessions:
            for run_name in run_names:
                run_folder = run_folder_for_run(
                    session, run_name, output_base_directory=output_base_directory, data_root=data_root
                )
                if Path(run_folder).is_dir():
                    run_folders.append(run_folder)
        return run_folders

    def _on_output_location_changed(self, event: object = None) -> None:
        """Re-point the existing-runs tree at wherever the outputs now live.

        The run folders the selection named are gone from the new location, so every
        session is treated as newly selected and re-inherits the bulk run-name choice.
        """
        self._sessions_in_run_selection = []
        self._apply_selected_run_folders([])
        self._on_sessions_changed()

    def refresh_individual_outputs(self) -> None:
        """Re-list the outputs FileSelector so newly-created run dirs (e.g. from step 1) appear."""
        sessions = self._sessions_for_run_selection()
        # Step 1 may have just created the output base directory the tree could not be rooted
        # at before, so re-point it before re-listing.
        self._retarget_outputs_selector(sessions)
        self.outputs_selector._refresh()
        self._refresh_run_name_options(sessions)

    def _prospective_dandi_sessions(self) -> list[str]:
        """Return the local session directories the selected DANDI assets map to.

        The directories are not created here; ``_resolve_dandi_sessions`` does that when
        the pipeline actually runs.
        """
        output_root = self.dandi_selector.output_root
        if not output_root:
            return []
        sessions = []
        for uri in self.dandi_selector.selected_uris:
            asset_path = uri.split("/", 3)[-1]
            session_stem = Path(asset_path).stem
            sessions.append(str(Path(output_root) / session_stem))
        return sessions

    def _resolve_dandi_sessions(self) -> tuple[list[str], str, dict[str, str]]:
        """
        Materialize DANDI asset selections into local session directories.

        For each selected ``dandi://`` URI, create a directory under the user-chosen
        output root named after the asset's basename (minus suffix). The returned
        ``dandi_uri_map`` is keyed by that session directory — matching the key
        used by the orchestration layer when ``mode == "dandi"``.

        Returns
        -------
        folder_names : list[str]
            Absolute paths of the created session directories.
        output_root : str
            The user-chosen local output root.
        dandi_uri_map : dict[str, str]
            Mapping from session directory to the originating DANDI URI.
        """
        selected_uris = self.dandi_selector.selected_uris
        output_root = self.dandi_selector.output_root
        if not selected_uris:
            logger.error("DANDI mode: no NWB assets selected")
            raise ValueError("DANDI mode: select at least one NWB asset before running the pipeline")
        if not output_root:
            logger.error("DANDI mode: no local output directory selected")
            raise ValueError("DANDI mode: select a local output directory before running the pipeline")

        folder_names = self._prospective_dandi_sessions()
        for session_directory in folder_names:
            Path(session_directory).mkdir(parents=True, exist_ok=True)
        dandi_uri_map = dict(zip(folder_names, selected_uris, strict=True))
        return folder_names, output_root, dandi_uri_map

    def _add_comparison_row(self, event: object = None) -> None:
        """Append a blank comparison row to the table."""
        self.comparison_df_widget.value = pd.concat(
            [self.comparison_df_widget.value, _blank_comparison_rows(1)], ignore_index=True
        )

    def _remove_comparison_row(self, event: object) -> None:
        """Drop the clicked comparison row, keeping one blank row when the last one goes."""
        remaining = self.comparison_df_widget.value.drop(index=event.row).reset_index(drop=True)
        self.comparison_df_widget.value = remaining if len(remaining) else _blank_comparison_rows(1)

    def setup_group_parameters(self) -> None:
        """Build the group output-folder selection card and store its widgets as attributes."""
        self.mark_down_2 = pn.pane.Markdown(
            "**Existing groups:** pick the `<name>_group` directories to work with. The Group "
            "Analysis step averages into them, and Step 5 opens them — the same selection serves "
            "both, so you choose it once. To define a new group, use the Label Groups step.",
            width=950,
        )
        self.group_folders_selector = pn.widgets.FileSelector(
            self.folder_path, root_directory="/", name="Group output directories", width=950
        )

        self.group_analysis_wd_1 = pn.Column(
            self.mark_down_2,
            self.group_folders_selector,
            width=980,
        )
        self.group = pn.Card(
            self.group_analysis_wd_1,
            title="Group Output Folder Selection",
            styles=self.styles,
            width=1000,
            collapsed=True,
        )

    def refresh_group_folders(self) -> None:
        """Re-list the group selector so groups created since the last interaction appear."""
        self.group_folders_selector._refresh()

    def add_to_template(self) -> None:
        """Append the input/output folder, individual, and group cards to the template's main area."""
        self.template.main.append(self.input_folder_selection)
        self.template.main.append(self.output_folder_selection)
        self.template.main.append(self.individual)
        self.template.main.append(self.group)

    def _validate_numeric_parameters(self) -> None:
        """Validate the scalar numeric parameters at config time.

        Enforces the documented positivity, non-negativity, ordering, and
        host-core constraints on the numeric widgets so bad values are rejected
        with an informative message before any pipeline step starts, instead of
        failing late (or silently producing wrong results) mid-analysis. The
        step handlers surface the raised ``ValueError`` as a Panel notification.

        Raises
        ------
        ValueError
            If any numeric parameter is out of its documented range.
        """
        number_of_cores = self.numberOfCores.value
        validate_positive(value=number_of_cores, name="numberOfCores")
        available_cores = os.cpu_count() or 1
        if number_of_cores > available_cores:
            message = (
                f"numberOfCores={number_of_cores} exceeds the {available_cores} core(s) available on "
                f"this machine; choose a value between 1 and {available_cores}."
            )
            logger.error(message)
            raise ValueError(message)

        # filter_window and timeForLightsTurnOn accept 0 (0 disables filtering /
        # eliminates no data); the rest are strictly positive.
        validate_non_negative(value=self.moving_avg_filter.value, name="filter_window")
        validate_non_negative(value=self.timeForLightsTurnOn.value, name="timeForLightsTurnOn")
        validate_positive(value=self.moving_wd.value, name="moving_window")
        validate_positive(value=self.highAmpFilt.value, name="highAmpFilt")
        validate_positive(value=self.transientsThresh.value, name="transientsThresh")
        validate_positive(value=self.binnedMetricsWidth.value, name="binnedMetricsWidth")
        validate_significance_level(value=self.psthSignificanceAlpha.value, name="psthSignificanceAlpha")
        validate_positive(value=self.psthBootstrapResamples.value, name="psthBootstrapResamples")

        if self.nSecPrev.value >= self.nSecPost.value:
            message = (
                f"nSecPrev={self.nSecPrev.value} must be strictly less than nSecPost={self.nSecPost.value}; "
                "the PSTH window runs from nSecPrev (seconds before the event) to nSecPost (seconds after)."
            )
            logger.error(message)
            raise ValueError(message)

    def getInputParameters(self) -> dict[str, object]:
        """Collect and return all current widget values as an input-parameters dictionary.

        Returns
        -------
        dict
            Flat dictionary containing every parameter needed to run the GuPPy
            pipeline, keyed by the parameter names expected by the orchestration
            layer (e.g. ``"session_folders"``, ``"zscore_method"``, ``"nSecPrev"``).
        """
        self._validate_numeric_parameters()

        if self.source_mode.value == "dandi":
            folder_names, abspath_value, dandi_uri_map = self._resolve_dandi_sessions()
            mode = "dandi"
        else:
            # Local mode requires a selection somewhere: individual sessions, or the group
            # card's members or existing-group picker for a group-only workflow.
            validate_required_folder_selection(file_selectors=[self.files_1, self.group_folders_selector])
            folder_names = self.files_1.value
            abspath_value = common_parent_directory(paths=list(folder_names)) if folder_names else None
            dandi_uri_map = None
            mode = "local"

        data_root = self.data_root
        output_base_directory = self.output_base_directory
        if self.output_location_mode.value == SEPARATE_OUTPUT_DIRECTORY:
            if output_base_directory is None:
                raise ValueError(
                    "No output directory chosen. Pick one in the Output Folder Selection card, or set "
                    f"Output Location to '{OUTPUTS_INSIDE_SESSION}'."
                )
            if data_root is None:
                raise ValueError(
                    "No data root chosen. Pick the directory your session folders live under in the "
                    "Input Folder Selection card."
                )
            validate_output_base_directory(
                session_folders=list(folder_names),
                output_base_directory=output_base_directory,
                data_root=data_root,
            )
            # Created here rather than at save time so every step, and the Label Stores page
            # that writes the first run folder into it, can count on it existing.
            for session in folder_names:
                root = run_directory_root(
                    session_path=session, output_base_directory=output_base_directory, data_root=data_root
                )
                Path(root).mkdir(parents=True, exist_ok=True)

        inputParameters = {
            "mode": mode,
            "dandi_uri_map": dandi_uri_map,
            "abspath": abspath_value,
            "output_base_directory": output_base_directory,
            "data_root": data_root,
            "session_folders": folder_names,
            "numberOfCores": self.numberOfCores.value,
            "combine_data": self.combine_data.value,
            "isosbestic_control": self.isosbestic_control.value,
            "control_fit_method": self.control_fit_method.value,
            "controlFitWindowMode": self.control_fit_window_mode.value,
            "controlFitWindowStart": self.control_fit_window_strt.value,
            "controlFitWindowEnd": self.control_fit_window_end.value,
            "photobleaching_detrend": self.photobleaching_detrend.value,
            "timeForLightsTurnOn": self.timeForLightsTurnOn.value,
            "filter_window": self.moving_avg_filter.value,
            "zscore_method": self.z_score_computation.value,
            "baselineWindowStart": self.baseline_wd_strt.value,
            "baselineWindowEnd": self.baseline_wd_end.value,
            "nSecPrev": self.nSecPrev.value,
            "nSecPost": self.nSecPost.value,
            "computeCorr": self.computeCorr.value,
            "useTransientsAsEvents": self.useTransientsAsEvents.value,
            "timeInterval": self.timeInterval.value,
            "bin_psth_trials": self.bin_psth_trials.value,
            "use_time_or_trials": self.use_time_or_trials.value,
            "baselineCorrectionStart": self.baselineCorrectionStart.value,
            "baselineCorrectionEnd": self.baselineCorrectionEnd.value,
            "peak_startPoint": list(self.df_widget.value["Peak Start time"]),  # startPoint.value,
            "peak_endPoint": list(self.df_widget.value["Peak End time"]),  # endPoint.value,
            "computePsthSignificance": self.computePsthSignificance.value,
            "psthSignificanceAlpha": self.psthSignificanceAlpha.value,
            "psthBootstrapResamples": self.psthBootstrapResamples.value,
            "psthComparisonsA": list(self.comparison_df_widget.value["Event A"]),
            "psthComparisonsB": list(self.comparison_df_widget.value["Event B"]),
            "auc_units": self.auc_units.value,
            "selectForComputePsth": self.computePsth.value,
            "selectForTransientsComputation": self.transients.value,
            "moving_window": self.moving_wd.value,
            "highAmpFilt": self.highAmpFilt.value,
            "transientsThresh": self.transientsThresh.value,
            "computeBinnedMetrics": self.computeBinnedMetrics.value,
            "binnedMetricsWidth": self.binnedMetricsWidth.value,
            "selected_group_folders": list(self.group_folders_selector.value or []),
            "selected_runs": self._collect_selected_runs(),
        }
        return inputParameters

    def _scalar_parameter_widgets(self) -> dict[str, pn.widgets.Widget]:
        """Map each scalar analysis-parameter key to the widget whose ``.value`` holds it.

        Covers every key written to ``GuPPyParamtersUsed.json`` except the two
        peak-window columns (held in the ``df_widget`` Tabulator) and the
        provenance-only ``guppy_version``.

        Returns
        -------
        dict
            Mapping from JSON parameter key to its backing Panel widget.
        """
        return {
            "combine_data": self.combine_data,
            "isosbestic_control": self.isosbestic_control,
            "control_fit_method": self.control_fit_method,
            "controlFitWindowMode": self.control_fit_window_mode,
            "controlFitWindowStart": self.control_fit_window_strt,
            "controlFitWindowEnd": self.control_fit_window_end,
            "photobleaching_detrend": self.photobleaching_detrend,
            "timeForLightsTurnOn": self.timeForLightsTurnOn,
            "filter_window": self.moving_avg_filter,
            "zscore_method": self.z_score_computation,
            "baselineWindowStart": self.baseline_wd_strt,
            "baselineWindowEnd": self.baseline_wd_end,
            "nSecPrev": self.nSecPrev,
            "nSecPost": self.nSecPost,
            "computeCorr": self.computeCorr,
            "computePsthSignificance": self.computePsthSignificance,
            "psthSignificanceAlpha": self.psthSignificanceAlpha,
            "psthBootstrapResamples": self.psthBootstrapResamples,
            "useTransientsAsEvents": self.useTransientsAsEvents,
            "timeInterval": self.timeInterval,
            "bin_psth_trials": self.bin_psth_trials,
            "use_time_or_trials": self.use_time_or_trials,
            "baselineCorrectionStart": self.baselineCorrectionStart,
            "baselineCorrectionEnd": self.baselineCorrectionEnd,
            "auc_units": self.auc_units,
            "selectForComputePsth": self.computePsth,
            "selectForTransientsComputation": self.transients,
            "moving_window": self.moving_wd,
            "highAmpFilt": self.highAmpFilt,
            "transientsThresh": self.transientsThresh,
            "computeBinnedMetrics": self.computeBinnedMetrics,
            "binnedMetricsWidth": self.binnedMetricsWidth,
        }

    def setInputParameters(self, parameters: dict[str, object]) -> None:
        """Populate the form widgets from a saved-parameters dict (reverse of ``getInputParameters``).

        Only the analysis keys written to ``GuPPyParamtersUsed.json`` are
        applied; unknown keys (e.g. ``guppy_version``) are ignored.

        Parameters
        ----------
        parameters : dict
            Parameter dict as loaded from a ``GuPPyParamtersUsed.json`` file.
        """
        for key, widget in self._scalar_parameter_widgets().items():
            if key in parameters:
                widget.value = parameters[key]
        if "peak_startPoint" in parameters and "peak_endPoint" in parameters:
            df = self.df_widget.value.copy()
            df["Peak Start time"] = parameters["peak_startPoint"]
            df["Peak End time"] = parameters["peak_endPoint"]
            self.df_widget.value = df
        if "psthComparisonsA" in parameters and "psthComparisonsB" in parameters:
            # Rebuilt rather than assigned into: a saved run may hold any number of
            # comparisons, and assigning a longer list into the existing index raises.
            saved = pd.DataFrame({"Event A": parameters["psthComparisonsA"], "Event B": parameters["psthComparisonsB"]})
            self.comparison_df_widget.value = saved if len(saved) else _blank_comparison_rows(1)

    def _load_parameters_from_selected_runs(self, event: object) -> None:
        """Reload analysis parameters from the saved JSON of the selected output run(s).

        Fired when the individual-analysis output selector changes. Lets a user
        resume a run (e.g. relaunch and run steps 3–4) without the form's
        defaults silently overwriting the parameters the earlier steps used.
        When several runs are selected the parameters are applied only if every
        run with a saved snapshot agrees; conflicting snapshots are left for the
        user to reconcile.
        """
        saved = []
        for run_folder in event.new or []:
            json_path = Path(run_folder) / "GuPPyParamtersUsed.json"
            if json_path.exists():
                with json_path.open() as parameters_file:
                    saved.append(json.load(parameters_file))
        if not saved:
            return

        # Compare only the widget-backed analysis keys via JSON so NaN peak-window
        # entries compare equal (NaN != NaN under direct equality).
        keys = list(self._scalar_parameter_widgets().keys()) + ["peak_startPoint", "peak_endPoint"]

        def signature(params: dict[str, object]) -> str:
            return json.dumps({key: params.get(key) for key in keys}, sort_keys=True)

        reference = saved[0]
        if any(signature(params) != signature(reference) for params in saved[1:]):
            self._notify(
                "warning",
                "Selected output runs have different saved parameters; the form was left unchanged. "
                "Select runs that share parameters to auto-load them.",
            )
            return

        self.setInputParameters(reference)
        self._notify("info", "Loaded parameters from the selected output run(s).")

    @staticmethod
    def _notify(level: str, message: str) -> None:
        """Show a Panel notification when a notification area is available (no-op when headless)."""
        notifications = pn.state.notifications
        if notifications is not None:
            getattr(notifications, level)(message)

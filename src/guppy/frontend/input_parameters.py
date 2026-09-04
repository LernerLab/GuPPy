import json
import logging
import os

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
    run_folder_for_run,
)
from ..utils.validation import (
    validate_non_negative,
    validate_positive,
    validate_required_folder_selection,
    validate_significance_level,
)

logger = logging.getLogger(__name__)


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

    def __init__(self, *, template: object, start_path: str | None = None) -> None:
        self.template = template
        self.folder_path = start_path if start_path and os.path.isdir(start_path) else default_root_path()
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
        self.setup_visualization_parameters()
        self.add_to_template()
        self.files_1.param.watch(self._on_sessions_changed, "value")
        self.run_names_for_all_sessions.param.watch(self._on_run_names_for_all_sessions_change, "value")
        self.outputs_selector.param.watch(self._load_parameters_from_selected_runs, "value")
        self.dandi_selector.output_root_selector.param.watch(self._on_sessions_changed, "value")
        self.dandi_selector.attach_asset_selection_watcher(callback=self._on_sessions_changed)

    def setup_individual_parameters(self) -> None:
        """Build all widgets for the individual-analysis card and store them as instance attributes."""
        # Individual analysis components
        self.mark_down_1 = pn.pane.Markdown(
            """**Select folders for the analysis from the file selector below**""", width=600
        )

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

        self.files_1 = pn.widgets.FileSelector(self.folder_path, root_directory="/", name="session_folders", width=950)

        self.dandi_selector = DandiSelector(styles=self.styles, start_path=self.folder_path)
        # Hidden by default; shown when source_mode == "dandi"
        self.dandi_selector.panel.visible = False

        self.explain_time_artifacts = pn.pane.Markdown(
            """
                                - ***Number of cores :*** Number of cores used for analysis. Try to
                                keep it less than the number of cores in your machine.
                                - ***Combine Data? :*** Make this parameter ``` True ``` if user wants to combine
                                the data, especially when there is two different
                                data files for the same recording session.<br>
                                - ***Isosbestic Control Channel? :*** Make this parameter ``` False ``` if user
                                does not want to use isosbestic control channel in the analysis.<br>
                                - ***Photobleaching Detrend? :*** Make this parameter ``` True ``` to fit an
                                exponential decay to the corrected &#916;F/F and subtract it, removing the
                                residual photobleaching drift that remains after the control channel is
                                subtracted. Useful for long (multi-hour) recordings. Requires an isosbestic
                                control channel. Default is ``` False ```.<br>
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
                                - ***Compute Binned Metrics? :*** Make this parameter ``` True ``` to divide the
                                whole session into equal time bins and report the mean z-score, mean &#916;F/F and
                                number of transients in each one. Useful for correlating the signal against a
                                behavioral measure scored at a fixed cadence. Default is ``` False ```.<br>
                                - ***Bin Width :*** Width of those bins in seconds. The last bin is kept even
                                when the session does not divide evenly, so it may be shorter than the rest.
                                Default is 120 seconds.<br>
                                - ***Number of channels (Neurophotometrics only) :*** Number of
                                channels used while recording, when data files has no column names mentioning "Flags"
                                or "LedState".
                                """,
            width=350,
        )

        self.timeForLightsTurnOn = pn.widgets.IntInput(name="Eliminate first few seconds (int)", value=1, width=320)

        self.isosbestic_control = pn.widgets.Select(
            name="Isosbestic Control Channel? (bool)", value=True, options=[True, False], width=320
        )

        self.control_fit_method = pn.widgets.Select(
            name="Control Channel Fitting Method",
            options=["IRWLS", "OLS"],
            value="IRWLS",
            width=320,
        )

        self.control_fit_window_mode = pn.widgets.Select(
            name="Control Fit Window",
            options=["full trace", "baseline epoch"],
            value="full trace",
            width=320,
        )
        self.control_fit_window_strt = pn.widgets.IntInput(
            name="Control Fit Window Start Time (s) (int)", value=0, width=320
        )
        self.control_fit_window_end = pn.widgets.IntInput(
            name="Control Fit Window End Time (s) (int)", value=0, width=320
        )

        self.photobleaching_detrend = pn.widgets.Select(
            name="Photobleaching Detrend? (bool)", value=False, options=[True, False], width=320
        )

        self.numberOfCores = pn.widgets.IntInput(name="# of cores (int)", value=2, width=150)

        self.combine_data = pn.widgets.Select(
            name="Combine Data? (bool)", value=False, options=[True, False], width=150
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
            file_pattern="*_output_*",
            name="Existing runs (steps 2–5)",
            width=950,
        )

        self.computePsth = pn.widgets.Select(
            name="z_score and/or \u0394F/F? (psth)", options=["z_score", "dff", "Both"], width=320
        )

        self.transients = pn.widgets.Select(
            name="z_score and/or \u0394F/F? (transients)", options=["z_score", "dff", "Both"], width=320
        )

        self.moving_wd = pn.widgets.IntInput(
            name="Moving Window for transients detection (s) (int)", value=15, width=320
        )

        self.highAmpFilt = pn.widgets.IntInput(name="HAFT (int)", value=2, width=150)

        self.transientsThresh = pn.widgets.IntInput(name="TD Thresh (int)", value=3, width=150)

        self.computeBinnedMetrics = pn.widgets.Select(
            name="Compute Binned Metrics? (bool)", options=[True, False], value=False, width=200
        )

        self.binnedMetricsWidth = pn.widgets.IntInput(name="Bin Width (s) (int)", value=120, width=150)

        self.moving_avg_filter = pn.widgets.IntInput(
            name="Window for Moving Average filter (int)", value=100, width=320
        )

        self.no_channels_np = pn.widgets.IntInput(
            name="Number of channels (Neurophotometrics only)", value=2, width=320
        )

        self.z_score_computation = pn.widgets.Select(
            name="z-score computation Method",
            options=["standard z-score", "baseline z-score", "modified z-score"],
            value="standard z-score",
            width=200,
        )

        self.baseline_wd_strt = pn.widgets.IntInput(name="Baseline Window Start Time (s) (int)", value=0, width=200)
        self.baseline_wd_end = pn.widgets.IntInput(name="Baseline Window End Time (s) (int)", value=0, width=200)

        self.explain_z_score = pn.pane.Markdown(
            """
                        ***Note :***<br>
                        - Details about z-score computation methods are explained in Github wiki.<br>
                        - The details will make user understand what computation method to use for
                        their data.<br>
                        - **Baseline Window Parameters** are only used with the *baseline z-score*
                        method; keep both at 0 for other methods.<br>
                        - Both values are in **seconds** and must be within the signal's recorded
                        timespan. **Start** must be strictly less than **End**
                        (e.g. Start=0, End=60 for a 0–60 s baseline window).<br>
                        - If either value falls outside the available signal timespan you will
                        receive an error indicating the offending parameter, the value supplied,
                        and the valid range (e.g.
                        "baselineWindowEnd=120 exceeds signal duration 90.5s;
                        signal timespan is [0, 90.5]s — choose values within this range.").
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
                        recorded from different recording sites.
                        - ***Use Transients as Events :*** Make this parameter ```True```, when user studies
                        spontaneous activity and has no external event TTLs. The transients detected in each
                        recording site are then used as that recording site's event timestamps for the PSTH
                        and peak/area computation.
                        """,
            width=580,
        )

        self.nSecPrev = pn.widgets.IntInput(name="Seconds before 0 (int)", value=-10, width=120)

        self.nSecPost = pn.widgets.IntInput(name="Seconds after 0 (int)", value=20, width=120)

        self.computeCorr = pn.widgets.Select(
            name="Compute Cross-correlation (bool)", options=[True, False], value=False, width=200
        )

        self.computePsthSignificance = pn.widgets.Select(
            name="Compute PSTH Significance? (bool)", options=[True, False], value=False, width=240
        )

        self.psthSignificanceAlpha = pn.widgets.FloatInput(
            name="Significance Level (alpha) (float)", value=0.05, step=0.01, width=220
        )

        self.psthBootstrapResamples = pn.widgets.IntInput(
            name="Bootstrap Resamples (int)", value=1000, step=100, width=200
        )

        self.useTransientsAsEvents = pn.widgets.Select(
            name="Use Transients as Events? (bool)", options=[True, False], value=False, width=200
        )

        self.timeInterval = pn.widgets.IntInput(name="Time Interval (s)", value=2, width=120)

        self.use_time_or_trials = pn.widgets.Select(
            name="Bin PSTH trials (str)", options=["Time (min)", "# of trials"], value="Time (min)", width=120
        )

        self.bin_psth_trials = pn.widgets.IntInput(
            name="Time(min) / # of trials \n for binning? (int)", value=0, width=200
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

        self.baselineCorrectionStart = pn.widgets.IntInput(
            name="Baseline Correction Start time(int)", value=-5, width=200
        )

        self.baselineCorrectionEnd = pn.widgets.IntInput(name="Baseline Correction End time(int)", value=0, width=200)

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
            pn.Row(self.useTransientsAsEvents),
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
                        - ***AUC Units :*** ```seconds``` reports the area in z-score (or ΔF/F) × seconds, the unit
                        commonly reported in the literature. ```samples``` integrates with one-sample spacing instead,
                        so the area also scales with the recording's sampling rate.
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

        self.auc_units = pn.widgets.Select(
            name="AUC Units (str)", options=["samples", "seconds"], value="samples", width=200
        )

        self.peak_param_wd = pn.WidgetBox(
            "### Peak and AUC Parameters", self.peak_explain, self.df_widget, self.auc_units, width=600
        )

        self.significance_explain = pn.pane.Markdown(
            """
                        - ***Significance Level (alpha) :*** The two-sided threshold the confidence
                        interval is computed at. Default is 0.05, i.e. a 95% interval.
                        - ***Bootstrap Resamples :*** How many times the trials are resampled to build
                        each interval. More resamples means less run-to-run variation and a longer
                        run. Default is 1000.
                        - Every event is tested against zero automatically. The table below names the
                        pairs of events to compare against each other, for example rewarded versus
                        unrewarded nose pokes. Leave it blank to run only the tests against zero.
                        - Each pair is compared within every recording site and metric, using the event
                        labels assigned in **Step 1: Label Stores**.
                        - Comparisons run inside one output folder. In a session run folder the trials
                        are resampled; in a group folder the session averages are.
                        """,
            width=580,
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

        self.significance_param_wd = pn.WidgetBox(
            "### PSTH Significance Parameters",
            self.significance_explain,
            self.computePsthSignificance,
            pn.Row(self.psthSignificanceAlpha, self.psthBootstrapResamples),
            self.comparison_df_widget,
            self.add_comparison_button,
            width=600,
        )

        self.individual_analysis_wd_2 = pn.Column(
            self.explain_time_artifacts,
            pn.Row(self.numberOfCores, self.combine_data),
            self.isosbestic_control,
            self.control_fit_method,
            self.control_fit_window_mode,
            self.control_fit_window_strt,
            self.control_fit_window_end,
            self.photobleaching_detrend,
            self.timeForLightsTurnOn,
            self.moving_avg_filter,
            self.computePsth,
            self.transients,
            self.moving_wd,
            pn.Row(self.highAmpFilt, self.transientsThresh),
            pn.Row(self.computeBinnedMetrics, self.binnedMetricsWidth),
            self.no_channels_np,
        )

        self.psth_baseline_param = pn.Column(
            self.zscore_param_wd,
            self.psth_param_wd,
            self.baseline_param_wd,
            self.peak_param_wd,
            self.significance_param_wd,
        )

        self.input_folder_selection_widget = pn.Column(
            pn.Row(pn.pane.Markdown("**Data Source:**"), self.source_mode),
            self.files_1,
            self.dandi_selector.panel,
        )
        self.input_folder_selection = pn.Card(
            self.input_folder_selection_widget,
            title="Input Folder Selection",
            styles=self.styles,
            width=1000,
        )

        self.output_folder_selection_widget = pn.Column(
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

        self.widget = pn.Column(
            self.mark_down_1,
            pn.Row(self.individual_analysis_wd_2, self.psth_baseline_param),
        )
        self.individual = pn.Card(
            self.widget, title="Individual Analysis", styles=self.styles, width=1000, collapsed=True
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

    def _collect_selected_runs(self) -> dict[str, list[str]]:
        """Group the FileSelector's selected output dirs by parent session.

        Raises
        ------
        ValueError
            If a group output directory was selected as an individual run.
        """
        grouped: dict[str, list[str]] = {}
        for path in self.outputs_selector.value or []:
            _reject_group_folder_selected_as_run(path=path)
            session = os.path.dirname(path)
            grouped.setdefault(session, []).append(parse_run_name(path))
        return grouped

    def validate_selected_runs_for_consumers(self) -> None:
        """Ensure every selected session that has output dirs on disk also has at least one selected.

        Run this from the click handlers for steps 2–5 (which consume existing
        output directories). Skips sessions with no ``_output_<run>`` subdirs
        yet — those are typically pre-step-1 states.
        """
        for session in self.files_1.value or []:
            _reject_group_folder_selected_as_run(path=session)
        grouped = self._collect_selected_runs()
        missing = [
            session
            for session in (self.files_1.value or [])
            if discover_run_folders(session) and not grouped.get(session)
        ]
        if missing:
            raise ValueError(
                f"No output directory selected for session(s) {missing!r}. "
                "Open the Output Folder Selection panel and pick at least one "
                "_output_<run> directory per selected session."
            )

    def _sessions_for_run_selection(self) -> list[str]:
        """Return the existing session directories whose runs steps 2-5 will read."""
        if self.source_mode.value == "dandi":
            candidates = self._prospective_dandi_sessions()
        else:
            candidates = list(self.files_1.value or [])
        return [session for session in candidates if os.path.isdir(session)]

    @staticmethod
    def _run_names_for_sessions(sessions: list[str]) -> list[str]:
        """Return every run name present in at least one of ``sessions``, in run-name order.

        A name need not exist in every session to be offered: applying it selects the runs
        that do exist and leaves the remaining sessions to the folder tree.
        """
        run_names = []
        for session in sessions:
            for directory in discover_run_folders(session):
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
        """Root the existing-runs FileSelector so all selected sessions' `_output_*` dirs are reachable.

        - Zero sessions: fall back to the form's starting directory.
        - One session: root and starting directory both set to that session so its `_output_*`
          children show directly (no extra click).
        - Multiple sessions: root set to their common parent so every session is navigable;
          starting directory set to the first session so the user lands on one session's
          outputs and can navigate up to switch between sessions.
        - DANDI mode: root set to the chosen output root, which holds every mirrored session.
        """
        dandi_output_root = self.dandi_selector.output_root if self.source_mode.value == "dandi" else None
        if dandi_output_root:
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

        selected = [path for path in (self.outputs_selector.value or []) if os.path.dirname(path) in sessions]
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
        self._apply_selected_run_folders([path for path in run_folders if os.path.dirname(path) in sessions])

    def _on_run_names_for_all_sessions_change(self, event: object) -> None:
        """Select or deselect the runs matching the bulk choice, leaving hand-picked ones alone."""
        if self._suppressing_run_name_propagation:
            return
        added = [run_name for run_name in event.new if run_name not in event.old]
        removed = [run_name for run_name in event.old if run_name not in event.new]
        kept = [path for path in (self.outputs_selector.value or []) if parse_run_name(path) not in removed]
        sessions = self._sessions_for_run_selection()
        self._apply_selected_run_folders(kept + self._run_folders_on_disk(sessions=sessions, run_names=added))

    @staticmethod
    def _run_folders_on_disk(*, sessions: list[str], run_names: list[str]) -> list[str]:
        """Return the existing run directories named by ``run_names`` across ``sessions``."""
        run_folders = []
        for session in sessions:
            for run_name in run_names:
                run_folder = run_folder_for_run(session, run_name)
                if os.path.isdir(run_folder):
                    run_folders.append(run_folder)
        return run_folders

    def refresh_individual_outputs(self) -> None:
        """Re-list the outputs FileSelector so newly-created run dirs (e.g. from step 1) appear."""
        self.outputs_selector._refresh()
        self._refresh_run_name_options(self._sessions_for_run_selection())

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
            session_stem = os.path.splitext(os.path.basename(asset_path))[0]
            sessions.append(os.path.join(output_root, session_stem))
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
            os.makedirs(session_directory, exist_ok=True)
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

    def setup_visualization_parameters(self) -> None:
        """Build all widgets for the visualization-parameters card and store them as instance attributes."""
        self.visualize_zscore_or_dff = pn.widgets.Select(
            name="z-score or \u0394F/F? (for visualization)", options=["z_score", "dff"], width=435
        )

        self.visualization_wd = pn.Row(self.visualize_zscore_or_dff)
        self.visualize = pn.Card(
            self.visualization_wd, title="Visualization Parameters", styles=self.styles, width=1000, collapsed=True
        )

    def add_to_template(self) -> None:
        """Append the input/output folder, individual, group, and visualization cards to the template's main area."""
        self.template.main.append(self.input_folder_selection)
        self.template.main.append(self.output_folder_selection)
        self.template.main.append(self.individual)
        self.template.main.append(self.group)
        self.template.main.append(self.visualize)

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

        inputParameters = {
            "mode": mode,
            "dandi_uri_map": dandi_uri_map,
            "abspath": abspath_value,
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
            "noChannels": self.no_channels_np.value,
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
            "visualize_zscore_or_dff": self.visualize_zscore_or_dff.value,
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
            "noChannels": self.no_channels_np,
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
            "visualize_zscore_or_dff": self.visualize_zscore_or_dff,
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
            json_path = os.path.join(run_folder, "GuPPyParamtersUsed.json")
            if os.path.exists(json_path):
                with open(json_path) as parameters_file:
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

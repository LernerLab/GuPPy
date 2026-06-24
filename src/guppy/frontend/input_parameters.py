import json
import logging
import os

import numpy as np
import pandas as pd
import panel as pn

from .dandi_selector import DandiSelector
from .frontend_utils import default_root_path
from ..utils.utils import discover_output_dirs, parse_run_name
from ..utils.validation import (
    validate_required_folder_selection,
    validate_same_parent_directory,
)

logger = logging.getLogger(__name__)


def checkSameLocation(arr: list[str], abspath: object) -> str:
    """Check that all paths in ``arr`` share the same parent directory.

    Parameters
    ----------
    arr : sequence of str
        Paths to validate.
    abspath : object
        Ignored; retained for backwards-compatibility with existing callers.

    Returns
    -------
    str
        The common parent directory of all paths in ``arr``.
    """
    # abspath retained as a positional arg for backwards compatibility with existing
    # callers; only the contents of arr are inspected.
    del abspath
    return validate_same_parent_directory(paths=list(arr))


def getAbsPath(files_1: pn.widgets.FileSelector, files_2: pn.widgets.FileSelector) -> str:
    """Return the common parent directory of the selected folders.

    Parameters
    ----------
    files_1 : pn.widgets.FileSelector
        Primary file selector (individual analysis).
    files_2 : pn.widgets.FileSelector
        Secondary file selector (group analysis).

    Returns
    -------
    str
        Absolute path of the common parent directory shared by the selected
        folders.
    """
    validate_required_folder_selection(file_selectors=[files_1, files_2])
    selected = files_1.value if len(files_1.value) > 0 else files_2.value
    return validate_same_parent_directory(paths=list(selected))


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
        self.group_selected_outputs_widgets: dict[str, pn.widgets.Select] = {}
        self.setup_individual_parameters()
        self.setup_group_parameters()
        self.setup_visualization_parameters()
        self.add_to_template()
        self.files_1.param.watch(self._retarget_outputs_selector, "value")
        self.files_2.param.watch(self._rebuild_group_selected_outputs_widgets, "value")
        self.outputs_selector.param.watch(self._load_parameters_from_selected_outputs, "value")

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

        self.files_1 = pn.widgets.FileSelector(self.folder_path, root_directory="/", name="folderNames", width=950)

        self.dandi_selector = DandiSelector(styles=self.styles)
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

        self.numberOfCores = pn.widgets.IntInput(name="# of cores (int)", value=2, width=150)

        self.combine_data = pn.widgets.Select(
            name="Combine Data? (bool)", value=False, options=[True, False], width=150
        )

        self.outputs_selector_header = pn.pane.Markdown(
            "**Existing runs (steps 2–5):** Pick at least one existing output directory per "
            "selected session. To create a new run, use the Storenames GUI in step 1.",
            width=950,
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

        self.plot_zScore_dff = pn.widgets.Select(
            name="z-score plot and/or \u0394F/F plot?",
            options=["z_score", "dff", "Both", "None"],
            value="None",
            width=320,
        )

        self.moving_wd = pn.widgets.IntInput(
            name="Moving Window for transients detection (s) (int)", value=15, width=320
        )

        self.highAmpFilt = pn.widgets.IntInput(name="HAFT (int)", value=2, width=150)

        self.transientsThresh = pn.widgets.IntInput(name="TD Thresh (int)", value=3, width=150)

        self.moving_avg_filter = pn.widgets.IntInput(
            name="Window for Moving Average filter (int)", value=100, width=320
        )

        self.removeArtifacts = pn.widgets.Select(
            name="removeArtifacts? (bool)", value=False, options=[True, False], width=150
        )

        self.artifactsRemovalMethod = pn.widgets.Select(
            name="removeArtifacts method", value="concatenate", options=["concatenate", "replace with NaN"], width=150
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
                        recorded from different brain regions.
                        """,
            width=580,
        )

        self.nSecPrev = pn.widgets.IntInput(name="Seconds before 0 (int)", value=-10, width=120)

        self.nSecPost = pn.widgets.IntInput(name="Seconds after 0 (int)", value=20, width=120)

        self.computeCorr = pn.widgets.Select(
            name="Compute Cross-correlation (bool)", options=[True, False], value=False, width=200
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
            self.control_fit_method,
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

    def _collect_selected_outputs(self) -> dict[str, list[str]]:
        """Group the FileSelector's selected output dirs by parent session."""
        grouped: dict[str, list[str]] = {}
        for path in self.outputs_selector.value or []:
            session = os.path.dirname(path)
            grouped.setdefault(session, []).append(parse_run_name(path))
        return grouped

    def validate_selected_outputs_for_consumers(self) -> None:
        """Ensure every selected session that has output dirs on disk also has at least one selected.

        Run this from the click handlers for steps 2–5 (which consume existing
        output directories). Skips sessions with no ``_output_<run>`` subdirs
        yet — those are typically pre-step-1 states.
        """
        grouped = self._collect_selected_outputs()
        missing = [
            session
            for session in (self.files_1.value or [])
            if discover_output_dirs(session) and not grouped.get(session)
        ]
        if missing:
            raise ValueError(
                f"No output directory selected for session(s) {missing!r}. "
                "Open the Output Folder Selection panel and pick at least one "
                "_output_<run> directory per selected session."
            )

    def _retarget_outputs_selector(self, event: object) -> None:
        """Root the existing-runs FileSelector so all selected sessions' `_output_*` dirs are reachable.

        - Zero sessions: fall back to ``default_root_path()``.
        - One session: root and starting directory both set to that session so its `_output_*`
          children show directly (no extra click).
        - Multiple sessions: root set to their common parent so every session is navigable;
          starting directory set to the first session so the user lands on one session's
          outputs and can navigate up to switch between sessions.
        """
        sessions = [s for s in (event.new or []) if os.path.isdir(s)]
        if not sessions:
            root_target = default_root_path()
            directory_target = default_root_path()
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
        # Clear any prior selection that no longer makes sense for the new root.
        self.outputs_selector.value = []
        # Sync the FileSelector's internal _cwd and re-enumerate. Without this, _cwd remains
        # at the construction-time path; clicking a sub-dir uses the stale _cwd to compute
        # the navigated path, that path doesn't exist, and the FileSelector silently snaps
        # back to the stale _cwd — visible to the user as "selection resets the directory".
        self.outputs_selector._update_files()

    def _rebuild_group_selected_outputs_widgets(self, event: object) -> None:
        """Rebuild the per-session group-run-name Selects when files_2 changes."""
        self._rebuild_per_session_widgets(
            sessions=event.new,
            target_box=self.group_selected_outputs_box,
            store=self.group_selected_outputs_widgets,
            scope="group",
        )

    def refresh_individual_outputs(self) -> None:
        """Re-list the outputs FileSelector so newly-created run dirs (e.g. from step 1) appear."""
        self.outputs_selector._refresh()

    def refresh_group_outputs(self) -> None:
        """Re-discover output directories for the currently-selected group sessions."""
        self._rebuild_per_session_widgets(
            sessions=self.files_2.value,
            target_box=self.group_selected_outputs_box,
            store=self.group_selected_outputs_widgets,
            scope="group",
        )

    @staticmethod
    def _make_outputs_placeholder(scope: str) -> pn.pane.Markdown:
        text = (
            "**Run-name filter:** No output directories yet — run step 1 first."
            if scope == "individual"
            else "**Run-name filter (group):** No output directories yet — run step 1 first."
        )
        return pn.pane.Markdown(text, width=520)

    @classmethod
    def _rebuild_per_session_widgets(
        cls, sessions: list[str] | None, target_box: pn.Column, store: dict[str, pn.widgets.Select], scope: str
    ) -> None:
        new_objects = []
        new_store = {}
        for session in sessions or []:
            run_names = [parse_run_name(directory) for directory in discover_output_dirs(session)]
            # Skip sessions with no output dirs — nothing to filter, no widget needed.
            if not run_names:
                continue
            existing = store.get(session)
            if existing is not None:
                # Preserve the user's prior selection across rebuilds when it remains valid.
                preserved = existing.value if existing.value in run_names else run_names[0]
                existing.options = run_names
                existing.value = preserved
                widget = existing
            else:
                widget = pn.widgets.Select(
                    name=f"Outputs for {os.path.basename(session)}",
                    value=run_names[0],
                    options=run_names,
                    width=320,
                )
            new_store[session] = widget
            new_objects.append(widget)
        store.clear()
        store.update(new_store)
        # When no per-session widgets are populated, show the placeholder so the
        # box has a stable, always-visible footprint in the layout.
        target_box.objects = new_objects or [cls._make_outputs_placeholder(scope)]

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

        folder_names = []
        dandi_uri_map = {}
        for uri in selected_uris:
            asset_path = uri.split("/", 3)[-1]
            session_stem = os.path.splitext(os.path.basename(asset_path))[0]
            session_directory = os.path.join(output_root, session_stem)
            os.makedirs(session_directory, exist_ok=True)
            folder_names.append(session_directory)
            dandi_uri_map[session_directory] = uri
        return folder_names, output_root, dandi_uri_map

    def setup_group_parameters(self) -> None:
        """Build all widgets for the group-analysis card and store them as instance attributes."""
        self.mark_down_2 = pn.pane.Markdown(
            """**Select folders for the average analysis from the file selector below**""", width=600
        )

        self.files_2 = pn.widgets.FileSelector(
            self.folder_path, root_directory="/", name="folderNamesForAvg", width=950
        )

        self.averageForGroup = pn.widgets.Select(
            name="Average Group? (bool)", value=False, options=[True, False], width=435
        )

        self.group_selected_outputs_box = pn.Column(self._make_outputs_placeholder("group"))

        self.group_analysis_wd_1 = pn.Column(
            self.mark_down_2, self.files_2, self.group_selected_outputs_box, self.averageForGroup, width=800
        )
        self.group = pn.Card(
            self.group_analysis_wd_1, title="Group Analysis", styles=self.styles, width=1000, collapsed=True
        )

    def setup_visualization_parameters(self) -> None:
        """Build all widgets for the visualization-parameters card and store them as instance attributes."""
        self.visualizeAverageResults = pn.widgets.Select(
            name="Visualize Average Results? (bool)", value=False, options=[True, False], width=435
        )

        self.visualize_zscore_or_dff = pn.widgets.Select(
            name="z-score or \u0394F/F? (for visualization)", options=["z_score", "dff"], width=435
        )

        self.visualization_wd = pn.Row(self.visualize_zscore_or_dff, pn.Spacer(width=60), self.visualizeAverageResults)
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

    def getInputParameters(self) -> dict[str, object]:
        """Collect and return all current widget values as an input-parameters dictionary.

        Returns
        -------
        dict
            Flat dictionary containing every parameter needed to run the GuPPy
            pipeline, keyed by the parameter names expected by the orchestration
            layer (e.g. ``"folderNames"``, ``"zscore_method"``, ``"nSecPrev"``).
        """
        # Re-discover group output dirs so the per-session filters reflect any new dirs
        # produced by step 1 since the user last deselected/reselected their session folder.
        self.refresh_group_outputs()

        if self.source_mode.value == "dandi":
            folder_names, abspath_value, dandi_uri_map = self._resolve_dandi_sessions()
            mode = "dandi"
        else:
            abspath = getAbsPath(self.files_1, self.files_2)
            folder_names = self.files_1.value
            abspath_value = abspath[0]
            dandi_uri_map = None
            mode = "local"

        inputParameters = {
            "mode": mode,
            "dandi_uri_map": dandi_uri_map,
            "abspath": abspath_value,
            "folderNames": folder_names,
            "numberOfCores": self.numberOfCores.value,
            "combine_data": self.combine_data.value,
            "isosbestic_control": self.isosbestic_control.value,
            "control_fit_method": self.control_fit_method.value,
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
            "selectedOutputs": self._collect_selected_outputs(),
            "groupSelectedOutputs": {
                session: [widget.value] for session, widget in self.group_selected_outputs_widgets.items()
            },
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
            "timeForLightsTurnOn": self.timeForLightsTurnOn,
            "filter_window": self.moving_avg_filter,
            "removeArtifacts": self.removeArtifacts,
            "artifactsRemovalMethod": self.artifactsRemovalMethod,
            "noChannels": self.no_channels_np,
            "zscore_method": self.z_score_computation,
            "baselineWindowStart": self.baseline_wd_strt,
            "baselineWindowEnd": self.baseline_wd_end,
            "nSecPrev": self.nSecPrev,
            "nSecPost": self.nSecPost,
            "computeCorr": self.computeCorr,
            "timeInterval": self.timeInterval,
            "bin_psth_trials": self.bin_psth_trials,
            "use_time_or_trials": self.use_time_or_trials,
            "baselineCorrectionStart": self.baselineCorrectionStart,
            "baselineCorrectionEnd": self.baselineCorrectionEnd,
            "selectForComputePsth": self.computePsth,
            "selectForTransientsComputation": self.transients,
            "moving_window": self.moving_wd,
            "highAmpFilt": self.highAmpFilt,
            "transientsThresh": self.transientsThresh,
            "plot_zScore_dff": self.plot_zScore_dff,
            "visualize_zscore_or_dff": self.visualize_zscore_or_dff,
            "averageForGroup": self.averageForGroup,
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

    def _load_parameters_from_selected_outputs(self, event: object) -> None:
        """Reload analysis parameters from the saved JSON of the selected output run(s).

        Fired when the individual-analysis output selector changes. Lets a user
        resume a run (e.g. relaunch and run steps 3–4) without the form's
        defaults silently overwriting the parameters the earlier steps used.
        When several runs are selected the parameters are applied only if every
        run with a saved snapshot agrees; conflicting snapshots are left for the
        user to reconcile.
        """
        saved = []
        for output_dir in event.new or []:
            json_path = os.path.join(output_dir, "GuPPyParamtersUsed.json")
            if os.path.exists(json_path):
                with open(json_path) as f:
                    saved.append(json.load(f))
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

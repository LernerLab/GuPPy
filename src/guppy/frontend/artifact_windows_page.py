"""Panel page for the optional Select Artifact Windows step.

The user marks the periods where artifacts occurred on a run's control/signal/fit
traces. Saving writes the complementary keep-windows to
``coordsForPreProcessing_<recording_site>.npy`` and records the removal method into the
run's ``GuPPyParamtersUsed.json``, which the Remove Artifacts step then consumes.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import panel as pn

from .artifact_removal import build_run_folder_page, load_pair_traces
from ..analysis.io_utils import fetchCoords
from ..orchestration.save_parameters import record_artifact_provenance
from ..utils.artifact_windows import (
    complement_windows,
    coords_to_windows,
    merge_windows,
    windows_to_coords,
)
from ..utils.utils import discover_run_folders, parse_run_name
from ..visualization.preprocessing import (
    build_markable_trace,
    make_spans_pipe,
    set_drag_gesture,
)

pn.extension(notifications=True)

logger = logging.getLogger(__name__)

ARTIFACT_REMOVAL_METHODS = ["replace with NaN", "concatenate"]

MARK_MODE = "Mark artifacts"
NAVIGATE_MODE = "Navigate"
INTERACTION_MODES = [MARK_MODE, NAVIGATE_MODE]

# Which of a recording site's three traces the plot shows, in selector order.
CONTROL_TRACE, SIGNAL_TRACE, FIT_TRACE = "control", "signal", "fit"

# Seconds per arrow-key press on a window bound, for nudging an edge into place.
_NUDGE_STEP = 0.1
_INPUT_WIDTH = 110

_INSTRUCTIONS = (
    "Mark the periods where **artifacts occurred** for each recording site: drag "
    "horizontally across the trace to mark a period, or type its start and end time "
    "(seconds) into a row. Either way the period lands in an editable row below the plot "
    "and the shaded spans update to match, so a bound dragged roughly into place can be "
    "nudged the rest of the way with the arrow keys. Everything outside the marked "
    "periods is kept — mark nothing to keep the entire recording.\n\n"
    "Switch to **Navigate** to drag the view around instead of marking; the scroll wheel "
    "zooms in either mode, and switching back to **Mark artifacts** keeps wherever you "
    "have zoomed to. The **Trace** selector chooses which of the site's three traces you "
    "mark on — the periods belong to the recording site, so it makes no difference which "
    "one you mark from.\n\n"
    "A period that reaches the start or the end of the trace trims that end of the "
    "recording outright. This is how a single session gets more taken off its opening "
    "than *Eliminate first few seconds*, which trims every session in the batch alike.\n\n"
    "Click **Save** to write the windows, then run **Remove Artifacts** to apply them."
)

_COPY_HELP = (
    "Artifact windows belong to the recording, not to the parameters, so another run of "
    "this session can be marked once and reused here."
)

_NO_WINDOWS_HINT = "_No artifact periods marked — the entire recording will be kept._"

_METHOD_HELP = (
    "How the marked periods are applied. **replace with NaN** (recommended) blanks them "
    "while preserving the recording timeline. **concatenate** splices the kept segments "
    "together; this re-times the kept samples onto a new timeline, breaks alignment to the "
    "acquisition clock, is unsupported by NWB export, and is incompatible with "
    "cross-correlation."
)


def _prepared_windows(
    windows: list[tuple[float, float]], timestamps: np.ndarray
) -> tuple[list[tuple[float, float]], list[str]]:
    """Pull marked artifact windows into the recording, reporting the bounds that moved.

    A bound at or beyond the first or last sample lands on the margined span, so the
    complement holds no leading or trailing chunk instead of one too short to survive the
    moving-average filter. A bound further out than that lands there too.

    Returns
    -------
    prepared : list of (float, float)
        The windows as they will be saved.
    moved : list of str
        One clause per bound that had to be pulled in, for reporting back to the user;
        empty when every bound was already inside the recording.

    Raises
    ------
    ValueError
        If a window is inverted, or lies entirely outside the recording.
    """
    low, high = float(timestamps[0]), float(timestamps[-1])
    span_start, span_end = _margined_span(timestamps)
    prepared, moved = [], []
    for start, end in windows:
        if start >= end:
            message = f"Artifact window start={start} must be less than end={end}."
            logger.error(message)
            raise ValueError(message)
        if end <= span_start or start >= span_end:
            message = (
                f"Artifact window [{start}, {end}] lies entirely outside the recording timespan "
                f"[{low}, {high}]; choose values within this range."
            )
            logger.error(message)
            raise ValueError(message)
        bounds = (span_start if start <= low else start, span_end if end >= high else end)
        if bounds[0] != start:
            moved.append(
                f"the window starting at {start:g} s began before the recording, which starts at "
                f"{low:g} s, so its start was moved to the start of the recording"
            )
        if bounds[1] != end:
            moved.append(
                f"the window ending at {end:g} s ran past the recording, which ends at {high:g} s, "
                "so its end was moved to the end of the recording"
            )
        prepared.append(bounds)
    return prepared, moved


def _margined_span(timestamps: np.ndarray) -> tuple[float, float]:
    """Return the keep-window span, widened by one sample so the endpoints survive."""
    dt = float(timestamps[1] - timestamps[0])
    return float(timestamps[0]) - dt, float(timestamps[-1]) + dt


def _has_saved_windows(filepath: str, site: str) -> bool:
    """Whether a run folder holds a keep-windows file for one recording site."""
    return (Path(filepath) / f"coordsForPreProcessing_{site}.npy").exists()


def _sites_with_saved_windows(filepath: str) -> list[str]:
    """The recording sites a run folder holds keep-windows for, in filename order."""
    prefix = "coordsForPreProcessing_"
    paths = Path(filepath).glob(prefix + "*.npy")
    return sorted(path.name[len(prefix) : -len(".npy")] for path in paths)


def _saved_artifact_windows(filepath: str, site: str, timestamps: np.ndarray) -> list[tuple[float, float]]:
    """Re-derive the marked artifact windows from the keep-windows saved on disk."""
    if not _has_saved_windows(filepath, site):
        return []
    keep_windows = coords_to_windows(coords=fetchCoords(filepath, site, timestamps))
    span_start, span_end = _margined_span(timestamps)
    return complement_windows(windows=keep_windows, span_start=span_start, span_end=span_end)


def _runs_with_saved_windows(filepath: str) -> dict[str, str]:
    """Map run name → path for the other runs of this session that have windows saved.

    Parameters
    ----------
    filepath : str
        The run folder being marked; its own windows are never offered back.

    Returns
    -------
    dict
        Run name → run folder, ordered as the session lists its runs.
    """
    return {
        parse_run_name(run_folder): run_folder
        for run_folder in discover_run_folders(str(Path(filepath).parent))
        if Path(run_folder).resolve() != Path(filepath).resolve()
        and any(Path(run_folder).glob("coordsForPreProcessing_*.npy"))
    }


class ArtifactWindowRow:
    """One marked artifact period: a start bound, an end bound, and a delete button.

    The bounds are numeric inputs limited to the recording timespan, so an edge can be
    nudged with the arrow keys. The limits are a browser-side guardrail only — a value
    set programmatically is not clamped, so callers still validate before saving.
    """

    def __init__(
        self,
        *,
        start: float | None,
        end: float | None,
        span: tuple[float, float],
        on_change: Callable[[], None],
        on_remove: Callable[["ArtifactWindowRow"], None],
    ) -> None:
        low, high = span
        self.start_input = pn.widgets.FloatInput(value=start, start=low, end=high, step=_NUDGE_STEP, width=_INPUT_WIDTH)
        self.end_input = pn.widgets.FloatInput(value=end, start=low, end=high, step=_NUDGE_STEP, width=_INPUT_WIDTH)
        self.remove_button = pn.widgets.Button(
            icon="trash", button_type="light", width=40, height=38, description="Delete this period"
        )

        self.start_input.param.watch(lambda event: on_change(), "value")
        self.end_input.param.watch(lambda event: on_change(), "value")
        self.remove_button.on_click(lambda event: on_remove(self))

        self.widget = pn.Row(self.start_input, self.end_input, self.remove_button, align="center", margin=(0, 0, 4, 0))

    @property
    def window(self) -> tuple[float, float] | None:
        """The ``(start, end)`` period, or None while either bound is still blank."""
        if self.start_input.value is None or self.end_input.value is None:
            return None
        return float(self.start_input.value), float(self.end_input.value)


class ArtifactWindowSelector:
    """Interactive artifact-marking page for one run folder across all recording sites.

    Shows the selected site's control/signal/fit traces with the marked artifact periods
    shaded, one editable row per period, and a removal-method selector. On save, writes
    one ``coordsForPreProcessing_<site>.npy`` per site holding the keep-windows.
    """

    def __init__(self, filepath: str, pair_traces: dict[str, dict[str, object]]) -> None:
        self.filepath = filepath
        self.pair_traces = pair_traces
        self.sites = list(pair_traces.keys())

        self.site_to_rows: dict[str, list[ArtifactWindowRow]] = {
            site: [
                self._build_row(site, start, end)
                for start, end in _saved_artifact_windows(filepath, site, pair_traces[site]["x"])
            ]
            for site in self.sites
        }

        self.site_select = pn.widgets.Select(name="Recording site", options=self.sites, value=self.sites[0])
        self.trace_select = pn.widgets.Select(
            name="Trace", options=self._trace_options(self.sites[0]), value=SIGNAL_TRACE, width=200
        )
        self.mode_toggle = pn.widgets.RadioButtonGroup(
            name="Drag gesture",
            options=INTERACTION_MODES,
            value=MARK_MODE,
            button_type="primary",
            button_style="outline",
            width=240,
            align="end",
        )
        self.runs_with_windows = _runs_with_saved_windows(filepath)
        self.copy_from_select = pn.widgets.Select(
            name="Copy windows from run", options=list(self.runs_with_windows), width=200
        )
        self.copy_from_button = pn.widgets.Button(name="Load", button_type="default", icon="copy", align="end")
        self.add_row_button = pn.widgets.Button(name="Add period", button_type="default", icon="plus")
        self.apply_to_all_button = pn.widgets.Button(name="Apply to all recording sites", button_type="default")
        self.method_select = pn.widgets.Select(
            name="Removal method", options=ARTIFACT_REMOVAL_METHODS, value=ARTIFACT_REMOVAL_METHODS[0], width=200
        )
        self.save_button = pn.widgets.Button(name="Save", button_type="primary")

        self.spans_pipe = make_spans_pipe(windows=self.windows_for(self.sites[0]))
        self.figure = None
        self.marking_pane = pn.pane.HoloViews(self._make_marking_plot(), sizing_mode="stretch_width")
        self.rows_container = pn.Column()
        self._render_rows()

        self.site_select.param.watch(self._on_site_change, "value")
        self.trace_select.param.watch(lambda event: self._rebuild_marking_plot(), "value")
        self.mode_toggle.param.watch(lambda event: self._apply_mode(), "value")
        self.copy_from_button.on_click(self._on_copy)
        self.add_row_button.on_click(lambda event: self.add_window_row())
        self.apply_to_all_button.on_click(lambda event: self.apply_windows_to_all_sites())
        self.save_button.on_click(self._on_save)

        copy_from_section = (
            [pn.pane.Markdown(_COPY_HELP), pn.Row(self.copy_from_select, self.copy_from_button)]
            if self.runs_with_windows
            else []
        )
        self.widget = pn.Column(
            f"# Select Artifact Windows — {Path(filepath).name}",
            pn.pane.Markdown(_INSTRUCTIONS),
            *copy_from_section,
            pn.Row(self.site_select, self.trace_select, self.mode_toggle),
            self.marking_pane,
            self.rows_container,
            pn.Row(self.add_row_button, self.apply_to_all_button),
            self.method_select,
            pn.pane.Markdown(_METHOD_HELP),
            self.save_button,
            sizing_mode="stretch_width",
        )

    def _build_row(self, site: str, start: float | None, end: float | None) -> ArtifactWindowRow:
        return ArtifactWindowRow(
            start=start,
            end=end,
            span=_margined_span(self.pair_traces[site]["x"]),
            on_change=self.refresh_spans,
            on_remove=self._remove_row,
        )

    def _column_header(self) -> pn.Row:
        return pn.Row(
            pn.pane.HTML("<b>start (s)</b>", width=_INPUT_WIDTH),
            pn.pane.HTML("<b>end (s)</b>", width=_INPUT_WIDTH),
            pn.Spacer(width=40),
            margin=(0, 0, 2, 0),
        )

    def _render_rows(self) -> None:
        rows = self.site_to_rows[self.site_select.value]
        if not rows:
            self.rows_container[:] = [pn.pane.Markdown(_NO_WINDOWS_HINT)]
            return
        self.rows_container[:] = [self._column_header()] + [row.widget for row in rows]

    def windows_for(self, site: str) -> list[tuple[float, float]]:
        """The fully-entered artifact periods marked for one recording site."""
        return [row.window for row in self.site_to_rows[site] if row.window is not None]

    def set_windows(self, site: str, windows: list[tuple[float, float]]) -> None:
        """Replace one recording site's marked periods."""
        self.site_to_rows[site] = [self._build_row(site, start, end) for start, end in windows]
        if site == self.site_select.value:
            self._render_rows()
        self.refresh_spans()

    def refresh_spans(self) -> None:
        """Repaint the shaded spans from the current rows, leaving the traces untouched."""
        self.spans_pipe.send(self.windows_for(self.site_select.value))

    def add_window_row(self, start: float | None = None, end: float | None = None) -> None:
        """Append a period to the selected site, blank unless bounds are given."""
        site = self.site_select.value
        self.site_to_rows[site].append(self._build_row(site, start, end))
        self._render_rows()
        self.refresh_spans()

    def mark_window_from_drag(self, start: float, end: float) -> str | None:
        """Add the period a horizontal drag across the trace covers.

        The bounds are clamped to the recording and rounded to the millisecond, leaving
        the row to carry whatever precision the drag actually needs. A bound that is
        clamped lands on the span exactly rather than on its rounding, so the period is
        already at the edge by the time it is saved. A drag too short to cover any time —
        a stray click — marks nothing.

        Returns
        -------
        str or None
            Message naming the bounds that were clamped, or None when the drag stayed
            inside the recording and nothing had to move.
        """
        timestamps = self.pair_traces[self.site_select.value]["x"]
        first, last = float(timestamps[0]), float(timestamps[-1])
        span_start, span_end = _margined_span(timestamps)
        low, high = sorted((float(start), float(end)))
        clamped_low = span_start if low <= span_start else round(low, 3)
        clamped_high = span_end if high >= span_end else round(high, 3)
        if clamped_high <= clamped_low:
            return None
        self.add_window_row(clamped_low, clamped_high)

        notices = []
        if clamped_low != round(low, 3):
            notices.append(
                f"The drag began before the recording, which starts at {first:g} s, so the period "
                "was marked from the start of the recording."
            )
        if clamped_high != round(high, 3):
            notices.append(
                f"The drag ran past the recording, which ends at {last:g} s, so the period was "
                "marked up to the end of the recording."
            )
        return " ".join(notices) if notices else None

    def copy_windows_from_run(self, run_name: str) -> str:
        """Replace this run's periods with those saved for another run of this session.

        Recording sites are matched by name, so only the sites the other run saved windows
        for are replaced; the rest keep whatever is currently marked.

        The windows are loaded into the editable rows rather than written straight to
        disk, so they can be reviewed and adjusted before Save.

        Returns
        -------
        str
            Message naming the sites that were loaded, for reporting back to the user.

        Raises
        ------
        ValueError
            If the other run saved windows for none of this run's recording sites.
        """
        run_folder = self.runs_with_windows[run_name]
        matched = [site for site in self.sites if _has_saved_windows(run_folder, site)]
        if not matched:
            message = (
                f"Run {run_name} has no artifact windows saved for {' or '.join(self.sites)} — it has "
                f"them for {', '.join(_sites_with_saved_windows(run_folder))}. Recording sites are matched "
                "by name, so a site labeled differently in Label Stores cannot be reused; relabel it to "
                "match, or mark this run by hand."
            )
            logger.error(message)
            raise ValueError(message)

        for site in matched:
            self.set_windows(site, _saved_artifact_windows(run_folder, site, self.pair_traces[site]["x"]))
        logger.info("Loaded artifact windows from run %s for %s.", run_name, ", ".join(matched))

        message = f"Loaded artifact windows from run {run_name} for {', '.join(matched)}."
        skipped = [site for site in self.sites if site not in matched]
        if skipped:
            message += f" Run {run_name} has none saved for {', '.join(skipped)}, left as marked."
        return message

    def remove_window_row(self, row_index: int) -> None:
        """Drop one period from the selected site."""
        self._remove_row(self.site_to_rows[self.site_select.value][row_index])

    def apply_windows_to_all_sites(self) -> None:
        """Copy the selected site's periods into every other recording site."""
        windows = self.windows_for(self.site_select.value)
        for site in self.sites:
            if site != self.site_select.value:
                self.set_windows(site, windows)

    def save(self) -> list[str]:
        """Write the keep-windows for every site, then record the removal method.

        All sites are prepared before anything is written, so an invalid period raises
        up-front without leaving a partially-written set of coords files. A site with no
        marked periods writes no file (keep-the-entire-recording default). A bound that
        had to be pulled into the recording is written back into its row, so the page
        shows what was saved.

        Returns
        -------
        list of str
            One message per recording site whose bounds had to be pulled in, for
            reporting back to the user.
        """
        site_to_prepared = {}
        site_to_keep_windows = {}
        clamped = []
        for site in self.sites:
            timestamps = self.pair_traces[site]["x"]
            artifact_windows, moved = _prepared_windows(self.windows_for(site), timestamps)
            site_to_prepared[site] = artifact_windows
            if moved:
                clamped.append(f"On recording site {site}, " + "; ".join(moved) + ".")
            if not artifact_windows:
                continue
            span_start, span_end = _margined_span(timestamps)
            site_to_keep_windows[site] = complement_windows(
                windows=merge_windows(windows=artifact_windows), span_start=span_start, span_end=span_end
            )

        for site, artifact_windows in site_to_prepared.items():
            if artifact_windows != self.windows_for(site):
                self.set_windows(site, artifact_windows)

        for site, keep_windows in site_to_keep_windows.items():
            np.save(
                Path(self.filepath) / (f"coordsForPreProcessing_{site}.npy"),
                windows_to_coords(windows=keep_windows),
            )
            logger.info("Saved %s keep-window(s) for recording site %s.", len(keep_windows), site)

        record_artifact_provenance(destination=self.filepath, artifacts_removal_method=self.method_select.value)
        for message in clamped:
            logger.warning(message)
        return clamped

    def _remove_row(self, row: ArtifactWindowRow) -> None:
        self.site_to_rows[self.site_select.value].remove(row)
        self._render_rows()
        self.refresh_spans()

    def _on_site_change(self, event: object) -> None:
        self._render_rows()
        self.trace_select.options = self._trace_options(self.site_select.value)
        self._rebuild_marking_plot()

    def _trace_options(self, site: str) -> dict[str, str]:
        """Label each of the site's three traces by the store it came from."""
        control_name, signal_name, fit_name = self.pair_traces[site]["plot_name"]
        return {control_name: CONTROL_TRACE, signal_name: SIGNAL_TRACE, fit_name: FIT_TRACE}

    def _rebuild_marking_plot(self) -> None:
        """Redraw the plot for the selected site and trace, on a fresh span stream.

        The plot is a new object, so the discarded one is given up along with its pipe.
        """
        self.spans_pipe = make_spans_pipe(windows=self.windows_for(self.site_select.value))
        self.figure = None
        self.marking_pane.object = self._make_marking_plot()

    def _make_marking_plot(self) -> object:
        trace = self.pair_traces[self.site_select.value]
        control_name, signal_name, fit_name = trace["plot_name"]
        choice = self.trace_select.value
        values, overlay, title = {
            CONTROL_TRACE: (trace["control"], None, control_name),
            SIGNAL_TRACE: (trace["signal"], None, signal_name),
            FIT_TRACE: (trace["signal"], trace["fit"], fit_name),
        }[choice]
        return build_markable_trace(
            x=trace["x"],
            values=values,
            overlay=overlay,
            title=f"{Path(self.filepath).name} — {title}",
            spans=self.spans_pipe,
            on_x_select=self._on_drag,
            hooks=[self._capture_figure],
        )

    def _capture_figure(self, plot: object, element: object) -> None:
        """Hold on to the rendered figure so the mode toggle can re-arm its drag tool."""
        self.figure = plot.state
        self._apply_mode()

    def _apply_mode(self) -> None:
        """Arm the plot's drag gesture for the selected mode, if it has been rendered."""
        if self.figure is not None:
            set_drag_gesture(figure=self.figure, marking=self.mode_toggle.value == MARK_MODE)

    def _on_drag(self, start: float, end: float) -> None:
        clamped = self.mark_window_from_drag(start, end)
        if clamped is not None:
            logger.warning(clamped)
            if pn.state.notifications is not None:
                pn.state.notifications.warning(clamped, duration=4000)

    def _on_copy(self, event: object) -> None:
        try:
            message = self.copy_windows_from_run(self.copy_from_select.value)
        except ValueError as error:
            if pn.state.notifications is not None:
                pn.state.notifications.error(str(error), duration=0)
            return
        if pn.state.notifications is not None:
            pn.state.notifications.success(message, duration=4000)

    def _on_save(self, event: object) -> None:
        try:
            clamped = self.save()
        except ValueError as error:
            logger.error(str(error))
            if pn.state.notifications is not None:
                pn.state.notifications.error(str(error), duration=0)
            return
        if pn.state.notifications is None:
            return
        if clamped:
            pn.state.notifications.warning("Artifact windows saved. " + " ".join(clamped), duration=0)
            return
        pn.state.notifications.success("Artifact windows saved.", duration=4000)


def build_artifact_window_page(*, run_folders: list[str]) -> pn.viewable.Viewable:
    """Compose the Select Artifact Windows page across all run folders.

    Parameters
    ----------
    run_folders : list of str
        Session output (run) directories to offer for marking.

    Returns
    -------
    pn.viewable.Viewable
        A single page; when there is more than one run folder a selector switches
        between them so only the selected folder's traces are rendered at a time.
    """
    return build_run_folder_page(
        run_folders=run_folders,
        build_folder_page=lambda filepath: ArtifactWindowSelector(filepath, load_pair_traces(filepath)).widget,
    )

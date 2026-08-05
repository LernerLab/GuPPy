"""Panel page for the optional Select Artifact Windows step.

The user marks the periods where artifacts occurred on a run's control/signal/fit
traces. Saving writes the complementary keep-windows to
``coordsForPreProcessing_<recording_site>.npy`` and records the removal method into the
run's ``GuPPyParamtersUsed.json``, which the Remove Artifacts step then consumes.
"""

import logging
import os
from collections.abc import Callable

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
from ..visualization.preprocessing import build_control_signal_fit, make_spans_pipe

pn.extension(notifications=True)

logger = logging.getLogger(__name__)

ARTIFACT_REMOVAL_METHODS = ["replace with NaN", "concatenate"]

# Seconds per arrow-key press on a window bound, for nudging an edge into place.
_NUDGE_STEP = 0.1
_INPUT_WIDTH = 110

_INSTRUCTIONS = (
    "Mark the periods where **artifacts occurred** for each recording site: enter the "
    "start and end time (seconds) of each period and the shaded spans update to match. "
    "Everything outside the marked periods is kept — mark nothing to keep the entire "
    "recording.\n\n"
    "Click **Save** to write the windows, then run **Remove Artifacts** to apply them."
)

_NO_WINDOWS_HINT = "_No artifact periods marked — the entire recording will be kept._"

_METHOD_HELP = (
    "How the marked periods are applied. **replace with NaN** (recommended) blanks them "
    "while preserving the recording timeline. **concatenate** splices the kept segments "
    "together; this re-times the kept samples onto a new timeline, breaks alignment to the "
    "acquisition clock, is unsupported by NWB export, and is incompatible with "
    "cross-correlation."
)


def _validated_windows(windows: list[tuple[float, float]], timestamps: np.ndarray) -> list[tuple[float, float]]:
    """Validate marked artifact windows against the recording timespan.

    Raises
    ------
    ValueError
        If a window is inverted or falls outside the recording.
    """
    low, high = float(timestamps[0]), float(timestamps[-1])
    for start, end in windows:
        if start >= end:
            message = f"Artifact window start={start} must be less than end={end}."
            logger.error(message)
            raise ValueError(message)
        if start < low or end > high:
            message = (
                f"Artifact window [{start}, {end}] is outside the recording timespan [{low}, {high}]; "
                "choose values within this range."
            )
            logger.error(message)
            raise ValueError(message)
    return list(windows)


def _margined_span(timestamps: np.ndarray) -> tuple[float, float]:
    """Return the keep-window span, widened by one sample so the endpoints survive."""
    dt = float(timestamps[1] - timestamps[0])
    return float(timestamps[0]) - dt, float(timestamps[-1]) + dt


def _saved_artifact_windows(filepath: str, site: str, timestamps: np.ndarray) -> list[tuple[float, float]]:
    """Re-derive the marked artifact windows from the keep-windows saved on disk."""
    if not os.path.exists(os.path.join(filepath, f"coordsForPreProcessing_{site}.npy")):
        return []
    keep_windows = coords_to_windows(coords=fetchCoords(filepath, site, timestamps))
    span_start, span_end = _margined_span(timestamps)
    return complement_windows(windows=keep_windows, span_start=span_start, span_end=span_end)


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
        self.add_row_button = pn.widgets.Button(name="Add period", button_type="default", icon="plus")
        self.apply_to_all_button = pn.widgets.Button(name="Apply to all recording sites", button_type="default")
        self.method_select = pn.widgets.Select(
            name="Removal method", options=ARTIFACT_REMOVAL_METHODS, value=ARTIFACT_REMOVAL_METHODS[0], width=200
        )
        self.save_button = pn.widgets.Button(name="Save", button_type="primary")

        self.spans_pipe = make_spans_pipe(windows=self.windows_for(self.sites[0]))
        self.marking_pane = pn.pane.HoloViews(self._make_marking_plot(), width=800)
        self.rows_container = pn.Column()
        self._render_rows()

        self.site_select.param.watch(self._on_site_change, "value")
        self.add_row_button.on_click(lambda event: self.add_window_row())
        self.apply_to_all_button.on_click(lambda event: self.apply_windows_to_all_sites())
        self.save_button.on_click(self._on_save)

        self.widget = pn.Column(
            "# Select Artifact Windows — {}".format(os.path.basename(filepath)),
            pn.pane.Markdown(_INSTRUCTIONS),
            self.site_select,
            self.marking_pane,
            self.rows_container,
            pn.Row(self.add_row_button, self.apply_to_all_button),
            self.method_select,
            pn.pane.Markdown(_METHOD_HELP),
            self.save_button,
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

    def add_window_row(self) -> None:
        """Append a blank period to the selected site."""
        site = self.site_select.value
        self.site_to_rows[site].append(self._build_row(site, None, None))
        self._render_rows()

    def remove_window_row(self, row_index: int) -> None:
        """Drop one period from the selected site."""
        self._remove_row(self.site_to_rows[self.site_select.value][row_index])

    def apply_windows_to_all_sites(self) -> None:
        """Copy the selected site's periods into every other recording site."""
        windows = self.windows_for(self.site_select.value)
        for site in self.sites:
            if site != self.site_select.value:
                self.set_windows(site, windows)

    def save(self) -> None:
        """Write the keep-windows for every site, then record the removal method.

        All sites are validated before anything is written, so an invalid period raises
        up-front without leaving a partially-written set of coords files. A site with no
        marked periods writes no file (keep-the-entire-recording default).
        """
        site_to_keep_windows = {}
        for site in self.sites:
            timestamps = self.pair_traces[site]["x"]
            artifact_windows = _validated_windows(self.windows_for(site), timestamps)
            if not artifact_windows:
                continue
            span_start, span_end = _margined_span(timestamps)
            site_to_keep_windows[site] = complement_windows(
                windows=merge_windows(windows=artifact_windows), span_start=span_start, span_end=span_end
            )

        for site, keep_windows in site_to_keep_windows.items():
            np.save(
                os.path.join(self.filepath, f"coordsForPreProcessing_{site}.npy"),
                windows_to_coords(windows=keep_windows),
            )
            logger.info(f"Saved {len(keep_windows)} keep-window(s) for recording site {site}.")

        record_artifact_provenance(destination=self.filepath, artifacts_removal_method=self.method_select.value)

    def _remove_row(self, row: ArtifactWindowRow) -> None:
        self.site_to_rows[self.site_select.value].remove(row)
        self._render_rows()
        self.refresh_spans()

    def _on_site_change(self, event: object) -> None:
        self._render_rows()
        # A new site means new traces, so the layout is rebuilt; give it a fresh pipe
        # so the discarded plot stops receiving updates.
        self.spans_pipe = make_spans_pipe(windows=self.windows_for(self.site_select.value))
        self.marking_pane.object = self._make_marking_plot()

    def _make_marking_plot(self) -> object:
        site = self.site_select.value
        trace = self.pair_traces[site]
        return build_control_signal_fit(
            x=trace["x"],
            control=trace["control"],
            signal=trace["signal"],
            fit=trace["fit"],
            titles=trace["plot_name"],
            suptitle=os.path.basename(self.filepath),
            spans=self.spans_pipe,
        )

    def _on_save(self, event: object) -> None:
        try:
            self.save()
        except ValueError as error:
            logger.error(str(error))
            if pn.state.notifications is not None:
                pn.state.notifications.error(str(error), duration=0)
            return
        if pn.state.notifications is not None:
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

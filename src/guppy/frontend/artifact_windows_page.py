"""Panel page for the optional Select Artifact Windows step.

The user marks the periods where artifacts occurred on a run's control/signal/fit
traces. Saving writes the complementary keep-windows to
``coordsForPreProcessing_<recording_site>.npy`` and records the removal method into the
run's ``GuPPyParamtersUsed.json``, which the Remove Artifacts step then consumes.
"""

import logging
import os

import numpy as np
import pandas as pd
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

pn.extension("tabulator", notifications=True)

logger = logging.getLogger(__name__)

ARTIFACT_REMOVAL_METHODS = ["replace with NaN", "concatenate"]

_INSTRUCTIONS = (
    "Mark the periods where **artifacts occurred** for each recording site: type the "
    "start/end times (seconds) into the table and the shaded spans update to match. "
    "Everything outside the marked periods is kept. Leave the table empty to keep the "
    "entire recording. Click **Save** to write the windows, then run **Remove Artifacts**."
)

_METHOD_HELP = (
    "How the marked periods are applied. **replace with NaN** (recommended) blanks them "
    "while preserving the recording timeline. **concatenate** splices the kept segments "
    "together; this re-times the kept samples onto a new timeline, breaks alignment to the "
    "acquisition clock, is unsupported by NWB export, and is incompatible with "
    "cross-correlation."
)


def _empty_windows_df() -> pd.DataFrame:
    return pd.DataFrame({"start": pd.Series(dtype="float"), "end": pd.Series(dtype="float")})


def _windows_df(windows: list[tuple[float, float]]) -> pd.DataFrame:
    if not windows:
        return _empty_windows_df()
    return pd.DataFrame({"start": [start for start, _ in windows], "end": [end for _, end in windows]})


def _complete_windows(df: pd.DataFrame) -> list[tuple[float, float]]:
    """Return the ``(start, end)`` rows that have both bounds filled in."""
    return [
        (float(row["start"]), float(row["end"]))
        for _, row in df.iterrows()
        if not pd.isna(row["start"]) and not pd.isna(row["end"])
    ]


def _validated_windows(df: pd.DataFrame, timestamps: np.ndarray) -> list[tuple[float, float]]:
    """Validate the complete rows of an artifact-window table against the recording timespan.

    Raises
    ------
    ValueError
        If a window is inverted or falls outside the recording.
    """
    low, high = float(timestamps[0]), float(timestamps[-1])
    windows = []
    for start, end in _complete_windows(df):
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
        windows.append((start, end))
    return windows


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


class ArtifactWindowSelector:
    """Interactive artifact-marking page for one run folder across all recording sites.

    Shows the selected site's control/signal/fit traces with the marked artifact periods
    shaded, an editable ``(start, end)`` table, and a removal-method selector. On save,
    writes one ``coordsForPreProcessing_<site>.npy`` per site holding the keep-windows.
    """

    def __init__(self, filepath: str, pair_traces: dict[str, dict[str, object]]) -> None:
        self.filepath = filepath
        self.pair_traces = pair_traces
        self.sites = list(pair_traces.keys())

        self.site_to_table = {
            site: pn.widgets.Tabulator(
                _windows_df(_saved_artifact_windows(filepath, site, pair_traces[site]["x"])),
                show_index=False,
                selectable="checkbox",
                widths=180,
            )
            for site in self.sites
        }
        self.site_select = pn.widgets.Select(name="Recording site", options=self.sites, value=self.sites[0])
        self.add_row_button = pn.widgets.Button(name="Add window", button_type="default")
        self.remove_rows_button = pn.widgets.Button(name="Remove selected", button_type="default")
        self.apply_to_all_button = pn.widgets.Button(name="Apply to all recording sites", button_type="default")
        self.method_select = pn.widgets.Select(
            name="Removal method", options=ARTIFACT_REMOVAL_METHODS, value=ARTIFACT_REMOVAL_METHODS[0], width=200
        )
        self.save_button = pn.widgets.Button(name="Save", button_type="primary")

        self.spans_pipe = make_spans_pipe(windows=self._marked_windows(self.sites[0]))
        self.marking_pane = pn.pane.HoloViews(self._make_marking_plot(), width=800)
        self.table_container = pn.Row(self.site_to_table[self.sites[0]])

        self.site_select.param.watch(self._on_site_change, "value")
        for table in self.site_to_table.values():
            table.on_edit(self._on_table_edit)
        self.add_row_button.on_click(lambda event: self.add_window_row())
        self.remove_rows_button.on_click(lambda event: self.remove_selected_window_rows())
        self.apply_to_all_button.on_click(lambda event: self.apply_windows_to_all_sites())
        self.save_button.on_click(self._on_save)

        self.widget = pn.Column(
            "# Select Artifact Windows — {}".format(os.path.basename(filepath)),
            pn.pane.Markdown(_INSTRUCTIONS),
            self.site_select,
            self.marking_pane,
            self.table_container,
            pn.Row(self.add_row_button, self.remove_rows_button, self.apply_to_all_button),
            self.method_select,
            pn.pane.Markdown(_METHOD_HELP),
            self.save_button,
        )

    def _marked_windows(self, site: str) -> list[tuple[float, float]]:
        """Complete rows of a site's table, unvalidated — used for the live preview."""
        return _complete_windows(self.site_to_table[site].value)

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
            artifacts_have_been_removed=False,
            spans=self.spans_pipe,
        )

    def refresh_spans(self) -> None:
        """Repaint the shaded spans from the current table, leaving the traces untouched."""
        self.spans_pipe.send(self._marked_windows(self.site_select.value))

    def add_window_row(self) -> None:
        """Append an empty window row to the selected site's table."""
        table = self.site_to_table[self.site_select.value]
        table.value = pd.concat([table.value, pd.DataFrame({"start": [np.nan], "end": [np.nan]})], ignore_index=True)

    def remove_selected_window_rows(self) -> None:
        """Drop the checked rows from the selected site's table."""
        table = self.site_to_table[self.site_select.value]
        table.value = table.value.drop(index=table.value.index[table.selection]).reset_index(drop=True)
        table.selection = []
        self.refresh_spans()

    def apply_windows_to_all_sites(self) -> None:
        """Copy the selected site's windows into every other recording site."""
        source = self.site_to_table[self.site_select.value].value
        for site, table in self.site_to_table.items():
            if site != self.site_select.value:
                table.value = source.copy()
        self.refresh_spans()

    def save(self) -> None:
        """Write the keep-windows for every site, then record the removal method.

        All sites are validated before anything is written, so an invalid window raises
        up-front without leaving a partially-written set of coords files. A site with no
        marked windows writes no file (keep-the-entire-recording default).
        """
        site_to_keep_windows = {}
        for site, table in self.site_to_table.items():
            timestamps = self.pair_traces[site]["x"]
            artifact_windows = _validated_windows(table.value, timestamps)
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

    def _on_site_change(self, event: object) -> None:
        self.table_container[:] = [self.site_to_table[self.site_select.value]]
        # A new site means new traces, so the layout is rebuilt; give it a fresh pipe
        # so the discarded plot stops receiving updates.
        self.spans_pipe = make_spans_pipe(windows=self._marked_windows(self.site_select.value))
        self.marking_pane.object = self._make_marking_plot()

    def _on_table_edit(self, event: object) -> None:
        self.refresh_spans()

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

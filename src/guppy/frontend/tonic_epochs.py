"""Panel page for defining tonic/basal analysis epoch windows.

Tonic analysis averages the preprocessed z-score / dF/F trace over user-defined
absolute-time windows. Because injection timing can differ per recording site
(e.g. an ICV injection reaches sites at different times), windows are defined
per recording site. This page shows each site's z-score and dF/F traces with the
current windows shaded, plus one editable ``(label, start, end)`` row per window.
Saving averages the traces over the windows and writes both the definitions
(``tonic_epochs_<site>.csv``) and the per-epoch means (``tonic_<site>.h5``).
"""

import glob
import logging
import os
from collections.abc import Callable

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn

from .artifact_removal import build_run_folder_page
from ..analysis.io_utils import read_hdf5, recording_site_from_preprocessed_label
from ..analysis.standard_io import read_tonic_epochs, write_tonic_to_hdf5
from ..analysis.tonic import (
    TONIC_EPOCH_COLUMNS,
    compute_tonic_means,
    validate_tonic_epochs,
)
from ..visualization.preprocessing import build_stacked_traces, make_spans_pipe

pn.extension(notifications=True)
hv.extension("bokeh")

logger = logging.getLogger(__name__)

# Seconds per arrow-key press on a window bound, for nudging an edge into place.
_NUDGE_STEP = 0.1
_LABEL_WIDTH = 160
_INPUT_WIDTH = 110

_INSTRUCTIONS = (
    "Define named epoch windows for each recording site: enter a label and the start and "
    "end time (seconds) of each window, and the shaded spans update to match. Use "
    "**Apply to all recording sites** when the injection is systemic.\n\n"
    "Click **Save** to compute and store each window's mean z-score and ΔF/F; the results "
    "appear on the Tonic tab of the visualization. A site with no windows is skipped."
)

_NO_WINDOWS_HINT = "_No epoch windows defined — this recording site will be skipped._"

# Bar hues for the results view: the baseline epoch is the reference the others read against.
_EPOCH_BAR_COLOR = "#1f77b4"
_BASELINE_BAR_COLOR = "#ff7f0e"

_BASELINE_HINT = (
    "Bars show each epoch's mean; the **baseline epoch** is highlighted and its value drawn "
    "as a dashed reference line, so every other bar's distance from it is the change from "
    "baseline reported in the table below."
)


def load_site_traces(filepath: str) -> dict[str, dict[str, np.ndarray]]:
    """Load the preprocessed z-score / dF/F traces for every recording site in a run folder.

    Parameters
    ----------
    filepath : str
        Session output (run) directory containing ``z_score_*`` / ``dff_*`` files.

    Returns
    -------
    dict
        Mapping recording-site name → ``{"x", "y_zscore", "y_dff"}`` arrays.
    """
    site_traces: dict[str, dict[str, np.ndarray]] = {}
    for path in sorted(glob.glob(os.path.join(filepath, "z_score_*"))):
        basename = os.path.basename(path).split(".")[0]
        site = recording_site_from_preprocessed_label(basename)
        site_traces[site] = {
            "x": np.asarray(read_hdf5("timeCorrection_" + site, filepath, "timestampNew")).ravel(),
            "y_zscore": np.asarray(read_hdf5("", path, "data")).ravel(),
            "y_dff": np.asarray(read_hdf5("", os.path.join(filepath, "dff_" + site + ".hdf5"), "data")).ravel(),
        }
    return site_traces


def _saved_epoch_windows(filepath: str, site: str) -> list[tuple[str, float, float]]:
    """Re-read the epoch windows saved on disk for one recording site."""
    epochs = read_tonic_epochs(filepath, site)
    if epochs.empty:
        return []
    return [(str(row["label"]), float(row["start"]), float(row["end"])) for _, row in epochs.iterrows()]


class TonicEpochRow:
    """One epoch window: a label, a start bound, an end bound, and a delete button.

    The bounds are numeric inputs limited to the recording timespan, so an edge can be
    nudged with the arrow keys. The limits are a browser-side guardrail only — a value
    set programmatically is not clamped, so callers still validate before saving.
    """

    def __init__(
        self,
        *,
        label: str,
        start: float | None,
        end: float | None,
        span: tuple[float, float],
        on_change: Callable[[], None],
        on_remove: Callable[["TonicEpochRow"], None],
    ) -> None:
        low, high = span
        self.label_input = pn.widgets.TextInput(value=label, width=_LABEL_WIDTH)
        self.start_input = pn.widgets.FloatInput(value=start, start=low, end=high, step=_NUDGE_STEP, width=_INPUT_WIDTH)
        self.end_input = pn.widgets.FloatInput(value=end, start=low, end=high, step=_NUDGE_STEP, width=_INPUT_WIDTH)
        self.remove_button = pn.widgets.Button(
            icon="trash", button_type="light", width=40, height=38, description="Delete this epoch"
        )

        self.start_input.param.watch(lambda event: on_change(), "value")
        self.end_input.param.watch(lambda event: on_change(), "value")
        self.remove_button.on_click(lambda event: on_remove(self))

        self.widget = pn.Row(
            self.label_input,
            self.start_input,
            self.end_input,
            self.remove_button,
            align="center",
            margin=(0, 0, 4, 0),
        )

    @property
    def epoch(self) -> tuple[str, float, float] | None:
        """The ``(label, start, end)`` window, or None while a bound is still blank."""
        if self.start_input.value is None or self.end_input.value is None:
            return None
        return self.label_input.value, float(self.start_input.value), float(self.end_input.value)

    @property
    def window(self) -> tuple[float, float] | None:
        """The ``(start, end)`` bounds used for shading, or None while a bound is blank."""
        epoch = self.epoch
        return None if epoch is None else (epoch[1], epoch[2])


class TonicEpochConfig:
    """Interactive epoch-definition page for one run folder across all recording sites.

    Shows the selected site's z-score and dF/F traces with the current epoch windows
    shaded, one editable row per window, and an "apply to all recording sites" button.
    On save, writes each site's windows and their per-epoch means.
    """

    def __init__(self, filepath: str, site_traces: dict[str, dict[str, np.ndarray]]) -> None:
        self.filepath = filepath
        self.site_traces = site_traces
        self.sites = list(site_traces.keys())

        self.site_to_rows: dict[str, list[TonicEpochRow]] = {
            site: [
                self._build_row(site, label, start, end) for label, start, end in _saved_epoch_windows(filepath, site)
            ]
            for site in self.sites
        }

        self.site_select = pn.widgets.Select(name="Recording site", options=self.sites, value=self.sites[0])
        self.add_row_button = pn.widgets.Button(name="Add epoch", button_type="default", icon="plus")
        self.apply_to_all_button = pn.widgets.Button(name="Apply to all recording sites", button_type="default")
        self.save_button = pn.widgets.Button(name="Save", button_type="primary")

        self.spans_pipe = make_spans_pipe(windows=self.windows_for(self.sites[0]))
        self.plot_pane = pn.pane.HoloViews(self._make_plot(), sizing_mode="stretch_width")
        self.rows_container = pn.Column()
        self._render_rows()

        self.site_select.param.watch(self._on_site_change, "value")
        self.add_row_button.on_click(lambda event: self.add_epoch_row())
        self.apply_to_all_button.on_click(lambda event: self.apply_epochs_to_all_sites())
        self.save_button.on_click(self._on_save)

        self.widget = pn.Column(
            "# Define Tonic Epochs — {}".format(os.path.basename(filepath)),
            pn.pane.Markdown(_INSTRUCTIONS),
            self.site_select,
            self.plot_pane,
            self.rows_container,
            pn.Row(self.add_row_button, self.apply_to_all_button),
            self.save_button,
            sizing_mode="stretch_width",
        )

    def _build_row(self, site: str, label: str, start: float | None, end: float | None) -> TonicEpochRow:
        timestamps = self.site_traces[site]["x"]
        return TonicEpochRow(
            label=label,
            start=start,
            end=end,
            span=(float(timestamps[0]), float(timestamps[-1])),
            on_change=self.refresh_spans,
            on_remove=self._remove_row,
        )

    def _column_header(self) -> pn.Row:
        return pn.Row(
            pn.pane.HTML("<b>label</b>", width=_LABEL_WIDTH),
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

    def epochs_for(self, site: str) -> list[tuple[str, float, float]]:
        """The fully-entered epoch windows defined for one recording site."""
        return [row.epoch for row in self.site_to_rows[site] if row.epoch is not None]

    def windows_for(self, site: str) -> list[tuple[float, float]]:
        """The ``(start, end)`` bounds of one recording site's fully-entered windows."""
        return [row.window for row in self.site_to_rows[site] if row.window is not None]

    def set_epochs(self, site: str, epochs: list[tuple[str, float, float]]) -> None:
        """Replace one recording site's epoch windows."""
        self.site_to_rows[site] = [self._build_row(site, label, start, end) for label, start, end in epochs]
        if site == self.site_select.value:
            self._render_rows()
        self.refresh_spans()

    def refresh_spans(self) -> None:
        """Repaint the shaded spans from the current rows, leaving the traces untouched."""
        self.spans_pipe.send(self.windows_for(self.site_select.value))

    def add_epoch_row(self) -> None:
        """Append a blank epoch window to the selected site."""
        site = self.site_select.value
        self.site_to_rows[site].append(self._build_row(site, "", None, None))
        self._render_rows()

    def apply_epochs_to_all_sites(self) -> None:
        """Copy the selected site's epoch windows into every other recording site."""
        epochs = self.epochs_for(self.site_select.value)
        for site in self.sites:
            if site != self.site_select.value:
                self.set_epochs(site, epochs)

    def save(self) -> None:
        """Validate and write ``tonic_epochs_<site>.csv`` for every site with a complete window.

        Every site's windows are validated (against that site's own timespan)
        before anything is written, so an invalid window raises up-front without
        leaving a partially-written set of epoch files.

        Averaging the traces over the windows is a read-only reduction over the
        Step-3 outputs already in memory here, so the means are computed and
        written alongside the windows rather than on a later preprocessing pass.
        """
        to_write = {}
        for site in self.sites:
            epochs = self.epochs_for(site)
            if not epochs:
                continue
            complete = pd.DataFrame(epochs, columns=TONIC_EPOCH_COLUMNS)
            timestamps = self.site_traces[site]["x"]
            validate_tonic_epochs(complete, float(timestamps[0]), float(timestamps[-1]))
            to_write[site] = complete

        for site, complete in to_write.items():
            complete.to_csv(os.path.join(self.filepath, "tonic_epochs_" + site + ".csv"), index=False)
            trace = self.site_traces[site]
            write_tonic_to_hdf5(
                self.filepath,
                compute_tonic_means(trace["y_zscore"], trace["y_dff"], trace["x"], complete),
                site,
            )
            logger.info(f"Saved {len(complete)} tonic epoch(s) and means for recording site {site}.")

    def _remove_row(self, row: TonicEpochRow) -> None:
        self.site_to_rows[self.site_select.value].remove(row)
        self._render_rows()
        self.refresh_spans()

    def _on_site_change(self, event: object) -> None:
        self._render_rows()
        # A new site means new traces, so the layout is rebuilt; give it a fresh pipe
        # so the discarded plot stops receiving updates.
        self.spans_pipe = make_spans_pipe(windows=self.windows_for(self.site_select.value))
        self.plot_pane.object = self._make_plot()

    def _make_plot(self) -> object:
        site = self.site_select.value
        trace = self.site_traces[site]
        return build_stacked_traces(
            x=trace["x"],
            traces={"z-score": trace["y_zscore"], "ΔF/F": trace["y_dff"]},
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
            pn.state.notifications.success("Tonic epoch windows saved.", duration=4000)


def build_tonic_epoch_page(*, run_folders: list[str]) -> pn.viewable.Viewable:
    """Compose the Define Tonic Epochs page across all run folders.

    Parameters
    ----------
    run_folders : list of str
        Session output (run) directories to offer for epoch definition.

    Returns
    -------
    pn.viewable.Viewable
        A single page; when there is more than one run folder a selector switches
        between them so only the selected folder's traces are rendered at a time.
    """
    return build_run_folder_page(
        run_folders=run_folders,
        build_folder_page=lambda filepath: TonicEpochConfig(filepath, load_site_traces(filepath)).widget,
    )


def _tonic_result_sites(filepath: str) -> list[str]:
    sites = []
    for path in sorted(glob.glob(os.path.join(filepath, "tonic_*.h5"))):
        basename = os.path.basename(path)
        sites.append(basename[len("tonic_") : -len(".h5")])
    return sites


class TonicResultsView:
    """Read-only view of tonic results for a single run folder.

    For each recording site it shows a bar chart of the per-epoch means — the
    headline result — above the preprocessed traces with the analysed windows
    shaded, and a table of the same numbers. Because the difference from baseline
    is a viewing choice, a baseline-epoch selector marks the reference epoch on
    the bars and adds live ``diff_zscore`` / ``diff_dff`` columns.
    """

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.sites = _tonic_result_sites(filepath)

        self.site_select = pn.widgets.Select(name="Recording site", options=self.sites, value=self.sites[0])
        means = self._means(self.sites[0])
        self.baseline_select = pn.widgets.Select(
            name="Baseline epoch", options=list(means.index), value=list(means.index)[0]
        )
        self.bars_pane = pn.pane.HoloViews(self._make_bars(), sizing_mode="stretch_width")
        self.plot_pane = pn.pane.HoloViews(self._make_plot(), sizing_mode="stretch_width")
        self.table_pane = pn.pane.DataFrame(self._means_with_diff(), width=520)

        self.site_select.param.watch(self._on_site_change, "value")
        self.baseline_select.param.watch(self._refresh, "value")

        self.widget = pn.Column(
            "## Tonic / basal analysis — {}".format(os.path.basename(filepath)),
            pn.Row(self.site_select, self.baseline_select),
            self.bars_pane,
            pn.pane.Markdown(_BASELINE_HINT),
            self.plot_pane,
            self.table_pane,
            sizing_mode="stretch_width",
        )

    def _means(self, site: str) -> pd.DataFrame:
        return pd.read_hdf(os.path.join(self.filepath, "tonic_" + site + ".h5"), key="df")

    def _make_bars(self) -> hv.Layout:
        """Per-epoch means as bars, one panel per signal.

        The z-score and dF/F means differ by orders of magnitude, so they get a panel
        each rather than a shared axis that would flatten one of them.
        """
        means = self._means(self.site_select.value)
        baseline = self.baseline_select.value

        panels = []
        for column, label in (("mean_zscore", "mean z-score"), ("mean_dff", "mean ΔF/F")):
            epochs = list(means.index)
            values = [float(means.loc[epoch, column]) for epoch in epochs]
            colors = [_BASELINE_BAR_COLOR if epoch == baseline else _EPOCH_BAR_COLOR for epoch in epochs]
            bars = hv.Bars(list(zip(epochs, values)), "epoch", label).opts(
                color=hv.dim("epoch").categorize(dict(zip(epochs, colors))),
                title=label,
                responsive=True,
                height=280,
                xrotation=30,
                tools=["hover"],
            )
            # A line at the baseline value turns each bar's height difference from the
            # reference into something readable straight off the chart.
            reference = hv.HLine(float(means.loc[baseline, column])).opts(
                color=_BASELINE_BAR_COLOR, line_dash="dashed", line_width=1.5
            )
            panels.append(bars * reference)
        return hv.Layout(panels).cols(2)

    def _means_with_diff(self) -> pd.DataFrame:
        means = self._means(self.site_select.value)
        baseline = self.baseline_select.value
        diff = means - means.loc[baseline]
        return pd.DataFrame(
            {
                "mean_zscore": means["mean_zscore"],
                "mean_dff": means["mean_dff"],
                "diff_zscore": diff["mean_zscore"],
                "diff_dff": diff["mean_dff"],
            }
        )

    def _make_plot(self) -> object:
        site = self.site_select.value
        timestamps = np.asarray(read_hdf5("timeCorrection_" + site, self.filepath, "timestampNew")).ravel()
        traces = {
            "z-score": np.asarray(
                read_hdf5("", os.path.join(self.filepath, "z_score_" + site + ".hdf5"), "data")
            ).ravel(),
            "ΔF/F": np.asarray(read_hdf5("", os.path.join(self.filepath, "dff_" + site + ".hdf5"), "data")).ravel(),
        }
        # The analysed windows are fixed on disk, so the pipe is seeded once per rebuild
        # rather than driven by edits; it is retained so the spans layer keeps its source.
        epochs = read_tonic_epochs(self.filepath, site)
        self.spans_pipe = make_spans_pipe(
            windows=[(float(row["start"]), float(row["end"])) for _, row in epochs.iterrows()]
        )
        return build_stacked_traces(
            x=timestamps,
            traces=traces,
            suptitle=os.path.basename(self.filepath),
            spans=self.spans_pipe,
        )

    def _on_site_change(self, event: object) -> None:
        means = self._means(self.site_select.value)
        self.baseline_select.options = list(means.index)
        self.baseline_select.value = list(means.index)[0]
        self._refresh(event)

    def _refresh(self, event: object) -> None:
        self.bars_pane.object = self._make_bars()
        self.plot_pane.object = self._make_plot()
        self.table_pane.object = self._means_with_diff()


def build_tonic_results_view(filepath: str) -> pn.Column:
    """Build the Step-5 tonic results panel for a run folder.

    Returns a short note when the folder holds no ``tonic_<site>.h5`` results so
    the tab can be shown unconditionally.
    """
    if not _tonic_result_sites(filepath):
        return pn.Column(pn.pane.Markdown("_No tonic/basal analysis results in this session._"))
    return TonicResultsView(filepath).widget

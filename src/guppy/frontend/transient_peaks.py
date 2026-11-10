"""Reusable Panel component for Step-4 transient-peak visualization.

A terminal read-only display composed by ``orchestration/transients_view.py`` and served
on the persistent main app after the transient-analysis compute job finishes. A run
folder selector and a trace selector pick one ``z_score`` / ``dff`` trace, shown with its
detected transient peaks marked.
"""

import logging
from pathlib import Path

import holoviews as hv
import numpy as np
import panel as pn

from .plot_card import CONTROL_MARGIN, RUN_FOLDER_WIDTH, plot_page, toolbar_field
from ..analysis.standard_io import read_transients_from_hdf5
from ..utils.utils import relative_output_labels
from ..visualization.labels import trace_title
from ..visualization.transients import build_peaks_overlay

# Load the HoloViews bokeh backend for the peak-overlay plots.
pn.extension(notifications=True)
hv.extension("bokeh")

logger = logging.getLogger(__name__)


def _trace_paths(filepath: str, select_for_transients: str) -> list[str]:
    if select_for_transients == "z_score":
        return list(Path(filepath).glob("z_score_*"))
    if select_for_transients == "dff":
        return list(Path(filepath).glob("dff_*"))
    return list(Path(filepath).glob("z_score_*")) + list(Path(filepath).glob("dff_*"))


def load_peaks(run_folders: list[str], select_for_transients: str) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Load z-score / dF-F traces and their detected peaks for the given run folders.

    Parameters
    ----------
    run_folders : list of str
        Session output (run) directories to load transient outputs from.
    select_for_transients : {'z_score', 'dff', ...}
        Which preprocessed signal(s) to load; any value other than ``'z_score'`` or
        ``'dff'`` loads both.

    Returns
    -------
    dict
        Mapping run folder label → trace name → ``{"z_score", "timestamps", "peaksInd"}``.
    """
    labels = relative_output_labels(run_folders)
    runs: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for filepath in run_folders:
        traces: dict[str, dict[str, np.ndarray]] = {}
        for path in _trace_paths(filepath, select_for_transients):
            title = path.name.split(".")[0]
            z_score, timestamps, peaksInd = read_transients_from_hdf5(filepath, title)
            traces[title] = {
                "z_score": np.asarray(z_score).ravel(),
                "timestamps": np.asarray(timestamps).ravel(),
                "peaksInd": np.asarray(peaksInd).ravel().astype(int),
            }
        runs[labels[str(filepath)]] = traces
    return runs


class PeaksReviewView:
    """Read-only transient-peak review: a run folder, then one of its traces, with its peaks marked."""

    def __init__(self, runs: dict[str, dict[str, dict[str, np.ndarray]]]) -> None:
        self.runs = runs
        first_run = next(iter(runs))

        self.run_select = pn.widgets.Select(
            options=list(runs), value=first_run, width=RUN_FOLDER_WIDTH, margin=CONTROL_MARGIN
        )
        self.trace_select = pn.widgets.Select(
            options=self._trace_options(first_run), value=next(iter(runs[first_run])), width=220, margin=CONTROL_MARGIN
        )
        self.plot_pane = pn.pane.HoloViews(self._make_plot(), sizing_mode="stretch_width")
        self.run_select.param.watch(self._on_run_change, "value")
        self.trace_select.param.watch(self._refresh, "value")

        self.widget = plot_page(
            note="Each transient peak detected on the trace is marked in red.",
            fields=[toolbar_field("Run Folder", self.run_select), toolbar_field("Trace", self.trace_select)],
            body=[self.plot_pane],
        )

    def _make_plot(self) -> hv.DynamicMap:
        entry = self.runs[self.run_select.value][self.trace_select.value]
        return build_peaks_overlay(
            title=trace_title(self.trace_select.value),
            z_score=entry["z_score"],
            timestamps=entry["timestamps"],
            peaksIndex=entry["peaksInd"],
        )

    def _on_run_change(self, event: object) -> None:
        traces = list(self.runs[event.new])
        # Keep the same trace when the new run has it, so runs can be compared trace by trace.
        current = self.trace_select.value
        self.trace_select.options = self._trace_options(event.new)
        if current in traces:
            self._refresh(event)
        else:
            self.trace_select.value = traces[0]

    def _trace_options(self, run: str) -> dict[str, str]:
        """Name each of a run's traces by what it is and its site, keyed to its file name."""
        return {trace_title(trace): trace for trace in self.runs[run]}

    def _refresh(self, event: object) -> None:
        self.plot_pane.object = self._make_plot()


def build_peaks_view_page(run_folders: list[str], select_for_transients: str) -> pn.viewable.Viewable:
    """Compose the read-only transient-peak review page over the given run folders.

    Parameters
    ----------
    run_folders : list of str
        Session output (run) directories produced by the transient-analysis job.
    select_for_transients : str
        Which preprocessed signal(s) to show (``'z_score'``, ``'dff'``, or both).

    Returns
    -------
    pn.viewable.Viewable
        The page content: run folder and trace selectors over the detected-peak plot.
    """
    return PeaksReviewView(load_peaks(run_folders, select_for_transients)).widget

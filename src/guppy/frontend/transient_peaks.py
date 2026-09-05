"""Reusable Panel component for Step-4 transient-peak visualization.

A terminal read-only display composed by ``orchestration/transients_view.py`` and served
on the persistent main app after the transient-analysis compute job finishes. A single
selector switches between every ``z_score`` / ``dff`` trace across the run folders, each
shown with its detected transient peaks marked.
"""

import logging
from pathlib import Path

import holoviews as hv
import numpy as np
import panel as pn

from ..analysis.standard_io import read_transients_from_hdf5
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


def load_peaks(run_folders: list[str], select_for_transients: str) -> dict[str, dict[str, np.ndarray]]:
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
        Mapping ``"<run folder> / <trace>"`` label → ``{"z_score", "timestamps", "peaksInd"}``.
    """
    entries: dict[str, dict[str, np.ndarray]] = {}
    for filepath in run_folders:
        for path in _trace_paths(filepath, select_for_transients):
            title = path.name.split(".")[0]
            suptitle = path.parent.name
            z_score, timestamps, peaksInd = read_transients_from_hdf5(filepath, title)
            entries[f"{suptitle} / {title}"] = {
                "z_score": np.asarray(z_score).ravel(),
                "timestamps": np.asarray(timestamps).ravel(),
                "peaksInd": np.asarray(peaksInd).ravel().astype(int),
                "title": title,
                "suptitle": suptitle,
            }
    return entries


class PeaksReviewView:
    """Read-only transient-peak review with a selector over every loaded trace."""

    def __init__(self, entries: dict[str, dict[str, np.ndarray]]) -> None:
        self.entries = entries
        self.labels = list(entries.keys())

        self.trace_select = pn.widgets.Select(name="Trace", options=self.labels, value=self.labels[0])
        self.plot_pane = pn.pane.HoloViews(self._make_plot(), sizing_mode="stretch_width")
        self.trace_select.param.watch(self._refresh, "value")

        self.widget = pn.Column("## Transient peaks", self.trace_select, self.plot_pane, sizing_mode="stretch_width")

    def _make_plot(self) -> hv.DynamicMap:
        entry = self.entries[self.trace_select.value]
        return build_peaks_overlay(
            title=entry["title"],
            suptitle=entry["suptitle"],
            z_score=entry["z_score"],
            timestamps=entry["timestamps"],
            peaksIndex=entry["peaksInd"],
        )

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
        The page content (a trace selector over the detected-peak plots).
    """
    return PeaksReviewView(load_peaks(run_folders, select_for_transients)).widget

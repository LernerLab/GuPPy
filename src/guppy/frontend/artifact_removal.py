"""Reusable Panel components for the read-only preprocessing and artifact views.

These are terminal displays composed by ``orchestration/preprocess_view.py`` and
``orchestration/artifact_view.py`` and served as pages on the persistent main app after a
compute job finishes — the components never signal a waiting backend (no blocking, no
server teardown):

* :class:`PreprocessingReviewView` — read-only review of one run folder's preprocessing
  output.
* :func:`build_run_folder_page` — composes a per-folder page over all run folders.

The interactive marking page lives in ``frontend/artifact_windows_page.py``.
"""

import glob
import logging
import os
from collections.abc import Callable

import holoviews as hv
import numpy as np
import panel as pn

from ..analysis.io_utils import (
    decide_naming_convention,
    read_hdf5,
    recording_site_from_channel_path,
    recording_site_from_preprocessed_label,
)
from ..visualization.preprocessing import build_control_signal_fit

# Load the bokeh HoloViews backend these components rely on for the trace plots.
pn.extension(notifications=True)
hv.extension("bokeh")

logger = logging.getLogger(__name__)


def load_pair_traces(filepath: str) -> dict[str, dict[str, object]]:
    """Load the control/signal/fit traces for every channel pair in a run folder.

    Parameters
    ----------
    filepath : str
        Session output (run) directory containing ``control_*`` / ``signal_*`` /
        ``cntrl_sig_fit_*`` HDF5 files.

    Returns
    -------
    dict
        Mapping recording-site name → ``{"x", "control", "signal", "fit", "plot_name"}``.
    """
    path = decide_naming_convention(filepath)
    pair_traces: dict[str, dict[str, object]] = {}
    for i in range(path.shape[1]):
        site = recording_site_from_channel_path(path[0, i])
        pair_traces[site] = {
            "x": np.asarray(read_hdf5("timeCorrection_" + site, filepath, "timestampNew")).ravel(),
            "control": np.asarray(read_hdf5("", path[0, i], "data")).ravel(),
            "signal": np.asarray(read_hdf5("", path[1, i], "data")).ravel(),
            "fit": np.asarray(read_hdf5("cntrl_sig_fit_" + site, filepath, "data")).ravel(),
            "plot_name": [
                os.path.basename(path[0, i]).split(".")[0],
                os.path.basename(path[1, i]).split(".")[0],
                "cntrl_sig_fit_" + site,
            ],
        }
    return pair_traces


def load_preprocessed_traces(filepath: str) -> dict[str, dict[str, np.ndarray]]:
    """Load the z-score and dF-F traces for every recording site in a run folder.

    Parameters
    ----------
    filepath : str
        Session output (run) directory containing ``z_score_*`` / ``dff_*`` files.

    Returns
    -------
    dict
        Mapping recording-site name → ``{"x", "y_zscore", "y_dff"}``.
    """
    traces: dict[str, dict[str, np.ndarray]] = {}
    for path in sorted(glob.glob(os.path.join(filepath, "z_score_*"))):
        site = recording_site_from_preprocessed_label(os.path.basename(path).split(".")[0])
        traces[site] = {
            "x": np.asarray(read_hdf5("timeCorrection_" + site, filepath, "timestampNew")).ravel(),
            "y_zscore": np.asarray(read_hdf5("", path, "data")).ravel(),
            "y_dff": np.asarray(read_hdf5("dff_" + site, filepath, "data")).ravel(),
        }
    return traces


class PreprocessingReviewView:
    """Read-only review of one run folder's preprocessing output, with a site selector.

    A recording site's five traces — control, signal, signal+fit, z-score and dF-F — stack
    in one axis-linked layout, so an excursion in the raw signal can be followed straight
    down into the metrics derived from it.
    """

    def __init__(
        self,
        filepath: str,
        pair_traces: dict[str, dict[str, object]],
        preprocessed_traces: dict[str, dict[str, np.ndarray]],
        *,
        artifacts_removed: bool,
    ) -> None:
        self.filepath = filepath
        self.pair_traces = pair_traces
        self.preprocessed_traces = preprocessed_traces
        self.sites = list(pair_traces.keys())

        self.site_select = pn.widgets.Select(name="Recording site", options=self.sites, value=self.sites[0])
        self.plot_pane = pn.pane.HoloViews(self._make_plot(), sizing_mode="stretch_width")
        self.site_select.param.watch(self._refresh, "value")

        heading = "Artifact removal review" if artifacts_removed else "Preprocessing review"
        self.widget = pn.Column(
            f"## {heading} — {os.path.basename(filepath)}",
            self.site_select,
            self.plot_pane,
            sizing_mode="stretch_width",
        )

    def _make_plot(self) -> hv.Layout:
        site = self.site_select.value
        trace = self.pair_traces[site]
        preprocessed = self.preprocessed_traces[site]
        return build_control_signal_fit(
            x=trace["x"],
            control=trace["control"],
            signal=trace["signal"],
            fit=trace["fit"],
            titles=trace["plot_name"],
            suptitle=os.path.basename(self.filepath),
            extra_traces={
                f"z_score_{site}": preprocessed["y_zscore"],
                f"dff_{site}": preprocessed["y_dff"],
            },
        )

    def _refresh(self, event: object) -> None:
        self.plot_pane.object = self._make_plot()


def build_run_folder_page(
    *, run_folders: list[str], build_folder_page: Callable[[str], pn.viewable.Viewable]
) -> pn.viewable.Viewable:
    """Compose a per-folder page across all run folders, with a folder selector.

    Parameters
    ----------
    run_folders : list of str
        Session output (run) directories to offer.
    build_folder_page : callable
        Builds the page content for one run folder.

    Returns
    -------
    pn.viewable.Viewable
        A single page; when there is more than one run folder a selector switches
        between them so only the selected folder is rendered at a time.
    """
    run_folders = list(run_folders)
    content = pn.Column(build_folder_page(run_folders[0]), sizing_mode="stretch_width")
    if len(run_folders) == 1:
        return content

    options = {f"{os.path.basename(os.path.dirname(f))}/{os.path.basename(f)}": f for f in run_folders}
    folder_select = pn.widgets.Select(name="Run folder", options=options, value=run_folders[0])

    def _on_folder_change(event: object) -> None:
        content[:] = [build_folder_page(folder_select.value)]

    folder_select.param.watch(_on_folder_change, "value")
    return pn.Column(folder_select, content, sizing_mode="stretch_width")


def _build_review_page(filepath: str, *, artifacts_removed: bool) -> pn.viewable.Viewable:
    """Compose one run folder's preprocessing review."""
    return PreprocessingReviewView(
        filepath,
        load_pair_traces(filepath),
        load_preprocessed_traces(filepath),
        artifacts_removed=artifacts_removed,
    ).widget


def build_preprocess_view_page(*, run_folders: list[str]) -> pn.viewable.Viewable:
    """Compose the read-only Step-3 preprocessing view across all run folders.

    Parameters
    ----------
    run_folders : list of str
        Session output (run) directories produced by the preprocessing job.

    Returns
    -------
    pn.viewable.Viewable
        The composed page.
    """
    return build_run_folder_page(
        run_folders=run_folders,
        build_folder_page=lambda filepath: _build_review_page(filepath, artifacts_removed=False),
    )


def build_artifact_review_page(*, run_folders: list[str]) -> pn.viewable.Viewable:
    """Compose the read-only post-removal review across all run folders.

    Parameters
    ----------
    run_folders : list of str
        Session output (run) directories the Remove Artifacts step processed.

    Returns
    -------
    pn.viewable.Viewable
        The composed page.
    """
    return build_run_folder_page(
        run_folders=run_folders,
        build_folder_page=lambda filepath: _build_review_page(filepath, artifacts_removed=True),
    )

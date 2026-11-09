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

import logging
from collections.abc import Callable
from pathlib import Path

import holoviews as hv
import numpy as np
import panel as pn

from .plot_card import CONTROL_MARGIN, RUN_FOLDER_WIDTH, plot_page, toolbar_field
from ..analysis.io_utils import (
    decide_naming_convention,
    read_hdf5,
    recording_site_from_channel_path,
    recording_site_from_preprocessed_label,
)
from ..utils.utils import relative_output_labels
from ..visualization.preprocessing import build_control_signal_fit

# Load the bokeh HoloViews backend these components rely on for the trace plots.
pn.extension(notifications=True)
hv.extension("bokeh")

logger = logging.getLogger(__name__)

_REVIEW_NOTES = {
    False: "The control and signal, the control fitted to the signal, then the z-score and ΔF/F derived from them.",
    True: "The same traces after artifact removal: control, signal, the fit, then the z-score and ΔF/F.",
}


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
                Path(path[0, i]).name.split(".")[0],
                Path(path[1, i]).name.split(".")[0],
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
    for path in sorted(Path(filepath).glob("z_score_*")):
        site = recording_site_from_preprocessed_label(path.name.split(".")[0])
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
        run_field: pn.viewable.Viewable,
        artifacts_removed: bool,
    ) -> None:
        self.filepath = filepath
        self.pair_traces = pair_traces
        self.preprocessed_traces = preprocessed_traces
        self.sites = list(pair_traces.keys())

        self.site_select = pn.widgets.Select(options=self.sites, value=self.sites[0], width=220, margin=CONTROL_MARGIN)
        self.plot_pane = pn.pane.HoloViews(self._make_plot(), sizing_mode="stretch_width")
        self.site_select.param.watch(self._refresh, "value")

        self.widget = plot_page(
            note=_REVIEW_NOTES[artifacts_removed],
            fields=[run_field, toolbar_field("Recording Site", self.site_select)],
            body=[self.plot_pane],
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
            extra_traces={
                f"z_score_{site}": preprocessed["y_zscore"],
                f"dff_{site}": preprocessed["y_dff"],
            },
        )

    def _refresh(self, event: object) -> None:
        self.plot_pane.object = self._make_plot()


def build_run_folder_page(
    *,
    run_folders: list[str],
    build_folder_page: Callable[[str, pn.viewable.Viewable], pn.viewable.Viewable],
) -> pn.viewable.Viewable:
    """Compose a per-folder page across all run folders, with a run folder selector.

    Parameters
    ----------
    run_folders : list of str
        Session output (run) directories to offer.
    build_folder_page : callable
        Builds the page content for one run folder, given its path and the "Run Folder"
        toolbar field to place among the page's own controls.

    Returns
    -------
    pn.viewable.Viewable
        A single page; the selector switches between run folders so only the selected
        folder is rendered at a time.
    """
    run_folders = list(run_folders)
    # Run folders are named for the run alone, so a basename does not say which session
    # a run came from. The label is the run's real path below whatever the folders on
    # offer have in common, which is as much of it as it takes to tell them apart.
    labels = relative_output_labels(run_folders)
    folder_select = pn.widgets.Select(
        options={labels[str(folder)]: folder for folder in run_folders},
        value=run_folders[0],
        width=RUN_FOLDER_WIDTH,
        margin=CONTROL_MARGIN,
    )
    run_field = toolbar_field("Run Folder", folder_select)
    content = pn.Column(build_folder_page(run_folders[0], run_field), sizing_mode="stretch_width")

    def _on_folder_change(event: object) -> None:
        content[:] = [build_folder_page(folder_select.value, run_field)]

    folder_select.param.watch(_on_folder_change, "value")
    return content


def _build_review_page(
    filepath: str, run_field: pn.viewable.Viewable, *, artifacts_removed: bool
) -> pn.viewable.Viewable:
    """Compose one run folder's preprocessing review."""
    return PreprocessingReviewView(
        filepath,
        load_pair_traces(filepath),
        load_preprocessed_traces(filepath),
        run_field=run_field,
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
        build_folder_page=lambda filepath, run_field: _build_review_page(filepath, run_field, artifacts_removed=False),
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
        build_folder_page=lambda filepath, run_field: _build_review_page(filepath, run_field, artifacts_removed=True),
    )

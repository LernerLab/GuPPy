"""Panel pages for Step-3 preprocessing visualization.

Replaces the legacy matplotlib/TkAgg pop-ups. The preprocessing subprocess serves
these pages (blocking until the user continues) via ``serve_blocking_page``:

* :class:`ArtifactRemovalConfig` — interactive artifact marking. The user types
  good-chunk ``(start, end)`` windows per recording site in an editable table; the
  windows are shaded on the control/signal/fit traces and saved to
  ``coordsForPreProcessing_<site>.npy`` (the exact format ``fetchCoords`` consumes).
* :class:`PreprocessingReviewView` — read-only z-score / dF-F review.
* :class:`ArtifactReviewView` — read-only control/signal/fit review after artifacts
  have been removed, with the saved good-chunk windows shaded.
"""

import glob
import logging
import os
from collections.abc import Callable

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn

from ..analysis.io_utils import (
    decide_naming_convention,
    fetchCoords,
    read_hdf5,
    recording_site_from_channel_path,
    recording_site_from_preprocessed_label,
)
from ..visualization.preprocessing import (
    build_control_signal_fit,
    build_preprocessing_curve,
)

# The preprocessing subprocess that serves these pages never runs home.py, so load
# the Panel and HoloViews (bokeh) extensions here — matching tonic_epochs.py.
pn.extension(notifications=True)
hv.extension("bokeh")

logger = logging.getLogger(__name__)

_DEFAULT_WINDOW_ROWS = 3


def signal_options(plot_zScore_dff: str) -> list[str]:
    """Return the z-score / dF-F toggle options implied by the ``plot_zScore_dff`` parameter.

    ``"None"`` (no review requested) yields an empty list, so no z-score/dF-F section
    is rendered — mirroring the legacy behavior where no such figure was created.
    """
    if plot_zScore_dff == "Both":
        return ["z_score", "dff"]
    if plot_zScore_dff == "dff":
        return ["dff"]
    if plot_zScore_dff == "z_score":
        return ["z_score"]
    return []


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


def _windows_from_df(df: pd.DataFrame, timestamps: np.ndarray) -> list[tuple[float, float]]:
    """Extract validated ``(start, end)`` good-chunk windows from an editable table.

    Rows missing a start or end are skipped. Each complete row is validated
    (``start < end`` and both within the recording timespan); an invalid row raises
    ``ValueError`` so nothing is written for any site.
    """
    windows: list[tuple[float, float]] = []
    low, high = float(timestamps[0]), float(timestamps[-1])
    for _, row in df.iterrows():
        if pd.isna(row["start"]) or pd.isna(row["end"]):
            continue
        start, end = float(row["start"]), float(row["end"])
        if start >= end:
            raise ValueError(f"Artifact window start={start} must be less than end={end}.")
        if start < low or end > high:
            raise ValueError(
                f"Artifact window [{start}, {end}] is outside the recording timespan [{low}, {high}]; "
                "choose values within this range."
            )
        windows.append((start, end))
    return windows


def _windows_to_coords(windows: list[tuple[float, float]]) -> np.ndarray:
    """Flatten good-chunk windows into the interleaved ``(2M, 2)`` array ``fetchCoords`` expects.

    Column 0 holds ``[s0, e0, s1, e1, …]`` (which ``fetchCoords`` reshapes to
    ``[[s0, e0], …]``); column 1 is an unused placeholder (was the click y-coordinate
    in the legacy widget).
    """
    rows = []
    for start, end in windows:
        rows.append([start, 0.0])
        rows.append([end, 0.0])
    return np.array(rows)


def _empty_windows_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": [""] * _DEFAULT_WINDOW_ROWS,
            "start": [np.nan] * _DEFAULT_WINDOW_ROWS,
            "end": [np.nan] * _DEFAULT_WINDOW_ROWS,
        }
    )


class ArtifactRemovalConfig:
    """Interactive artifact-marking page for one run folder across all channel pairs.

    Shows the selected recording site's control/signal/fit traces with the current
    good-chunk windows shaded, an editable ``(label, start, end)`` table, and a
    z-score/dF-F review of the same site. On save, writes one
    ``coordsForPreProcessing_<site>.npy`` per site with defined windows.
    """

    def __init__(
        self,
        filepath: str,
        pair_traces: dict[str, dict[str, object]],
        preprocessed_traces: dict[str, dict[str, np.ndarray]],
        signals: list[str],
        on_done: Callable[[], None] | None = None,
    ) -> None:
        self.filepath = filepath
        self.pair_traces = pair_traces
        self.preprocessed_traces = preprocessed_traces
        self.sites = list(pair_traces.keys())
        # Called after a successful save so the serving loop can stop and the
        # preprocessing subprocess can advance to the next step.
        self.on_done = on_done

        self.signals = signals
        self.site_to_widget = {
            site: pn.widgets.Tabulator(_empty_windows_df(), show_index=False, widths=180) for site in self.sites
        }
        self.site_select = pn.widgets.Select(name="Recording site", options=self.sites, value=self.sites[0])
        self.save_button = pn.widgets.Button(name="Save and continue", button_type="primary")

        self.marking_pane = pn.pane.HoloViews(self._make_marking_plot(), width=800)
        self.table_container = pn.Row(self.site_to_widget[self.sites[0]])

        self.site_select.param.watch(self._refresh, "value")
        for widget in self.site_to_widget.values():
            widget.param.watch(self._refresh_marking, "value")
        self.save_button.on_click(self._on_save)

        sections = [
            "# Artifact Removal — {}".format(os.path.basename(filepath)),
            pn.pane.Markdown(
                "Define good-chunk windows to KEEP for each recording site: type exact "
                "start/end times (seconds) in the table and the shaded spans update to match. "
                "Data outside the windows is treated as artifact. Leave the table empty to keep "
                "the entire recording. Click **Save and continue** to proceed."
            ),
            self.site_select,
            self.marking_pane,
            self.table_container,
        ]

        # The z-score/dF-F review section only appears when the user requested it
        # (plot_zScore_dff != "None"); otherwise no such figure was ever created.
        if self.signals:
            self.signal_toggle = pn.widgets.RadioButtonGroup(name="Signal", options=signals, value=signals[0])
            self.review_pane = pn.pane.HoloViews(self._make_review_plot(), width=800)
            self.signal_toggle.param.watch(self._refresh_review, "value")
            sections += [pn.pane.Markdown("### z-score / dF-F review"), self.signal_toggle, self.review_pane]
        else:
            self.signal_toggle = None
            self.review_pane = None

        sections.append(self.save_button)
        self.widget = pn.Column(*sections)

    def _windows_for(self, site: str) -> list[tuple[float, float]]:
        return _windows_from_df(self.site_to_widget[site].value, self.pair_traces[site]["x"])

    def _make_marking_plot(self) -> hv.Layout:
        site = self.site_select.value
        trace = self.pair_traces[site]
        # Skip validation for live preview; only complete rows shade.
        df = self.site_to_widget[site].value
        windows = [
            (float(row["start"]), float(row["end"]))
            for _, row in df.iterrows()
            if not pd.isna(row["start"]) and not pd.isna(row["end"])
        ]
        return build_control_signal_fit(
            x=trace["x"],
            control=trace["control"],
            signal=trace["signal"],
            fit=trace["fit"],
            titles=trace["plot_name"],
            suptitle=os.path.basename(self.filepath),
            artifacts_have_been_removed=False,
            windows=windows,
        )

    def _make_review_plot(self) -> hv.Curve:
        site = self.site_select.value
        trace = self.preprocessed_traces[site]
        y = trace["y_zscore"] if self.signal_toggle.value == "z_score" else trace["y_dff"]
        return build_preprocessing_curve(
            suptitle=os.path.basename(self.filepath), title=f"{self.signal_toggle.value}_{site}", x=trace["x"], y=y
        )

    def _refresh_marking(self, event: object) -> None:
        self.marking_pane.object = self._make_marking_plot()

    def _refresh_review(self, event: object) -> None:
        self.review_pane.object = self._make_review_plot()

    def _refresh(self, event: object) -> None:
        self.table_container[:] = [self.site_to_widget[self.site_select.value]]
        self._refresh_marking(event)
        if self.review_pane is not None:
            self._refresh_review(event)

    def save(self) -> None:
        """Validate every site's windows, then write ``coordsForPreProcessing_<site>.npy`` per site.

        All sites are validated before anything is written, so an invalid window
        raises up-front without leaving a partially-written set of coords files. A
        site with no complete windows writes no file (keep-the-entire-recording default).
        """
        to_write = {}
        for site, widget in self.site_to_widget.items():
            windows = _windows_from_df(widget.value, self.pair_traces[site]["x"])
            if windows:
                to_write[site] = windows

        for site, windows in to_write.items():
            coords = _windows_to_coords(windows)
            np.save(os.path.join(self.filepath, "coordsForPreProcessing_" + site + ".npy"), coords)
            logger.info(f"Saved {len(windows)} artifact-removal window(s) for recording site {site}.")

    def _on_save(self, event: object) -> None:
        try:
            self.save()
        except ValueError as error:
            logger.error(str(error))
            if pn.state.notifications is not None:
                pn.state.notifications.error(str(error), duration=0)
            return
        if pn.state.notifications is not None:
            pn.state.notifications.success("Artifact-removal windows saved.", duration=4000)
        # Release the serving loop only after a clean save, so an invalid window
        # keeps the page open instead of advancing the pipeline.
        if self.on_done is not None:
            self.on_done()


class PreprocessingReviewView:
    """Read-only z-score / dF-F review for one run folder, with a site + signal selector."""

    def __init__(
        self, filepath: str, preprocessed_traces: dict[str, dict[str, np.ndarray]], signals: list[str]
    ) -> None:
        self.filepath = filepath
        self.preprocessed_traces = preprocessed_traces
        self.sites = list(preprocessed_traces.keys())

        self.site_select = pn.widgets.Select(name="Recording site", options=self.sites, value=self.sites[0])
        self.signal_toggle = pn.widgets.RadioButtonGroup(name="Signal", options=signals, value=signals[0])
        self.plot_pane = pn.pane.HoloViews(self._make_plot(), width=800)

        self.site_select.param.watch(self._refresh, "value")
        self.signal_toggle.param.watch(self._refresh, "value")

        self.widget = pn.Column(
            "## z-score / dF-F — {}".format(os.path.basename(filepath)),
            pn.Row(self.site_select, self.signal_toggle),
            self.plot_pane,
        )

    def _make_plot(self) -> hv.Curve:
        site = self.site_select.value
        trace = self.preprocessed_traces[site]
        y = trace["y_zscore"] if self.signal_toggle.value == "z_score" else trace["y_dff"]
        return build_preprocessing_curve(
            suptitle=os.path.basename(self.filepath), title=f"{self.signal_toggle.value}_{site}", x=trace["x"], y=y
        )

    def _refresh(self, event: object) -> None:
        self.plot_pane.object = self._make_plot()


class ArtifactReviewView:
    """Read-only control/signal/fit review after artifact removal, with saved windows shaded."""

    def __init__(self, filepath: str, pair_traces: dict[str, dict[str, object]]) -> None:
        self.filepath = filepath
        self.pair_traces = pair_traces
        self.sites = list(pair_traces.keys())

        self.site_select = pn.widgets.Select(name="Recording site", options=self.sites, value=self.sites[0])
        self.plot_pane = pn.pane.HoloViews(self._make_plot(), width=800)
        self.site_select.param.watch(self._refresh, "value")

        self.widget = pn.Column(
            "## Artifact removal review — {}".format(os.path.basename(filepath)),
            self.site_select,
            self.plot_pane,
        )

    def _make_plot(self) -> hv.Layout:
        site = self.site_select.value
        trace = self.pair_traces[site]
        coords = fetchCoords(self.filepath, site, trace["x"])
        windows = [(float(start), float(end)) for start, end in coords]
        return build_control_signal_fit(
            x=trace["x"],
            control=trace["control"],
            signal=trace["signal"],
            fit=trace["fit"],
            titles=trace["plot_name"],
            suptitle=os.path.basename(self.filepath),
            artifacts_have_been_removed=True,
            windows=windows,
        )

    def _refresh(self, event: object) -> None:
        self.plot_pane.object = self._make_plot()


def _continue_button(on_done: Callable[[], None] | None) -> pn.widgets.Button:
    button = pn.widgets.Button(name="Continue", button_type="primary")
    if on_done is not None:
        button.on_click(lambda event: on_done())
    return button


def build_artifact_removal_template(
    filepath: str, plot_zScore_dff: str, on_done: Callable[[], None] | None = None
) -> pn.template.BootstrapTemplate:
    """Build (without serving) the interactive artifact-marking page for one run folder."""
    template = pn.template.BootstrapTemplate(title="Artifact Removal - {}".format(os.path.basename(filepath)))
    config = ArtifactRemovalConfig(
        filepath,
        load_pair_traces(filepath),
        load_preprocessed_traces(filepath),
        signal_options(plot_zScore_dff),
        on_done=on_done,
    )
    template.main.append(config.widget)
    template._config = config  # test hook
    return template


def build_preprocessing_review_template(
    filepath: str, plot_zScore_dff: str, on_done: Callable[[], None] | None = None
) -> pn.template.BootstrapTemplate:
    """Build (without serving) the read-only z-score / dF-F review page for one run folder."""
    template = pn.template.BootstrapTemplate(title="z-score / dF-F - {}".format(os.path.basename(filepath)))
    view = PreprocessingReviewView(filepath, load_preprocessed_traces(filepath), signal_options(plot_zScore_dff))
    button = _continue_button(on_done)
    template.main.append(pn.Column(view.widget, button))
    template._view = view  # test hook
    template._continue_button = button
    return template


def build_artifact_review_template(
    filepath: str, on_done: Callable[[], None] | None = None
) -> pn.template.BootstrapTemplate:
    """Build (without serving) the read-only post-artifact-removal review page for one run folder."""
    template = pn.template.BootstrapTemplate(title="Artifact removal review - {}".format(os.path.basename(filepath)))
    view = ArtifactReviewView(filepath, load_pair_traces(filepath))
    button = _continue_button(on_done)
    template.main.append(pn.Column(view.widget, button))
    template._view = view  # test hook
    template._continue_button = button
    return template

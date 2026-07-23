"""Panel page for defining tonic/basal analysis epoch windows.

Tonic analysis averages the preprocessed z-score / dF/F trace over user-defined
absolute-time windows. Because injection timing can differ per recording site
(e.g. an ICV injection reaches sites at different times), windows are defined
per recording site. This page shows each site's preprocessed trace alongside an
editable ``(label, start, end)`` table so the user can see where to place a
window and type its exact bounds — the Panel-native replacement for the legacy
matplotlib popups. On save it writes one ``tonic_epochs_<site>.csv`` per site
into the run folder, which the preprocessing step then consumes.
"""

import glob
import logging
import os

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn

from .frontend_utils import scanPortsAndFind
from ..analysis.io_utils import read_hdf5, recording_site_from_preprocessed_label
from ..analysis.standard_io import read_tonic_epochs
from ..analysis.tonic import TONIC_EPOCH_COLUMNS, validate_tonic_epochs
from ..utils.utils import select_run_folders

# The preprocessing subprocess that serves this page never runs home.py, so load
# the Panel and HoloViews (bokeh) extensions here: without the bokeh backend,
# applying ``.opts()`` to the trace curves raises "No plotting extension loaded".
# ``notifications=True`` matches home.py so validation errors surface as toasts.
pn.extension(notifications=True)
hv.extension("bokeh")

logger = logging.getLogger(__name__)

_DEFAULT_EPOCH_ROWS = 3


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


def _empty_epoch_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": [""] * _DEFAULT_EPOCH_ROWS,
            "start": [np.nan] * _DEFAULT_EPOCH_ROWS,
            "end": [np.nan] * _DEFAULT_EPOCH_ROWS,
        }
    )


class TonicEpochConfig:
    """Per-run-folder editor for tonic epoch windows across all recording sites.

    Renders the selected site's preprocessed trace with the current epoch windows
    overlaid as shaded spans, plus an editable ``(label, start, end)`` table. A
    signal toggle switches the displayed trace between z-score and dF/F; a
    "copy to all sites" button replicates the current table to every site; and
    save writes one ``tonic_epochs_<site>.csv`` per site into the run folder.
    """

    def __init__(self, filepath: str, site_traces: dict[str, dict[str, np.ndarray]]) -> None:
        self.filepath = filepath
        self.site_traces = site_traces
        self.sites = list(site_traces.keys())

        self.site_to_widget = {
            site: pn.widgets.Tabulator(_empty_epoch_df(), show_index=False, widths=180) for site in self.sites
        }

        self.signal_toggle = pn.widgets.RadioButtonGroup(name="Signal", options=["z_score", "dff"], value="z_score")
        self.site_select = pn.widgets.Select(name="Recording site", options=self.sites, value=self.sites[0])
        self.copy_to_all_button = pn.widgets.Button(name="Copy windows to all sites", button_type="default")
        self.save_button = pn.widgets.Button(name="Save epoch windows", button_type="primary")

        self.plot_pane = pn.pane.HoloViews(self._make_plot(), width=750)

        self.signal_toggle.param.watch(self._refresh_plot, "value")
        self.site_select.param.watch(self._refresh_plot, "value")
        for widget in self.site_to_widget.values():
            widget.param.watch(self._refresh_plot, "value")
        self.copy_to_all_button.on_click(self._on_copy_to_all)
        self.save_button.on_click(self._on_save)

        self.widget = pn.Column(
            "# Tonic Epochs — {}".format(os.path.basename(filepath)),
            pn.pane.Markdown(
                "Define named epoch windows on each recording site's preprocessed trace. "
                "Type exact start/end times (seconds); the shaded spans update to match. "
                "Use **Copy windows to all sites** when the injection is systemic."
            ),
            pn.Row(self.site_select, self.signal_toggle),
            self.plot_pane,
            self._active_table_row(),
            pn.Row(self.copy_to_all_button, self.save_button),
        )

    def _active_table_row(self) -> pn.Row:
        return pn.Row(self.site_to_widget[self.site_select.value])

    def _epoch_spans(self, site: str) -> "hv.Overlay | hv.VSpan | None":
        overlay = None
        df = self.site_to_widget[site].value
        for _, row in df.iterrows():
            if pd.isna(row["start"]) or pd.isna(row["end"]):
                continue
            span = hv.VSpan(float(row["start"]), float(row["end"])).opts(color="orange", alpha=0.2)
            overlay = span if overlay is None else overlay * span
        return overlay

    def _make_plot(self) -> "hv.Overlay | hv.Curve":
        site = self.site_select.value
        trace = self.site_traces[site]
        y = trace["y_zscore"] if self.signal_toggle.value == "z_score" else trace["y_dff"]
        curve = hv.Curve((trace["x"], y), "time (s)", self.signal_toggle.value).opts(width=750, height=320)
        spans = self._epoch_spans(site)
        return curve if spans is None else curve * spans

    def _refresh_plot(self, event: object) -> None:
        self.widget[4] = self._active_table_row()
        self.plot_pane.object = self._make_plot()

    def _on_copy_to_all(self, event: object) -> None:
        source = self.site_to_widget[self.site_select.value].value.copy()
        for site, widget in self.site_to_widget.items():
            widget.value = source.copy()
        self.plot_pane.object = self._make_plot()

    def save(self) -> None:
        """Validate and write ``tonic_epochs_<site>.csv`` for every site with a complete window.

        Every site's windows are validated (against that site's own timespan)
        before anything is written, so an invalid window raises up-front without
        leaving a partially-written set of epoch files.
        """
        to_write = {}
        for site, widget in self.site_to_widget.items():
            df = widget.value
            complete = df[df["start"].notna() & df["end"].notna()][TONIC_EPOCH_COLUMNS]
            if complete.empty:
                continue
            timestamps = self.site_traces[site]["x"]
            validate_tonic_epochs(complete, float(timestamps[0]), float(timestamps[-1]))
            to_write[site] = complete

        for site, complete in to_write.items():
            complete.to_csv(os.path.join(self.filepath, "tonic_epochs_" + site + ".csv"), index=False)
            logger.info(f"Saved {len(complete)} tonic epoch(s) for recording site {site}.")

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


def build_tonic_epoch_template(filepath: str) -> pn.template.BootstrapTemplate:
    """Build (without serving) the tonic epoch-definition page for a single run folder."""
    template = pn.template.BootstrapTemplate(title="Tonic Epochs - {}".format(os.path.basename(filepath)))
    config = TonicEpochConfig(filepath, load_site_traces(filepath))
    template.main.append(config.widget)
    template._config = config  # test hook
    return template


def define_tonic_epochs(inputParameters: dict[str, object], session_folders: list) -> None:
    """Open the tonic epoch-definition page for each selected run folder.

    Iterates the run folders exactly like ``visualize_z_score`` and serves one
    blocking Panel page per folder so the user can define epoch windows on the
    preprocessed traces. Each page writes ``tonic_epochs_<site>.csv`` on save.
    """
    combine_data = inputParameters["combine_data"]
    run_folders = []
    for i in range(len(session_folders)):
        if combine_data == True:
            run_folders.append([session_folders[i][0]])
        else:
            filepath = session_folders[i]
            run_folders.append(select_run_folders(filepath, (inputParameters.get("selected_runs") or {}).get(filepath)))
    run_folders = np.concatenate(run_folders)

    for j in range(len(run_folders)):
        template = build_tonic_epoch_template(run_folders[j])
        number = scanPortsAndFind(start_port=5000, end_port=5200)
        template.show(port=number)


def _tonic_result_sites(filepath: str) -> list[str]:
    sites = []
    for path in sorted(glob.glob(os.path.join(filepath, "tonic_*.h5"))):
        basename = os.path.basename(path)
        sites.append(basename[len("tonic_") : -len(".h5")])
    return sites


class TonicResultsView:
    """Read-only view of tonic results for a single run folder.

    For each recording site it shows the preprocessed trace with the analysed
    epoch windows shaded, and a table of per-epoch means. Because the difference
    from baseline is a viewing choice, a baseline-epoch selector adds live
    ``diff_zscore`` / ``diff_dff`` columns computed against the chosen epoch.
    """

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.sites = _tonic_result_sites(filepath)

        self.site_select = pn.widgets.Select(name="Recording site", options=self.sites, value=self.sites[0])
        self.signal_toggle = pn.widgets.RadioButtonGroup(name="Signal", options=["z_score", "dff"], value="z_score")
        means = self._means(self.sites[0])
        self.baseline_select = pn.widgets.Select(
            name="Baseline epoch", options=list(means.index), value=list(means.index)[0]
        )
        self.plot_pane = pn.pane.HoloViews(self._make_plot(), width=750)
        self.table_pane = pn.pane.DataFrame(self._means_with_diff(), width=520)

        self.site_select.param.watch(self._on_site_change, "value")
        self.signal_toggle.param.watch(self._refresh, "value")
        self.baseline_select.param.watch(self._refresh, "value")

        self.widget = pn.Column(
            "## Tonic / basal analysis — {}".format(os.path.basename(filepath)),
            pn.Row(self.site_select, self.signal_toggle, self.baseline_select),
            self.plot_pane,
            self.table_pane,
        )

    def _means(self, site: str) -> pd.DataFrame:
        return pd.read_hdf(os.path.join(self.filepath, "tonic_" + site + ".h5"), key="df")

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

    def _make_plot(self) -> "hv.Overlay | hv.Curve":
        site = self.site_select.value
        timestamps = np.asarray(read_hdf5("timeCorrection_" + site, self.filepath, "timestampNew")).ravel()
        signal = "z_score" if self.signal_toggle.value == "z_score" else "dff"
        y = np.asarray(read_hdf5("", os.path.join(self.filepath, signal + "_" + site + ".hdf5"), "data")).ravel()
        curve = hv.Curve((timestamps, y), "time (s)", self.signal_toggle.value).opts(width=750, height=320)

        overlay = None
        for _, row in read_tonic_epochs(self.filepath, site).iterrows():
            span = hv.VSpan(float(row["start"]), float(row["end"])).opts(color="orange", alpha=0.2)
            overlay = span if overlay is None else overlay * span
        return curve if overlay is None else curve * overlay

    def _on_site_change(self, event: object) -> None:
        means = self._means(self.site_select.value)
        self.baseline_select.options = list(means.index)
        self.baseline_select.value = list(means.index)[0]
        self._refresh(event)

    def _refresh(self, event: object) -> None:
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

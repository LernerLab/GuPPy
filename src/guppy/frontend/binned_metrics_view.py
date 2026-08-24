"""Step-5 dashboard tab for whole-session time-binned metrics.

Reads the ``binned_metrics_<site>.h5`` tables Step 4 wrote and plots the selected
metric as bars under the trace it was reduced from. Sessions analysed without the
binned metrics parameter have no such tables, so the tab renders a short note
instead and can be added to the dashboard unconditionally.
"""

import glob
import logging
import os

import holoviews as hv
import numpy as np
import panel as pn

from ..analysis.io_utils import read_hdf5
from ..analysis.standard_io import read_binned_metrics_from_hdf5
from ..visualization.binned_metrics import build_binned_metrics_panel

logger = logging.getLogger(__name__)

_FILE_PREFIX = "binned_metrics_"

# Column -> (menu label, axis label, the trace the metric was reduced from).
METRICS = {
    "mean_zscore": ("Mean z-score", "mean z-score", "z_score"),
    "mean_dff": ("Mean ΔF/F", "mean ΔF/F", "dff"),
    "transient_count_z_score": ("Transient count (z-score)", "transients per bin", "z_score"),
    "transient_count_dff": ("Transient count (ΔF/F)", "transients per bin", "dff"),
}

TRACE_LABELS = {"z_score": "z-score", "dff": "ΔF/F"}


def binned_metrics_sites(filepath: str) -> list[str]:
    """Return the recording sites with binned metrics in ``filepath``, sorted.

    Parameters
    ----------
    filepath : str
        Session output directory.

    Returns
    -------
    list of str
        Recording site names, empty when the session has no binned metrics.
    """
    paths = glob.glob(os.path.join(filepath, _FILE_PREFIX + "*.h5"))
    return sorted(os.path.basename(path)[len(_FILE_PREFIX) : -len(".h5")] for path in paths)


class BinnedMetricsView:
    """Site and metric selectors over the binned metrics of one session.

    Parameters
    ----------
    filepath : str
        Session output directory holding the ``binned_metrics_<site>.h5`` tables.
    """

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.sites = binned_metrics_sites(filepath)

        self.site_select = pn.widgets.Select(name="Recording site", options=self.sites, value=self.sites[0], width=220)
        self.metric_select = pn.widgets.Select(name="Metric", options=self._metric_options(self.sites[0]), width=220)
        self.plot_pane = pn.pane.HoloViews(self._make_plot(), sizing_mode="stretch_width")

        self.site_select.param.watch(self._on_site_change, "value")
        self.metric_select.param.watch(self._on_metric_change, "value")

    def _metric_options(self, site: str) -> dict[str, str]:
        """Menu label -> column, for the metric columns this site actually has."""
        columns = read_binned_metrics_from_hdf5(self.filepath, site).columns
        return {METRICS[column][0]: column for column in METRICS if column in columns}

    def _make_plot(self) -> hv.Layout:
        site = self.site_select.value
        column = self.metric_select.value
        _, value_label, trace_name = METRICS[column]

        binned_metrics = read_binned_metrics_from_hdf5(self.filepath, site)
        timestamps = np.asarray(read_hdf5("timeCorrection_" + site, self.filepath, "timestampNew")).ravel()
        trace = np.asarray(read_hdf5(trace_name + "_" + site, self.filepath, "data")).ravel()

        return build_binned_metrics_panel(
            timestamps=timestamps,
            trace=trace,
            trace_label=TRACE_LABELS[trace_name],
            bin_starts=binned_metrics["bin_start"].to_numpy(),
            bin_ends=binned_metrics["bin_end"].to_numpy(),
            values=binned_metrics[column].to_numpy(),
            value_label=value_label,
            suptitle=site,
        )

    def _on_site_change(self, event: object) -> None:
        # A different site may have been analysed with a different transient
        # selection, so the metric menu is rebuilt before the plot is redrawn.
        self.metric_select.options = self._metric_options(self.site_select.value)
        self.plot_pane.object = self._make_plot()

    def _on_metric_change(self, event: object) -> None:
        self.plot_pane.object = self._make_plot()

    @property
    def widget(self) -> pn.Column:
        """The composed selectors and plot."""
        return pn.Column(pn.Row(self.site_select, self.metric_select), self.plot_pane)


def build_binned_metrics_view(filepath: str) -> pn.Column:
    """Build the Binned tab for one session.

    Parameters
    ----------
    filepath : str
        Session output directory.

    Returns
    -------
    pn.Column
        The selectors and plot, or a short note when the session was analysed
        without binned metrics.
    """
    if not binned_metrics_sites(filepath):
        return pn.Column(pn.pane.Markdown("_No binned metrics in this session._"))

    return BinnedMetricsView(filepath).widget

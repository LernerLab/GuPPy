"""Step-5 dashboard tab for behavioral covariates correlated with binned metrics.

Reads the ``covariate_correlations_<site>.h5`` and ``binned_covariates_<site>.h5``
tables Step 4 wrote and shows the selected covariate two ways: stacked with the
photometry trace and bins it was correlated against, on one linked time axis, and
scattered bin-by-bin against the selected metric. Sessions without a behavioral
covariate store have no such tables, so the tab renders a short note instead and can
be added to the dashboard unconditionally.
"""

import logging
from pathlib import Path

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn

from .binned_metrics_view import METRICS, TRACE_LABELS
from ..analysis.io_utils import read_hdf5
from ..analysis.standard_io import (
    read_binned_covariates_from_hdf5,
    read_binned_metrics_from_hdf5,
    read_covariate_correlations_from_hdf5,
    read_covariate_series,
)
from ..visualization.binned_metrics import build_binned_metrics_panel
from ..visualization.covariate_correlation import (
    build_covariate_panel,
    build_covariate_scatter,
)

logger = logging.getLogger(__name__)

_FILE_PREFIX = "covariate_correlations_"

# Four stacked panels share the tab, so each is shorter than the two the Binned tab stacks.
_PANEL_HEIGHT = 180


def covariate_correlation_sites(filepath: str) -> list[str]:
    """Return the recording sites with covariate correlations in ``filepath``, sorted.

    Parameters
    ----------
    filepath : str
        Session output directory.

    Returns
    -------
    list of str
        Recording site names, empty when the session has no covariate correlations.
    """
    paths = Path(filepath).glob(_FILE_PREFIX + "*.h5")
    return sorted(path.name[len(_FILE_PREFIX) : -len(".h5")] for path in paths)


class CovariateCorrelationView:
    """Site, metric and covariate selectors over one session's covariate correlations.

    Parameters
    ----------
    filepath : str
        Session output directory holding the covariate tables.
    """

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.sites = covariate_correlation_sites(filepath)
        self.covariate_series = read_covariate_series(filepath)

        first_site = self.sites[0]
        self.site_select = pn.widgets.Select(name="Recording site", options=self.sites, value=first_site, width=220)
        self.metric_select = pn.widgets.Select(name="Metric", options=self._metric_options(first_site), width=260)
        self.covariate_select = pn.widgets.Select(
            name="Covariate", options=self._covariate_options(first_site), width=220
        )
        self.trace_pane = pn.pane.HoloViews(self._make_trace_layout(), sizing_mode="stretch_width")
        self.plot_pane = pn.pane.HoloViews(self._make_plot(), sizing_mode="stretch_width")
        self.table_pane = pn.pane.DataFrame(self._correlations_table(), index=False, width=640)

        self.site_select.param.watch(self._on_site_change, "value")
        self.metric_select.param.watch(self._on_selection_change, "value")
        self.covariate_select.param.watch(self._on_selection_change, "value")

    def _metric_options(self, site: str) -> dict[str, str]:
        """Menu label -> metric column, for the metrics this site actually has."""
        metrics = read_covariate_correlations_from_hdf5(filepath=self.filepath, recording_site=site)["metric"]
        return {METRICS[metric][0]: metric for metric in METRICS if metric in set(metrics)}

    def _covariate_options(self, site: str) -> list[str]:
        """The covariate names this site was correlated against."""
        correlations = read_covariate_correlations_from_hdf5(filepath=self.filepath, recording_site=site)
        return sorted(set(correlations["covariate"]))

    def _correlations_table(self) -> pd.DataFrame:
        """Every (metric, covariate) pair for the selected site."""
        return read_covariate_correlations_from_hdf5(filepath=self.filepath, recording_site=self.site_select.value)

    def _make_trace_layout(self) -> hv.Layout:
        """The photometry trace and metric bins above the covariate trace and its bins."""
        site = self.site_select.value
        metric = self.metric_select.value
        covariate = self.covariate_select.value
        _, value_label, trace_name = METRICS[metric]

        binned_metrics = read_binned_metrics_from_hdf5(self.filepath, site)
        binned_covariates = read_binned_covariates_from_hdf5(filepath=self.filepath, recording_site=site)
        bin_starts = binned_metrics["bin_start"].to_numpy()
        bin_ends = binned_metrics["bin_end"].to_numpy()

        timestamps = np.asarray(read_hdf5("timeCorrection_" + site, self.filepath, "timestampNew")).ravel()
        trace = np.asarray(read_hdf5(trace_name + "_" + site, self.filepath, "data")).ravel()

        metric_panel = build_binned_metrics_panel(
            timestamps=timestamps,
            trace=trace,
            trace_label=TRACE_LABELS[trace_name],
            bin_starts=bin_starts,
            bin_ends=bin_ends,
            values=binned_metrics[metric].to_numpy(),
            value_label=value_label,
            suptitle=site,
            panel_height=_PANEL_HEIGHT,
        )
        covariate_timestamps, covariate_values = self.covariate_series[covariate]
        covariate_panel = build_covariate_panel(
            covariate_timestamps=covariate_timestamps,
            covariate_values=covariate_values,
            bin_starts=bin_starts,
            bin_ends=bin_ends,
            binned_values=binned_covariates[covariate].to_numpy(),
            covariate_label=covariate,
            panel_height=_PANEL_HEIGHT,
        )

        return hv.Layout(list(metric_panel) + list(covariate_panel)).cols(1)

    def _make_plot(self) -> hv.Overlay:
        site = self.site_select.value
        metric = self.metric_select.value
        covariate = self.covariate_select.value

        correlations = read_covariate_correlations_from_hdf5(filepath=self.filepath, recording_site=site)
        selected = correlations[(correlations["metric"] == metric) & (correlations["covariate"] == covariate)].iloc[0]

        binned_metrics = read_binned_metrics_from_hdf5(self.filepath, site)
        binned_covariates = read_binned_covariates_from_hdf5(filepath=self.filepath, recording_site=site)

        return build_covariate_scatter(
            covariate_values=binned_covariates[covariate].to_numpy(),
            metric_values=binned_metrics[metric].to_numpy(),
            covariate_label=covariate,
            metric_label=METRICS[metric][1],
            pearson_r=selected["pearson_r"],
            n_bins=int(selected["n_bins"]),
            suptitle=site,
        )

    def _on_site_change(self, event: object) -> None:
        # A different site may have been analysed with a different transient
        # selection, so both menus are rebuilt before the plots are redrawn.
        self.metric_select.options = self._metric_options(self.site_select.value)
        self.covariate_select.options = self._covariate_options(self.site_select.value)
        self._redraw()
        self.table_pane.object = self._correlations_table()

    def _on_selection_change(self, event: object) -> None:
        self._redraw()

    def _redraw(self) -> None:
        self.trace_pane.object = self._make_trace_layout()
        self.plot_pane.object = self._make_plot()

    @property
    def widget(self) -> pn.Column:
        """The composed selectors, stacked traces, scatter and correlations table."""
        return pn.Column(
            pn.Row(self.site_select, self.metric_select, self.covariate_select),
            self.trace_pane,
            self.plot_pane,
            self.table_pane,
        )


def build_covariate_correlation_view(filepath: str) -> pn.Column:
    """Build the Covariates tab for one session.

    Parameters
    ----------
    filepath : str
        Session output directory.

    Returns
    -------
    pn.Column
        The selectors, plots and table, or a short note when the session carried
        no behavioral covariate.
    """
    if not covariate_correlation_sites(filepath):
        return pn.Column(pn.pane.Markdown("_No behavioral covariates in this session._"))

    return CovariateCorrelationView(filepath).widget

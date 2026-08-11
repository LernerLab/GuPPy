"""Step-5 dashboard tab for behavioral covariates correlated with binned metrics.

Reads the ``covariate_correlations_<site>.h5`` and ``binned_covariates_<site>.h5``
tables Step 4 wrote and scatters the selected metric against the selected covariate.
Sessions without a behavioral covariate store have no such tables, so the tab renders
a short note instead and can be added to the dashboard unconditionally.
"""

import glob
import logging
import os

import holoviews as hv
import pandas as pd
import panel as pn

from ..analysis.standard_io import (
    read_binned_covariates_from_hdf5,
    read_binned_metrics_from_hdf5,
    read_covariate_correlations_from_hdf5,
)
from ..visualization.covariate_correlation import build_covariate_scatter

logger = logging.getLogger(__name__)

_FILE_PREFIX = "covariate_correlations_"

# Metric column -> axis label.
_METRIC_LABELS = {
    "mean_zscore": "mean z-score",
    "mean_dff": "mean ΔF/F",
    "transient_count_z_score": "transients per bin (z-score)",
    "transient_count_dff": "transients per bin (ΔF/F)",
}

NO_P_VALUE_NOTE = (
    "_Pearson r and Spearman rho are descriptive only. No p-value is reported, and one computed by "
    "hand from these numbers would not be valid either: the photometry metric and the behavioral "
    "score are both autocorrelated across bins, which breaks the independence the usual significance "
    "tests assume._"
)


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
    paths = glob.glob(os.path.join(filepath, _FILE_PREFIX + "*.h5"))
    return sorted(os.path.basename(path)[len(_FILE_PREFIX) : -len(".h5")] for path in paths)


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

        first_site = self.sites[0]
        self.site_select = pn.widgets.Select(name="Recording site", options=self.sites, value=first_site, width=220)
        self.metric_select = pn.widgets.Select(name="Metric", options=self._metric_options(first_site), width=260)
        self.covariate_select = pn.widgets.Select(
            name="Covariate", options=self._covariate_options(first_site), width=220
        )
        self.plot_pane = pn.pane.HoloViews(self._make_plot(), sizing_mode="stretch_width")
        self.table_pane = pn.pane.DataFrame(self._correlations_table(), index=False, width=640)

        self.site_select.param.watch(self._on_site_change, "value")
        self.metric_select.param.watch(self._on_selection_change, "value")
        self.covariate_select.param.watch(self._on_selection_change, "value")

    def _metric_options(self, site: str) -> dict[str, str]:
        """Menu label -> metric column, for the metrics this site actually has."""
        metrics = read_covariate_correlations_from_hdf5(filepath=self.filepath, recording_site=site)["metric"]
        return {_METRIC_LABELS.get(metric, metric): metric for metric in sorted(set(metrics))}

    def _covariate_options(self, site: str) -> list[str]:
        """The covariate names this site was correlated against."""
        correlations = read_covariate_correlations_from_hdf5(filepath=self.filepath, recording_site=site)
        return sorted(set(correlations["covariate"]))

    def _correlations_table(self) -> pd.DataFrame:
        """Every (metric, covariate) pair for the selected site."""
        return read_covariate_correlations_from_hdf5(filepath=self.filepath, recording_site=self.site_select.value)

    def _make_plot(self) -> hv.Points:
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
            bin_numbers=binned_covariates.index.to_numpy(),
            covariate_label=covariate,
            metric_label=_METRIC_LABELS.get(metric, metric),
            pearson_r=selected["pearson_r"],
            spearman_rho=selected["spearman_rho"],
            n_bins=int(selected["n_bins"]),
            suptitle=site,
        )

    def _on_site_change(self, event: object) -> None:
        # A different site may have been analysed with a different transient
        # selection, so both menus are rebuilt before the plot is redrawn.
        self.metric_select.options = self._metric_options(self.site_select.value)
        self.covariate_select.options = self._covariate_options(self.site_select.value)
        self.plot_pane.object = self._make_plot()
        self.table_pane.object = self._correlations_table()

    def _on_selection_change(self, event: object) -> None:
        self.plot_pane.object = self._make_plot()

    @property
    def widget(self) -> pn.Column:
        """The composed selectors, scatter, note and correlations table."""
        return pn.Column(
            pn.Row(self.site_select, self.metric_select, self.covariate_select),
            self.plot_pane,
            pn.pane.Markdown(NO_P_VALUE_NOTE, width=640),
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
        The selectors, scatter and table, or a short note when the session carried
        no behavioral covariate.
    """
    if not covariate_correlation_sites(filepath):
        return pn.Column(pn.pane.Markdown("_No behavioral covariates in this session._"))

    return CovariateCorrelationView(filepath).widget

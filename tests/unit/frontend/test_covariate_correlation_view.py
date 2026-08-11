import numpy as np
import pandas as pd
import panel as pn
import pytest

from guppy.analysis.standard_io import (
    write_binned_covariates_to_hdf5,
    write_binned_metrics_to_hdf5,
    write_covariate_correlations_to_hdf5,
)
from guppy.frontend.covariate_correlation_view import (
    CovariateCorrelationView,
    build_covariate_correlation_view,
    covariate_correlation_sites,
)


def write_site_outputs(filepath, site, metrics=("mean_zscore",), covariates=("akinesia",)):
    """Write the Step-4 binned and covariate tables one site needs."""
    geometry = {"bin_start": [0.0, 4.0, 8.0], "bin_end": [4.0, 8.0, 10.0]}

    binned_metrics = pd.DataFrame({**geometry, "n_samples": [4, 4, 2]}, index=pd.RangeIndex(3, name="bin"))
    for metric in metrics:
        binned_metrics[metric] = [1.5, 5.5, 9.0]
    write_binned_metrics_to_hdf5(filepath, binned_metrics, site)

    binned_covariates = pd.DataFrame(geometry, index=pd.RangeIndex(3, name="bin"))
    for covariate in covariates:
        binned_covariates[covariate] = [2.0, 4.0, 10.0]
    for covariate in covariates:
        binned_covariates["n_samples_" + covariate] = [2, 1, 1]
    write_binned_covariates_to_hdf5(filepath=filepath, binned_covariates=binned_covariates, recording_site=site)

    correlations = pd.DataFrame(
        [
            {
                "metric": metric,
                "covariate": covariate,
                "pearson_r": 0.9493907,
                "spearman_rho": 1.0,
                "n_bins": 3,
            }
            for metric in metrics
            for covariate in covariates
        ],
        index=pd.RangeIndex(len(metrics) * len(covariates), name="pair"),
    )
    write_covariate_correlations_to_hdf5(filepath=filepath, correlations=correlations, recording_site=site)


@pytest.fixture
def session(tmp_path, panel_extension):
    write_site_outputs(str(tmp_path), "dms")
    write_site_outputs(str(tmp_path), "nac", metrics=("mean_zscore", "mean_dff"), covariates=("akinesia", "tremor"))
    return str(tmp_path)


class TestCovariateCorrelationSites:
    def test_returns_sites_sorted(self, session):
        assert covariate_correlation_sites(session) == ["dms", "nac"]

    def test_returns_empty_without_covariate_tables(self, tmp_path):
        assert covariate_correlation_sites(str(tmp_path)) == []

    def test_preserves_underscores_in_site_names(self, tmp_path, panel_extension):
        write_site_outputs(str(tmp_path), "left_hemisphere")

        assert covariate_correlation_sites(str(tmp_path)) == ["left_hemisphere"]


class TestCovariateCorrelationView:
    def test_selector_options(self, session):
        view = CovariateCorrelationView(session)

        assert view.site_select.options == ["dms", "nac"]
        assert view.site_select.value == "dms"
        assert view.covariate_select.options == ["akinesia"]

    def test_metric_menu_labels_are_human_readable(self, session):
        view = CovariateCorrelationView(session)

        assert view.metric_select.options == {"mean z-score": "mean_zscore"}

    def test_changing_site_rebuilds_both_menus(self, session):
        view = CovariateCorrelationView(session)

        view.site_select.value = "nac"

        assert view.metric_select.options == {"mean z-score": "mean_zscore", "mean ΔF/F": "mean_dff"}
        assert view.covariate_select.options == ["akinesia", "tremor"]

    def test_changing_covariate_redraws_the_plot(self, session):
        view = CovariateCorrelationView(session)
        view.site_select.value = "nac"
        before = view.plot_pane.object

        view.covariate_select.value = "tremor"

        assert view.plot_pane.object is not before

    def test_plot_uses_the_selected_covariate_and_metric(self, session):
        view = CovariateCorrelationView(session)

        np.testing.assert_allclose(view.plot_pane.object.dimension_values("akinesia"), [2.0, 4.0, 10.0])
        np.testing.assert_allclose(view.plot_pane.object.dimension_values("mean z-score"), [1.5, 5.5, 9.0])

    def test_widget_shows_the_full_correlations_table(self, session):
        view = CovariateCorrelationView(session)

        assert list(view.table_pane.object.columns) == [
            "metric",
            "covariate",
            "pearson_r",
            "spearman_rho",
            "n_bins",
        ]


class TestBuildCovariateCorrelationView:
    def test_renders_the_note_when_no_covariates(self, tmp_path, panel_extension):
        result = build_covariate_correlation_view(str(tmp_path))

        assert isinstance(result[0], pn.pane.Markdown)
        assert "No behavioral covariates" in result[0].object

    def test_states_that_no_p_value_is_reported(self, session):
        result = build_covariate_correlation_view(session)

        note = next(pane for pane in result if isinstance(pane, pn.pane.Markdown))
        assert "No p-value is reported" in note.object

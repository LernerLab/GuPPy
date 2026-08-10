import numpy as np
import pandas as pd
import panel as pn
import pytest

from guppy.analysis.io_utils import write_hdf5
from guppy.analysis.standard_io import write_binned_metrics_to_hdf5
from guppy.frontend.binned_metrics_view import (
    BinnedMetricsView,
    binned_metrics_sites,
    build_binned_metrics_view,
)


def write_site_outputs(filepath, site, count_columns=("transient_count_z_score",)):
    """Write the Step-3 traces and the Step-4 binned table one site needs."""
    timestamps = np.arange(0, 11, 1, dtype=float)
    write_hdf5(timestamps, "timeCorrection_" + site, filepath, "timestampNew")
    write_hdf5(timestamps.copy(), "z_score_" + site, filepath, "data")
    write_hdf5(timestamps / 10, "dff_" + site, filepath, "data")

    binned_metrics = pd.DataFrame(
        {
            "bin_start": [0.0, 5.0],
            "bin_end": [5.0, 10.0],
            "n_samples": [5, 6],
            "mean_zscore": [2.0, 7.5],
            "mean_dff": [0.2, 0.75],
        },
        index=pd.RangeIndex(2, name="bin"),
    )
    for column in count_columns:
        binned_metrics[column] = [1, 3]
    write_binned_metrics_to_hdf5(filepath, binned_metrics, site)


@pytest.fixture
def session(tmp_path, panel_extension):
    write_site_outputs(str(tmp_path), "dms")
    write_site_outputs(str(tmp_path), "nac")
    return str(tmp_path)


@pytest.fixture
def view(session):
    binned_metrics_view = BinnedMetricsView(session)
    # Pin the site so tests do not depend on glob ordering.
    binned_metrics_view.site_select.value = "dms"
    return binned_metrics_view


class TestBinnedMetricsSites:
    def test_lists_sites_sorted(self, session):
        assert binned_metrics_sites(session) == ["dms", "nac"]

    def test_empty_when_the_session_has_no_binned_metrics(self, tmp_path):
        assert binned_metrics_sites(str(tmp_path)) == []

    def test_recording_sites_with_underscores_survive(self, tmp_path, panel_extension):
        write_site_outputs(str(tmp_path), "left_hemisphere")

        assert binned_metrics_sites(str(tmp_path)) == ["left_hemisphere"]


class TestBinnedMetricsView:
    def test_offers_every_site(self, view):
        assert view.site_select.options == ["dms", "nac"]

    def test_metric_menu_covers_the_columns_present(self, view):
        assert view.metric_select.options == {
            "Mean z-score": "mean_zscore",
            "Mean ΔF/F": "mean_dff",
            "Transient count (z-score)": "transient_count_z_score",
        }

    def test_metric_menu_omits_counts_the_detector_did_not_produce(self, tmp_path, panel_extension):
        # Only z-score transients were computed, so there is no ΔF/F count column.
        write_site_outputs(str(tmp_path), "dms", count_columns=())
        binned_metrics_view = BinnedMetricsView(str(tmp_path))

        assert "Transient count (z-score)" not in binned_metrics_view.metric_select.options
        assert "Mean z-score" in binned_metrics_view.metric_select.options

    def test_metric_menu_covers_both_counts_when_both_were_computed(self, tmp_path, panel_extension):
        write_site_outputs(str(tmp_path), "dms", count_columns=("transient_count_z_score", "transient_count_dff"))
        binned_metrics_view = BinnedMetricsView(str(tmp_path))

        assert "Transient count (z-score)" in binned_metrics_view.metric_select.options
        assert "Transient count (ΔF/F)" in binned_metrics_view.metric_select.options

    def test_plots_the_selected_metric_values(self, view):
        view.metric_select.value = "mean_zscore"

        bars = list(view.plot_pane.object)[1].array()
        np.testing.assert_allclose(bars[:, 3], [2.0, 7.5])

    def test_switching_metric_redraws_with_the_new_values(self, view):
        view.metric_select.value = "mean_dff"

        bars = list(view.plot_pane.object)[1].array()
        np.testing.assert_allclose(bars[:, 3], [0.2, 0.75])

    def test_transient_count_metric_plots_the_counts(self, view):
        view.metric_select.value = "transient_count_z_score"

        bars = list(view.plot_pane.object)[1].array()
        np.testing.assert_allclose(bars[:, 3], [1, 3])

    def test_switching_site_redraws(self, view, session):
        # Give the second site distinguishable values so the redraw is observable.
        binned_metrics = pd.DataFrame(
            {
                "bin_start": [0.0, 5.0],
                "bin_end": [5.0, 10.0],
                "n_samples": [5, 6],
                "mean_zscore": [-1.0, -4.0],
                "mean_dff": [0.1, 0.2],
                "transient_count_z_score": [0, 1],
            },
            index=pd.RangeIndex(2, name="bin"),
        )
        write_binned_metrics_to_hdf5(session, binned_metrics, "nac")

        view.metric_select.value = "mean_zscore"
        view.site_select.value = "nac"

        bars = list(view.plot_pane.object)[1].array()
        np.testing.assert_allclose(bars[:, 3], [-1.0, -4.0])

    def test_widget_composes_selectors_and_plot(self, view):
        assert isinstance(view.widget, pn.Column)


class TestBuildBinnedMetricsView:
    def test_returns_the_view_when_results_exist(self, session):
        result = build_binned_metrics_view(session)

        assert isinstance(result, pn.Column)
        assert not isinstance(result[0], pn.pane.Markdown)

    def test_returns_an_empty_state_note_when_there_are_no_results(self, tmp_path, panel_extension):
        result = build_binned_metrics_view(str(tmp_path))

        assert isinstance(result[0], pn.pane.Markdown)
        assert "No binned metrics" in result[0].object

import os

import numpy as np
import pandas as pd
import panel as pn
import pytest

from guppy.analysis.io_utils import PSTH_SIGNIFICANCE_DIRNAME
from guppy.frontend.psth_significance_view import (
    PsthSignificanceView,
    build_psth_significance_view,
    describe_comparison,
    metric_label_for,
    significance_comparisons,
)


def write_significance(filepath, name, *, significant_from=None, significant_to=None, n=12, n_b=None, alpha=0.05):
    """Write one significance table into the results subdirectory."""
    results_path = os.path.join(filepath, PSTH_SIGNIFICANCE_DIRNAME)
    os.makedirs(results_path, exist_ok=True)

    timestamps = np.linspace(-1.0, 1.0, 11)
    significant = np.zeros(11, dtype=int)
    if significant_from is not None:
        significant[significant_from:significant_to] = 1

    table = pd.DataFrame(
        {
            "timestamps": timestamps,
            "estimate": np.linspace(0.0, 2.0, 11),
            "ci_lower": np.linspace(-0.5, 1.5, 11),
            "ci_upper": np.linspace(0.5, 2.5, 11),
            "significant": significant,
            "alpha": alpha,
            "n": n,
        }
    )
    if n_b is not None:
        table["n_b"] = n_b

    table.to_hdf(os.path.join(results_path, "significance_" + name + ".h5"), key="df", mode="w")


@pytest.fixture
def results_folder(tmp_path):
    """An output directory with one one-sample and one two-sample comparison."""
    folder = tmp_path / "session_output_1"
    folder.mkdir()
    write_significance(str(folder), "rewarded_dms_z_score_dms", significant_from=6, significant_to=9)
    write_significance(
        str(folder), "rewarded_vs_unrewarded_dms_z_score_dms", significant_from=2, significant_to=4, n_b=30
    )
    return str(folder)


class TestSignificanceComparisons:
    def test_lists_comparison_names_sorted(self, results_folder):
        assert significance_comparisons(results_folder) == [
            "rewarded_dms_z_score_dms",
            "rewarded_vs_unrewarded_dms_z_score_dms",
        ]

    def test_is_empty_without_results(self, tmp_path):
        assert significance_comparisons(str(tmp_path)) == []


class TestMetricLabelFor:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("rewarded_dms_z_score_dms", "z-score"),
            ("rewarded_vs_unrewarded_dms_dff_dms", "\u0394F/F"),
            # An event whose own label contains a metric token must not shadow the
            # basename, which is always the rightmost one.
            ("my_dff_event_dms_z_score_dms", "z-score"),
        ],
    )
    def test_names_the_metric_the_comparison_ran_on(self, name, expected):
        assert metric_label_for(name) == expected

    def test_falls_back_when_no_metric_token_is_present(self):
        assert metric_label_for("unrecognizable") == "signal"


class TestDescribeComparison:
    def test_one_sample_says_it_was_tested_against_zero(self):
        caption = describe_comparison(name="rewarded_dms_z_score_dms", n=48, n_b=None)

        assert "against zero" in caption
        assert "48" in caption

    def test_two_sample_names_both_counts_and_the_sign_convention(self):
        caption = describe_comparison(name="a_vs_b_dms_z_score_dms", n=48, n_b=179)

        assert "48" in caption and "179" in caption
        assert "first event is larger" in caption


class TestPsthSignificanceView:
    def test_selector_lists_every_comparison(self, results_folder, panel_extension):
        view = PsthSignificanceView(results_folder)

        assert view.comparison_select.options == [
            "rewarded_dms_z_score_dms",
            "rewarded_vs_unrewarded_dms_z_score_dms",
        ]

    def test_plots_the_significant_stretch_as_a_bar(self, results_folder, panel_extension):
        view = PsthSignificanceView(results_folder)

        # The overlay is band * curve * zero line * significance bar; the bar is last.
        bars = list(view.plot_pane.object)[-1].array()
        # Timepoints 6..8 of a -1..1 axis in 11 steps are 0.2 and 0.6.
        np.testing.assert_allclose(bars[0, 0], 0.2)
        np.testing.assert_allclose(bars[0, 2], 0.6)

    def test_changing_the_comparison_redraws_the_plot_and_caption(self, results_folder, panel_extension):
        view = PsthSignificanceView(results_folder)
        first_caption = view.caption.object

        view.comparison_select.value = "rewarded_vs_unrewarded_dms_z_score_dms"

        assert view.caption.object != first_caption
        assert "179" not in view.caption.object and "30" in view.caption.object
        # The second comparison is significant over timepoints 2..3, i.e. -0.6 to -0.4.
        bars = list(view.plot_pane.object)[-1].array()
        np.testing.assert_allclose(bars[0, 0], -0.6)

    def test_axis_label_names_the_metric_rather_than_both(self, results_folder, panel_extension):
        view = PsthSignificanceView(results_folder)

        # The one-sample file is a z-score comparison, so the axis says so outright.
        assert view.plot_pane.object.dimensions()[1].name == "z-score"

        view.comparison_select.value = "rewarded_vs_unrewarded_dms_z_score_dms"
        assert view.plot_pane.object.dimensions()[1].name == "difference in z-score"

    def test_legend_names_the_band_the_estimate_and_the_significance_bar(self, results_folder, panel_extension):
        view = PsthSignificanceView(results_folder)

        labels = [element.label for element in view.plot_pane.object]
        assert "mean PSTH" in labels
        assert "95% confidence interval" in labels
        assert "significant (alpha = 0.05)" in labels

    def test_legend_reports_the_alpha_the_comparison_ran_at(self, tmp_path, panel_extension):
        folder = tmp_path / "strict_output_1"
        folder.mkdir()
        write_significance(str(folder), "rewarded_dms_z_score_dms", significant_from=2, significant_to=6, alpha=0.01)

        view = PsthSignificanceView(str(folder))

        labels = [element.label for element in view.plot_pane.object]
        assert "99% confidence interval" in labels
        assert "significant (alpha = 0.01)" in labels

    def test_a_comparison_with_nothing_significant_draws_no_bars(self, tmp_path, panel_extension):
        folder = tmp_path / "quiet_output_1"
        folder.mkdir()
        write_significance(str(folder), "rewarded_dms_z_score_dms")

        view = PsthSignificanceView(str(folder))

        assert len(list(view.plot_pane.object)[-1].array()) == 0


class TestBuildPsthSignificanceView:
    def test_returns_the_selector_and_plot_when_results_exist(self, results_folder, panel_extension):
        result = build_psth_significance_view(results_folder)

        assert isinstance(result, pn.Column)
        assert isinstance(result[0], pn.widgets.Select)

    def test_returns_an_empty_state_note_when_there_are_no_results(self, tmp_path, panel_extension):
        result = build_psth_significance_view(str(tmp_path))

        assert isinstance(result[0], pn.pane.Markdown)
        assert "No PSTH significance results" in result[0].object

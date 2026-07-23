import numpy as np
import pandas as pd
import pytest

from guppy.analysis.standard_io import write_tonic_to_hdf5
from guppy.frontend.tonic_epochs import (
    TonicEpochConfig,
    TonicResultsView,
    build_tonic_results_view,
)
from guppy.utils._hdf5_io import write_hdf5


@pytest.fixture
def site_traces():
    timestamps = np.arange(0.0, 11.0, 1.0)
    return {
        "DMS": {"x": timestamps, "y_zscore": timestamps.copy(), "y_dff": timestamps / 10.0},
        "DLS": {"x": timestamps, "y_zscore": timestamps.copy() * 2, "y_dff": timestamps / 5.0},
    }


class TestTonicEpochConfig:
    @pytest.fixture
    def config(self, panel_extension, tmp_path, site_traces):
        return TonicEpochConfig(str(tmp_path), site_traces)

    def test_save_writes_csv_only_for_sites_with_complete_rows(self, config, tmp_path):
        config.site_to_widget["DMS"].value = pd.DataFrame(
            {
                "label": ["baseline", "post", ""],
                "start": [0.0, 8.0, np.nan],
                "end": [2.0, 10.0, np.nan],
            }
        )
        config.save()

        dms_path = tmp_path / "tonic_epochs_DMS.csv"
        assert dms_path.exists()
        saved = pd.read_csv(dms_path)
        # The incomplete trailing row is dropped; only the two complete windows persist.
        expected = pd.DataFrame({"label": ["baseline", "post"], "start": [0.0, 8.0], "end": [2.0, 10.0]})
        pd.testing.assert_frame_equal(saved, expected)

        # DLS was left empty, so no file is written for it.
        assert not (tmp_path / "tonic_epochs_DLS.csv").exists()

    def test_save_raises_on_window_not_overlapping_recording(self, config, tmp_path):
        # site traces span t in [0, 10]; a [15, 20] window overlaps nothing -> reject early,
        # before any file is written, so the worker never sees an invalid epoch.
        config.site_to_widget["DMS"].value = pd.DataFrame({"label": ["late"], "start": [15.0], "end": [20.0]})
        with pytest.raises(ValueError, match="does not overlap"):
            config.save()
        assert not (tmp_path / "tonic_epochs_DMS.csv").exists()

    def test_on_save_swallows_validation_error_without_writing(self, config, tmp_path):
        config.site_to_widget["DMS"].value = pd.DataFrame({"label": ["late"], "start": [15.0], "end": [20.0]})
        # _on_save catches the ValueError (surfacing it as a notification when served)
        # instead of propagating; no file is written.
        config._on_save(None)
        assert not (tmp_path / "tonic_epochs_DMS.csv").exists()

    def test_copy_to_all_replicates_current_sites_windows(self, config):
        source = pd.DataFrame({"label": ["baseline"], "start": [0.0], "end": [2.0]})
        config.site_select.value = "DMS"
        config.site_to_widget["DMS"].value = source
        config._on_copy_to_all(None)

        pd.testing.assert_frame_equal(config.site_to_widget["DLS"].value, source)

    def test_make_plot_overlays_spans_for_defined_windows(self, config):
        import holoviews as hv

        config.site_to_widget["DMS"].value = pd.DataFrame({"label": ["baseline"], "start": [0.0], "end": [2.0]})
        config.site_select.value = "DMS"
        plot = config._make_plot()
        # Curve overlaid with one VSpan -> an Overlay, not a bare Curve.
        assert isinstance(plot, hv.Overlay)


def _write_tonic_results(filepath, site):
    timestamps = np.arange(0.0, 11.0, 1.0)
    write_hdf5(timestamps, "timeCorrection_" + site, str(filepath), "timestampNew")
    write_hdf5(timestamps.copy(), "z_score_" + site, str(filepath), "data")
    write_hdf5(timestamps / 10.0, "dff_" + site, str(filepath), "data")
    pd.DataFrame({"label": ["baseline", "post"], "start": [0.0, 8.0], "end": [2.0, 10.0]}).to_csv(
        filepath / f"tonic_epochs_{site}.csv", index=False
    )
    write_tonic_to_hdf5(
        str(filepath),
        pd.DataFrame(
            {"mean_zscore": [1.0, 9.0], "mean_dff": [0.1, 0.9]},
            index=pd.Index(["baseline", "post"], name="epoch"),
        ),
        site,
    )


class TestTonicResultsView:
    def test_build_returns_note_when_no_results(self, panel_extension, tmp_path):
        panel_column = build_tonic_results_view(str(tmp_path))
        markdown = panel_column.objects[0]
        assert "No tonic" in markdown.object

    def test_diff_table_is_relative_to_selected_baseline(self, panel_extension, tmp_path):
        _write_tonic_results(tmp_path, "DMS")
        view = TonicResultsView(str(tmp_path))

        table = view.table_pane.object
        # Baseline defaults to the first epoch ("baseline"), so its diff is 0 and
        # "post" carries the full step: 9.0 - 1.0 = 8.0 in z-score, 0.8 in dF/F.
        assert table.loc["baseline", "diff_zscore"] == 0.0
        assert table.loc["post", "diff_zscore"] == pytest.approx(8.0)
        assert table.loc["post", "diff_dff"] == pytest.approx(0.8)

        # Re-baselining on "post" flips the sign of the baseline row's diff.
        view.baseline_select.value = "post"
        rebased = view.table_pane.object
        assert rebased.loc["baseline", "diff_zscore"] == pytest.approx(-8.0)
        assert rebased.loc["post", "diff_zscore"] == 0.0

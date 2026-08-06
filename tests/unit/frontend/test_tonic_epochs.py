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
        config = TonicEpochConfig(str(tmp_path), site_traces)
        # Adding, removing, and the live preview act on the selected site; pin it so the
        # tests do not depend on dict iteration order.
        config.site_select.value = "DMS"
        return config

    def test_starts_with_no_epoch_windows(self, config):
        assert config.epochs_for("DMS") == []
        assert config.epochs_for("DLS") == []

    def test_reloads_epoch_windows_saved_on_disk(self, panel_extension, tmp_path, site_traces):
        pd.DataFrame({"label": ["baseline"], "start": [0.0], "end": [2.0]}).to_csv(
            tmp_path / "tonic_epochs_DMS.csv", index=False
        )

        config = TonicEpochConfig(str(tmp_path), site_traces)

        assert config.epochs_for("DMS") == [("baseline", 0.0, 2.0)]
        assert config.epochs_for("DLS") == []

    def test_save_writes_csv_only_for_sites_with_windows(self, config, tmp_path):
        config.set_epochs("DMS", [("baseline", 0.0, 2.0), ("post", 8.0, 10.0)])

        config.save()

        saved = pd.read_csv(tmp_path / "tonic_epochs_DMS.csv")
        expected = pd.DataFrame({"label": ["baseline", "post"], "start": [0.0, 8.0], "end": [2.0, 10.0]})
        pd.testing.assert_frame_equal(saved, expected)

        # DLS was left empty, so no file is written for it.
        assert not (tmp_path / "tonic_epochs_DLS.csv").exists()

    def test_blank_row_is_not_saved(self, config, tmp_path):
        config.set_epochs("DMS", [("baseline", 0.0, 2.0)])
        config.add_epoch_row()

        config.save()

        saved = pd.read_csv(tmp_path / "tonic_epochs_DMS.csv")
        pd.testing.assert_frame_equal(saved, pd.DataFrame({"label": ["baseline"], "start": [0.0], "end": [2.0]}))

    def test_removing_a_row_drops_its_window(self, config):
        config.set_epochs("DMS", [("baseline", 0.0, 2.0), ("post", 8.0, 10.0)])

        config._remove_row(config.site_to_rows["DMS"][0])

        assert config.epochs_for("DMS") == [("post", 8.0, 10.0)]

    def test_save_raises_on_window_not_overlapping_recording(self, config, tmp_path):
        # site traces span t in [0, 10]; a [15, 20] window overlaps nothing -> reject early,
        # before any file is written, so the worker never sees an invalid epoch.
        config.set_epochs("DMS", [("late", 15.0, 20.0)])
        with pytest.raises(ValueError, match="does not overlap"):
            config.save()
        assert not (tmp_path / "tonic_epochs_DMS.csv").exists()

    def test_on_save_swallows_validation_error_without_writing(self, config, tmp_path):
        config.set_epochs("DMS", [("late", 15.0, 20.0)])
        # _on_save catches the ValueError (surfacing it as a notification when served)
        # instead of propagating; no file is written.
        config._on_save(None)
        assert not (tmp_path / "tonic_epochs_DMS.csv").exists()

    def test_apply_to_all_replicates_current_sites_windows(self, config):
        config.set_epochs("DMS", [("baseline", 0.0, 2.0)])

        config.apply_epochs_to_all_sites()

        assert config.epochs_for("DLS") == [("baseline", 0.0, 2.0)]

    def test_editing_a_bound_repaints_the_spans_without_rebuilding_the_plot(self, config):
        config.set_epochs("DMS", [("baseline", 0.0, 2.0)])
        plot = config.plot_pane.object

        config.site_to_rows["DMS"][0].end_input.value = 4.0

        assert config.spans_pipe.data == [(0.0, 4.0)]
        # The traces are density-shaded server-side, so a bound edit must not re-aggregate them.
        assert config.plot_pane.object is plot


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

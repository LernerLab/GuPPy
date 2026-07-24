import holoviews as hv
import numpy as np
import pandas as pd
import pytest

from guppy.analysis.io_utils import fetchCoords
from guppy.frontend.artifact_removal import (
    build_artifact_removal_template,
    build_artifact_review_template,
    build_preprocessing_review_template,
    load_pair_traces,
    signal_options,
)
from guppy.utils._hdf5_io import write_hdf5

TIMESTAMPS = np.arange(0.0, 11.0, 1.0)


def _write_site(filepath, site):
    """Write the control/signal/fit + preprocessed HDF5 files for one recording site."""
    write_hdf5(TIMESTAMPS, "timeCorrection_" + site, str(filepath), "timestampNew")
    write_hdf5(np.ones(TIMESTAMPS.shape[0]), "control_" + site, str(filepath), "data")
    write_hdf5(np.ones(TIMESTAMPS.shape[0]) * 2, "signal_" + site, str(filepath), "data")
    write_hdf5(np.ones(TIMESTAMPS.shape[0]) * 3, "cntrl_sig_fit_" + site, str(filepath), "data")
    write_hdf5(TIMESTAMPS.copy(), "z_score_" + site, str(filepath), "data")
    write_hdf5(TIMESTAMPS / 10.0, "dff_" + site, str(filepath), "data")


@pytest.fixture
def run_folder(tmp_path):
    for site in ("DMS", "DLS"):
        _write_site(tmp_path, site)
    return tmp_path


def test_signal_options():
    assert signal_options("z_score") == ["z_score"]
    assert signal_options("dff") == ["dff"]
    assert signal_options("Both") == ["z_score", "dff"]
    assert signal_options("None") == []


def test_load_pair_traces_raises_for_mismatched_recording_sites(tmp_path):
    """A control recording site with no matching signal recording site raises a pairing error."""
    (tmp_path / "control_dms.hdf5").touch()
    (tmp_path / "signal_vms.hdf5").touch()
    with pytest.raises(ValueError) as exception_info:
        load_pair_traces(str(tmp_path))
    message = str(exception_info.value)
    assert "Mismatched control/signal files" in message
    assert "dms" in message
    assert "vms" in message


class TestArtifactRemovalConfig:
    @pytest.fixture
    def config(self, panel_extension, run_folder):
        template = build_artifact_removal_template(str(run_folder), "z_score")
        return template._config

    def test_sites_discovered_from_control_signal_pairs(self, config):
        assert sorted(config.sites) == ["DLS", "DMS"]

    def test_save_writes_interleaved_npy_roundtripping_through_fetchcoords(self, config, run_folder):
        config.site_to_widget["DMS"].value = pd.DataFrame(
            {"label": ["keep_a", "keep_b"], "start": [1.0, 6.0], "end": [3.0, 8.0]}
        )
        config.save()

        saved = np.load(str(run_folder / "coordsForPreProcessing_DMS.npy"))
        # Two windows -> shape (4, 2); column 0 interleaves [s0, e0, s1, e1].
        assert saved.shape == (4, 2)
        np.testing.assert_array_equal(saved[:, 0], np.array([1.0, 3.0, 6.0, 8.0]))
        # fetchCoords reshapes column 0 back into the two good-chunk pairs.
        coords = fetchCoords(str(run_folder), "DMS", TIMESTAMPS)
        np.testing.assert_array_equal(coords, np.array([[1.0, 3.0], [6.0, 8.0]]))

    def test_save_empty_table_writes_no_file(self, config, run_folder):
        config.save()
        assert not (run_folder / "coordsForPreProcessing_DMS.npy").exists()
        assert not (run_folder / "coordsForPreProcessing_DLS.npy").exists()

    def test_save_raises_when_start_not_before_end(self, config, run_folder):
        config.site_to_widget["DMS"].value = pd.DataFrame({"label": ["bad"], "start": [5.0], "end": [5.0]})
        with pytest.raises(ValueError, match="must be less than end"):
            config.save()
        assert not (run_folder / "coordsForPreProcessing_DMS.npy").exists()

    def test_save_raises_when_window_outside_timespan(self, config, run_folder):
        config.site_to_widget["DMS"].value = pd.DataFrame({"label": ["late"], "start": [8.0], "end": [15.0]})
        with pytest.raises(ValueError, match="outside the recording timespan"):
            config.save()
        assert not (run_folder / "coordsForPreProcessing_DMS.npy").exists()

    def test_invalid_site_blocks_all_writes(self, config, run_folder):
        # DLS is valid, DMS is invalid -> validate-all-before-write means neither is written.
        config.site_to_widget["DLS"].value = pd.DataFrame({"label": ["ok"], "start": [1.0], "end": [2.0]})
        config.site_to_widget["DMS"].value = pd.DataFrame({"label": ["bad"], "start": [5.0], "end": [5.0]})
        with pytest.raises(ValueError):
            config.save()
        assert not (run_folder / "coordsForPreProcessing_DLS.npy").exists()
        assert not (run_folder / "coordsForPreProcessing_DMS.npy").exists()

    def test_multiple_sites_each_write_own_file(self, config, run_folder):
        config.site_to_widget["DMS"].value = pd.DataFrame({"label": ["a"], "start": [1.0], "end": [2.0]})
        config.site_to_widget["DLS"].value = pd.DataFrame({"label": ["b"], "start": [3.0], "end": [4.0]})
        config.save()
        np.testing.assert_array_equal(fetchCoords(str(run_folder), "DMS", TIMESTAMPS), np.array([[1.0, 2.0]]))
        np.testing.assert_array_equal(fetchCoords(str(run_folder), "DLS", TIMESTAMPS), np.array([[3.0, 4.0]]))

    def test_on_save_swallows_validation_error_without_writing_or_signalling(self, config, run_folder):
        calls = []
        config.on_done = lambda: calls.append(1)
        config.site_to_widget["DMS"].value = pd.DataFrame({"label": ["bad"], "start": [5.0], "end": [5.0]})
        config._on_save(None)
        assert not (run_folder / "coordsForPreProcessing_DMS.npy").exists()
        assert calls == []

    def test_on_save_signals_done_on_success(self, config, run_folder):
        calls = []
        config.on_done = lambda: calls.append(1)
        config.site_to_widget["DMS"].value = pd.DataFrame({"label": ["a"], "start": [1.0], "end": [2.0]})
        config._on_save(None)
        assert calls == [1]

    def test_marking_plot_shades_defined_windows(self, config):
        config.site_to_widget["DMS"].value = pd.DataFrame({"label": ["a"], "start": [1.0], "end": [2.0]})
        config.site_select.value = "DMS"
        plot = config._make_marking_plot()
        # Every stacked axis becomes an Overlay carrying the VSpan once a window is defined.
        for element in plot.values():
            assert isinstance(element, hv.Overlay)
            assert any(isinstance(item, hv.VSpan) for item in element.values())


class TestReviewSectionGating:
    def test_review_section_present_when_requested(self, panel_extension, run_folder):
        config = build_artifact_removal_template(str(run_folder), "Both")._config
        assert config.signal_toggle is not None
        assert config.signal_toggle.options == ["z_score", "dff"]

    def test_review_section_absent_when_none(self, panel_extension, run_folder):
        config = build_artifact_removal_template(str(run_folder), "None")._config
        assert config.signal_toggle is None
        assert config.review_pane is None


class TestPreprocessingReviewView:
    def test_toggle_switches_between_zscore_and_dff(self, panel_extension, run_folder):
        template = build_preprocessing_review_template(str(run_folder), "Both")
        view = template._view
        view.site_select.value = "DMS"
        view.signal_toggle.value = "z_score"
        zscore_plot = view.plot_pane.object
        np.testing.assert_array_equal(zscore_plot.dimension_values(1), TIMESTAMPS)
        view.signal_toggle.value = "dff"
        dff_plot = view.plot_pane.object
        np.testing.assert_array_equal(dff_plot.dimension_values(1), TIMESTAMPS / 10.0)

    def test_continue_button_signals_done(self, panel_extension, run_folder):
        calls = []
        template = build_preprocessing_review_template(str(run_folder), "z_score", on_done=lambda: calls.append(1))
        template._continue_button.clicks += 1
        assert calls == [1]


class TestArtifactReviewView:
    def test_shades_saved_good_chunks(self, panel_extension, run_folder):
        # A saved coords file drives the shaded windows in review mode.
        np.save(str(run_folder / "coordsForPreProcessing_DMS.npy"), np.array([[2.0, 0.0], [5.0, 0.0]]))
        template = build_artifact_review_template(str(run_folder))
        view = template._view
        view.site_select.value = "DMS"
        plot = view.plot_pane.object
        for element in plot.values():
            vspans = [item for item in element.values() if isinstance(item, hv.VSpan)]
            assert len(vspans) == 1

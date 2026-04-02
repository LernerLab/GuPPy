import h5py
import numpy as np
import pytest

from guppy.analysis.standard_io import write_transients_to_hdf5
from guppy.orchestration.transients import (
    execute_visualize_peaks,
    execute_visualize_peaks_combined,
)

STUB_Z_SCORE = np.array([1.0, 2.0, 3.0])
STUB_TS = np.array([0.0, 0.5, 1.0])
STUB_PEAKS_IND = np.array([1])


def _write_stub_files(output_dir, basename):
    """Create the glob-target HDF5 and the transient data HDF5 for a given basename."""
    # Empty file so the glob pattern finds it
    with h5py.File(str(output_dir / f"{basename}.hdf5"), "w"):
        pass
    write_transients_to_hdf5(str(output_dir), basename, STUB_Z_SCORE, STUB_TS, STUB_PEAKS_IND)


@pytest.fixture
def output_dir(tmp_path):
    """One session folder with one output directory containing a z_score_DMS stub."""
    session_dir = tmp_path / "session1"
    session_dir.mkdir()
    output = session_dir / "session1_output_0"
    output.mkdir()
    _write_stub_files(output, "z_score_DMS")
    return session_dir


@pytest.fixture
def combined_output_dir(tmp_path):
    """Two session folders each with one output_0 directory containing a z_score_DMS stub."""
    session_a = tmp_path / "sessionA"
    session_b = tmp_path / "sessionB"
    for session in (session_a, session_b):
        session.mkdir()
        output = session / f"{session.name}_output_0"
        output.mkdir()
        _write_stub_files(output, "z_score_DMS")
    return [str(session_a), str(session_b)]


class TestExecuteVisualizePeaks:
    def test_calls_visualize_peaks_with_correct_arguments(self, output_dir, base_input_parameters, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "guppy.orchestration.transients.visualize_peaks",
            lambda title, suptitle, z_score, ts, peaksInd: calls.append(
                {"title": title, "suptitle": suptitle, "z_score": z_score, "ts": ts, "peaksInd": peaksInd}
            ),
        )
        monkeypatch.setattr("guppy.orchestration.transients.plt.show", lambda: None)

        base_input_parameters["selectForTransientsComputation"] = "z_score"
        execute_visualize_peaks([str(output_dir)], base_input_parameters)

        assert len(calls) == 1
        assert calls[0]["title"] == "z_score_DMS"
        assert calls[0]["suptitle"] == "session1_output_0"
        np.testing.assert_array_equal(calls[0]["z_score"], STUB_Z_SCORE)
        np.testing.assert_array_equal(calls[0]["ts"], STUB_TS)
        np.testing.assert_array_equal(calls[0]["peaksInd"], STUB_PEAKS_IND)

    def test_selects_dff_files_when_requested(self, tmp_path, base_input_parameters, monkeypatch):
        session_dir = tmp_path / "session1"
        session_dir.mkdir()
        output = session_dir / "session1_output_0"
        output.mkdir()
        _write_stub_files(output, "dff_DMS")

        calls = []
        monkeypatch.setattr(
            "guppy.orchestration.transients.visualize_peaks",
            lambda title, suptitle, z_score, ts, peaksInd: calls.append(title),
        )
        monkeypatch.setattr("guppy.orchestration.transients.plt.show", lambda: None)

        base_input_parameters["selectForTransientsComputation"] = "dff"
        execute_visualize_peaks([str(session_dir)], base_input_parameters)

        assert len(calls) == 1
        assert calls[0] == "dff_DMS"

    def test_selects_both_files_when_requested(self, tmp_path, base_input_parameters, monkeypatch):
        session_dir = tmp_path / "session1"
        session_dir.mkdir()
        output = session_dir / "session1_output_0"
        output.mkdir()
        _write_stub_files(output, "z_score_DMS")
        _write_stub_files(output, "dff_DMS")

        calls = []
        monkeypatch.setattr(
            "guppy.orchestration.transients.visualize_peaks",
            lambda title, suptitle, z_score, ts, peaksInd: calls.append(title),
        )
        monkeypatch.setattr("guppy.orchestration.transients.plt.show", lambda: None)

        base_input_parameters["selectForTransientsComputation"] = "both"
        execute_visualize_peaks([str(session_dir)], base_input_parameters)

        assert len(calls) == 2
        assert set(calls) == {"z_score_DMS", "dff_DMS"}


class TestExecuteVisualizePeaksCombined:
    def test_calls_visualize_peaks_with_correct_arguments(
        self, combined_output_dir, base_input_parameters, monkeypatch
    ):
        calls = []
        monkeypatch.setattr(
            "guppy.orchestration.transients.visualize_peaks",
            lambda title, suptitle, z_score, ts, peaksInd: calls.append(
                {"title": title, "suptitle": suptitle, "z_score": z_score, "ts": ts, "peaksInd": peaksInd}
            ),
        )
        monkeypatch.setattr("guppy.orchestration.transients.plt.show", lambda: None)

        base_input_parameters["selectForTransientsComputation"] = "z_score"
        execute_visualize_peaks_combined(combined_output_dir, base_input_parameters)

        # get_all_stores_for_combining_data groups both output_0 dirs into op[0];
        # execute_visualize_peaks_combined uses op[i][0] (the first of the group), so 1 call.
        assert len(calls) == 1
        assert calls[0]["title"] == "z_score_DMS"
        assert "output_0" in calls[0]["suptitle"]
        np.testing.assert_array_equal(calls[0]["z_score"], STUB_Z_SCORE)
        np.testing.assert_array_equal(calls[0]["ts"], STUB_TS)
        np.testing.assert_array_equal(calls[0]["peaksInd"], STUB_PEAKS_IND)

    def test_combined_selects_dff_files_when_requested(self, tmp_path, base_input_parameters, monkeypatch):
        session_dir = tmp_path / "session1"
        session_dir.mkdir()
        output = session_dir / "session1_output_0"
        output.mkdir()
        _write_stub_files(output, "dff_DMS")

        calls = []
        monkeypatch.setattr(
            "guppy.orchestration.transients.visualize_peaks",
            lambda title, suptitle, z_score, ts, peaksInd: calls.append(title),
        )
        monkeypatch.setattr("guppy.orchestration.transients.plt.show", lambda: None)

        base_input_parameters["selectForTransientsComputation"] = "dff"
        execute_visualize_peaks_combined([str(session_dir)], base_input_parameters)

        assert len(calls) == 1
        assert calls[0] == "dff_DMS"

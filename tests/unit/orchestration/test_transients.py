import h5py
import numpy as np
import pytest

from guppy.analysis.standard_io import write_transients_to_hdf5
from guppy.orchestration.transients import (
    execute_average_for_group,
    execute_visualize_peaks,
    execute_visualize_peaks_combined,
)

STUB_Z_SCORE = np.array([1.0, 2.0, 3.0])
STUB_TS = np.array([0.0, 0.5, 1.0])
STUB_PEAKS_IND = np.array([1])


def _write_stub_files(run_folder, basename):
    """Create the glob-target HDF5 and the transient data HDF5 for a given basename."""
    # Empty file so the glob pattern finds it
    with h5py.File(str(run_folder / f"{basename}.hdf5"), "w"):
        pass
    write_transients_to_hdf5(str(run_folder), basename, STUB_Z_SCORE, STUB_TS, STUB_PEAKS_IND)
    # select_run_folders validates that picked outputs have a storesList.csv (re-run step 1 if missing).
    (run_folder / "storesList.csv").write_text("")


@pytest.fixture
def capture_served(monkeypatch):
    """Capture the served build callable and the folders/select forwarded to the peaks page."""
    served = []
    build_calls = []
    monkeypatch.setattr("guppy.orchestration.transients.serve_blocking_page", lambda build: served.append(build))
    monkeypatch.setattr(
        "guppy.orchestration.transients.build_peaks_review_template",
        lambda folders, select, on_done: build_calls.append(([str(f) for f in folders], select)),
    )
    return served, build_calls


@pytest.fixture
def run_folder(tmp_path):
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
    def test_serves_peaks_page_with_resolved_run_folders(self, run_folder, base_input_parameters, capture_served):
        served, build_calls = capture_served
        base_input_parameters["selectForTransientsComputation"] = "z_score"
        base_input_parameters["selected_runs"] = {str(run_folder): ["0"]}

        execute_visualize_peaks([str(run_folder)], base_input_parameters)

        assert len(served) == 1
        # Invoking the captured build callable confirms which folders/select the page gets.
        served[0](lambda: None)
        assert build_calls == [([str(run_folder / "session1_output_0")], "z_score")]

    def test_forwards_dff_selection(self, run_folder, base_input_parameters, capture_served):
        served, build_calls = capture_served
        base_input_parameters["selectForTransientsComputation"] = "dff"
        base_input_parameters["selected_runs"] = {str(run_folder): ["0"]}

        execute_visualize_peaks([str(run_folder)], base_input_parameters)

        served[0](lambda: None)
        assert build_calls[0][1] == "dff"


class TestExecuteVisualizePeaksCombined:
    def test_serves_peaks_page_with_group_first_folders(
        self, combined_output_dir, base_input_parameters, capture_served
    ):
        served, build_calls = capture_served
        base_input_parameters["selectForTransientsComputation"] = "z_score"
        base_input_parameters["selected_runs"] = {session: ["0"] for session in combined_output_dir}

        execute_visualize_peaks_combined(combined_output_dir, base_input_parameters)

        assert len(served) == 1
        served[0](lambda: None)
        # get_all_stores_for_combining_data groups both output_0 dirs into one group;
        # the page is fed the first folder of each group, so a single folder here.
        folders, select = build_calls[0]
        assert len(folders) == 1
        assert "output_0" in folders[0]
        assert select == "z_score"

    def test_execute_average_for_group_raises_for_empty_folders(self, base_input_parameters):
        with pytest.raises(ValueError, match="No folders selected for group averaging"):
            execute_average_for_group(base_input_parameters, [])

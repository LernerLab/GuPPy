import numpy as np
import pytest

from guppy.orchestration.preprocess import (
    execute_artifact_removal,
    execute_combine_data,
    execute_zscore,
    extractTsAndSignal,
)

# ── error paths ───────────────────────────────────────────────────────────────


def test_execute_zscore_raises_for_mismatched_recording_sites(tmp_path, base_input_parameters):
    """control_dms paired with signal_vms (different recording sites) — raises naming both recording sites."""
    run_folder = tmp_path / "session_output_1"
    run_folder.mkdir()
    (run_folder / "control_dms.hdf5").touch()
    (run_folder / "signal_vms.hdf5").touch()
    base_input_parameters["combine_data"] = True

    with pytest.raises(ValueError) as exception_info:
        execute_zscore([[str(run_folder)]], base_input_parameters)
    message = str(exception_info.value)
    assert "Mismatched control/signal files" in message
    assert "dms" in message
    assert "vms" in message


def test_execute_combine_data_raises_for_mismatched_sampling_rates(monkeypatch, base_input_parameters):
    """When timeCorrection_*.hdf5 files report different sampling rates, the message
    lists both rates and the offending paths."""
    folder_names = ["/tmp/session_a", "/tmp/session_b"]
    store_array = np.array([["ctrl0", "sig0"], ["control_dms", "signal_dms"]])

    monkeypatch.setattr(
        "guppy.orchestration.preprocess.select_run_folders",
        lambda session, selected: (
            [folder_names[0] + "/run_folder"] if "session_a" in session else [folder_names[1] + "/run_folder"]
        ),
    )
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.glob.glob",
        lambda pattern: (
            [f"{folder_names[0]}/run_folder/timeCorrection_dms.hdf5"]
            if "session_a" in pattern
            else [f"{folder_names[1]}/run_folder/timeCorrection_dms.hdf5"] if "session_b" in pattern else []
        ),
    )

    rates = iter([np.array([100.0]), np.array([250.0])])
    monkeypatch.setattr("guppy.orchestration.preprocess.read_hdf5", lambda *a, **k: next(rates))
    monkeypatch.setattr("guppy.orchestration.preprocess.np.genfromtxt", lambda *a, **k: store_array)

    with pytest.raises(ValueError, match="sampling rates differ"):
        execute_combine_data(folder_names, base_input_parameters, store_array)


# ── compute (no GUI in the worker) ──────────────────────────────────────────────


def test_execute_zscore_computes_and_writes(monkeypatch, base_input_parameters):
    folder_names = [["/tmp/session_output_1"]]

    base_input_parameters["combine_data"] = True

    monkeypatch.setattr(
        "guppy.orchestration.preprocess.decide_naming_convention",
        lambda filepath: np.array([[f"{filepath}/control_DMS.hdf5"], [f"{filepath}/signal_DMS.hdf5"]]),
    )
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.read_corrected_data",
        lambda control_path, signal_path, filepath, name: (
            np.array([1.0, 1.0, 1.0]),
            np.array([2.0, 2.0, 2.0]),
            np.array([0.0, 1.0, 2.0]),
        ),
    )
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.get_coords", lambda filepath, name, ts_new, remove: np.array([])
    )
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.compute_z_score",
        lambda *args, **kwargs: (
            np.array([0.0, 0.1, 0.2]),
            np.array([0.0, 0.2, 0.4]),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        ),
    )

    write_calls = []
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.write_zscore",
        lambda filepath, name, z_score, dff, control_fit, temporary_control_array: write_calls.append((filepath, name)),
    )
    monkeypatch.setattr("guppy.orchestration.preprocess.writeToFile", lambda text, file_path: None)

    execute_zscore(folder_names, base_input_parameters)

    assert write_calls == [("/tmp/session_output_1", "DMS")]


@pytest.fixture
def stub_artifact_removal_io(monkeypatch):
    """Patch execute_artifact_removal's disk I/O so only its compute/write path is exercised."""
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.select_run_folders", lambda session, selected: ["/tmp/session_1/run_1"]
    )
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.np.genfromtxt", lambda *a, **k: np.array([["ctrl0"], ["control_dms"]])
    )
    monkeypatch.setattr("guppy.orchestration.preprocess.read_corrected_data_dict", lambda filepath, store_array: {})
    monkeypatch.setattr("guppy.orchestration.preprocess.read_corrected_timestamps_pairwise", lambda filepath: ({}, {}))
    monkeypatch.setattr("guppy.orchestration.preprocess.read_coords_pairwise", lambda filepath, tsNew: {})
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.read_corrected_ttl_timestamps", lambda filepath, store_array: {}
    )
    monkeypatch.setattr("guppy.orchestration.preprocess.remove_artifacts", lambda *a, **k: ({}, {}, {}))
    monkeypatch.setattr("guppy.orchestration.preprocess.writeToFile", lambda text, file_path: None)


def test_execute_artifact_removal_removes_and_writes_per_folder(
    monkeypatch, base_input_parameters, stub_artifact_removal_io
):
    """execute_artifact_removal applies removal and writes corrected data per run folder — no GUI."""
    base_input_parameters["combine_data"] = False

    write_calls = []
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.write_artifact_removal",
        lambda filepath, *a, **k: write_calls.append(filepath),
    )

    execute_artifact_removal(["/tmp/session_1"], base_input_parameters)

    assert write_calls == ["/tmp/session_1/run_1"]


# ── extractTsAndSignal dispatch (compute only; no visualization) ─────────────────


@pytest.fixture
def stub_extract_ts_and_signal_io(monkeypatch):
    """Patch extractTsAndSignal's sub-steps so only its dispatch branches run."""
    monkeypatch.setattr("guppy.orchestration.preprocess.save_parameters", lambda *, inputParameters: None)
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.select_run_folders", lambda session, selected: ["/tmp/session_1/run_1"]
    )
    monkeypatch.setattr("guppy.orchestration.preprocess.writeToFile", lambda text, file_path: None)
    monkeypatch.setattr("guppy.orchestration.preprocess.execute_timestamp_correction", lambda folders, params: None)
    monkeypatch.setattr("guppy.orchestration.preprocess.execute_zscore", lambda folders, params: None)
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.check_storeslistfile", lambda folders: np.array([["ctrl0"], ["control_dms"]])
    )
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.execute_combine_data", lambda folders, params, store_array: ["/tmp/combined_1"]
    )
    monkeypatch.setattr("guppy.orchestration.preprocess.write_combined_stores_list", lambda folders, store_array: None)


def test_extract_ts_and_signal_runs_artifact_removal_when_enabled(
    monkeypatch, base_input_parameters, stub_extract_ts_and_signal_io
):
    """Non-combine path: artifact removal runs on the session folders; the worker never visualizes."""
    base_input_parameters["session_folders"] = ["/tmp/session_1"]
    base_input_parameters["combine_data"] = False
    base_input_parameters["removeArtifacts"] = True

    artifact_calls = []
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.execute_artifact_removal",
        lambda folders, params: artifact_calls.append(folders),
    )

    extractTsAndSignal(base_input_parameters)

    assert artifact_calls == [["/tmp/session_1"]]


def test_extract_ts_and_signal_skips_artifact_removal_when_disabled(
    monkeypatch, base_input_parameters, stub_extract_ts_and_signal_io
):
    """When removeArtifacts is False the worker computes only — no removal, no visualization."""
    base_input_parameters["session_folders"] = ["/tmp/session_1"]
    base_input_parameters["combine_data"] = False
    base_input_parameters["removeArtifacts"] = False

    artifact_calls = []
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.execute_artifact_removal",
        lambda folders, params: artifact_calls.append(folders),
    )

    extractTsAndSignal(base_input_parameters)

    assert artifact_calls == []


def test_extract_ts_and_signal_combine_runs_artifact_removal_on_combined_folders(
    monkeypatch, base_input_parameters, stub_extract_ts_and_signal_io
):
    """Combine path: artifact removal runs on the combined-output folders."""
    base_input_parameters["session_folders"] = ["/tmp/session_1"]
    base_input_parameters["combine_data"] = True
    base_input_parameters["removeArtifacts"] = True

    artifact_calls = []
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.execute_artifact_removal",
        lambda folders, params: artifact_calls.append(folders),
    )

    extractTsAndSignal(base_input_parameters)

    assert artifact_calls == [["/tmp/combined_1"]]

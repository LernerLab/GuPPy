import numpy as np
import pytest

from guppy.orchestration.preprocess import (
    execute_artifact_removal,
    execute_combine_data,
    execute_zscore,
    extractTsAndSignal,
    removeArtifactsFromSignal,
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
        execute_zscore([[str(run_folder)]], base_input_parameters, remove_artifacts=False)
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


def test_execute_zscore_raises_when_detrending_without_isosbestic_control(base_input_parameters):
    base_input_parameters["isosbestic_control"] = False
    base_input_parameters["photobleaching_detrend"] = True

    with pytest.raises(ValueError) as exception_info:
        execute_zscore([["/tmp/session_output_1"]], base_input_parameters, remove_artifacts=False)
    message = str(exception_info.value)
    assert "photobleaching_detrend=True" in message
    assert "requires an isosbestic control channel" in message


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

    execute_zscore(folder_names, base_input_parameters, remove_artifacts=False)

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


# ── step entry points (compute only; no visualization) ──────────────────────────


@pytest.fixture
def recorded_calls():
    """Collects the sub-step invocations each entry point makes, in order."""
    return []


@pytest.fixture
def marked_run_folder(tmp_path):
    """A run folder that already has saved artifact windows, as Select Artifact Windows leaves it."""
    run_folder = tmp_path / "session_1_output_run_1"
    run_folder.mkdir()
    np.save(run_folder / "coordsForPreProcessing_dms.npy", np.array([[0.0, 0.0], [5.0, 0.0]]))
    return str(run_folder)


@pytest.fixture
def unmarked_run_folder(tmp_path):
    """A run folder with no saved artifact windows."""
    run_folder = tmp_path / "session_2_output_run_1"
    run_folder.mkdir()
    return str(run_folder)


@pytest.fixture
def stub_entry_point_io(monkeypatch, recorded_calls, marked_run_folder):
    """Patch the entry points' sub-steps so only their dispatch and ordering run."""
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.save_parameters",
        lambda *, inputParameters, remove_artifacts=None, artifacts_removal_method=None: recorded_calls.append(
            ("save_parameters", remove_artifacts)
        ),
    )
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.select_run_folders", lambda session, selected: ["/tmp/session_1/run_1"]
    )
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.execute_timestamp_correction",
        lambda folders, params: recorded_calls.append(("correction", folders)),
    )
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.execute_zscore",
        lambda folders, params, *, remove_artifacts: recorded_calls.append(("zscore", folders, remove_artifacts)),
    )
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.execute_artifact_removal",
        lambda folders, params: recorded_calls.append(("removal", folders)),
    )
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.check_storeslistfile", lambda folders: np.array([["ctrl0"], ["control_dms"]])
    )
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.execute_combine_data", lambda folders, params, store_array: ["/tmp/combined_1"]
    )
    monkeypatch.setattr("guppy.orchestration.preprocess.write_combined_stores_list", lambda folders, store_array: None)
    # Left unpatched so validate_artifact_coords_present runs against the real folder.
    monkeypatch.setattr(
        "guppy.orchestration.preprocess.resolve_run_folders",
        lambda session_folders, params: [marked_run_folder],
    )


@pytest.fixture
def started_totals(monkeypatch):
    """Captures the denominators handed to the progress bar."""
    totals = []
    monkeypatch.setattr("guppy.orchestration.preprocess.progress.start", lambda total: totals.append(total))
    return totals


class TestExtractTsAndSignal:
    """Step 3 corrects timestamps and z-scores over the full recording; it never excises."""

    def test_runs_correction_then_full_span_zscore(self, base_input_parameters, stub_entry_point_io, recorded_calls):
        base_input_parameters["session_folders"] = ["/tmp/session_1"]
        base_input_parameters["combine_data"] = False

        extractTsAndSignal(base_input_parameters)

        assert recorded_calls == [
            ("save_parameters", False),
            ("correction", ["/tmp/session_1"]),
            ("zscore", ["/tmp/session_1"], False),
        ]

    def test_combine_zscores_the_combined_folders(self, base_input_parameters, stub_entry_point_io, recorded_calls):
        base_input_parameters["session_folders"] = ["/tmp/session_1"]
        base_input_parameters["combine_data"] = True

        extractTsAndSignal(base_input_parameters)

        assert recorded_calls == [
            ("save_parameters", False),
            ("correction", ["/tmp/session_1"]),
            ("zscore", ["/tmp/combined_1"], False),
        ]

    def test_progress_budget_is_two_passes_per_folder(self, base_input_parameters, stub_entry_point_io, started_totals):
        base_input_parameters["session_folders"] = ["/tmp/session_1"]
        base_input_parameters["combine_data"] = False

        extractTsAndSignal(base_input_parameters)

        assert started_totals == [2]


class TestRemoveArtifactsFromSignal:
    """The optional step re-runs correction and coords-aware z-score, then excises."""

    def test_runs_correction_zscore_with_coords_then_removal(
        self, base_input_parameters, stub_entry_point_io, recorded_calls
    ):
        base_input_parameters["session_folders"] = ["/tmp/session_1"]
        base_input_parameters["combine_data"] = False

        removeArtifactsFromSignal(base_input_parameters)

        assert recorded_calls == [
            ("save_parameters", True),
            ("correction", ["/tmp/session_1"]),
            ("zscore", ["/tmp/session_1"], True),
            ("removal", ["/tmp/session_1"]),
        ]

    def test_combine_runs_later_passes_on_the_combined_folders(
        self, base_input_parameters, stub_entry_point_io, recorded_calls
    ):
        base_input_parameters["session_folders"] = ["/tmp/session_1"]
        base_input_parameters["combine_data"] = True

        removeArtifactsFromSignal(base_input_parameters)

        assert recorded_calls == [
            ("save_parameters", True),
            ("correction", ["/tmp/session_1"]),
            ("zscore", ["/tmp/combined_1"], True),
            ("removal", ["/tmp/combined_1"]),
        ]

    def test_progress_budget_is_three_passes_per_folder(
        self, base_input_parameters, stub_entry_point_io, started_totals
    ):
        base_input_parameters["session_folders"] = ["/tmp/session_1"]
        base_input_parameters["combine_data"] = False

        removeArtifactsFromSignal(base_input_parameters)

        assert started_totals == [3]

    def test_missing_coords_raises_before_any_pass_runs(
        self, monkeypatch, base_input_parameters, stub_entry_point_io, recorded_calls, unmarked_run_folder
    ):
        base_input_parameters["session_folders"] = ["/tmp/session_1"]
        base_input_parameters["combine_data"] = False
        monkeypatch.setattr(
            "guppy.orchestration.preprocess.resolve_run_folders",
            lambda session_folders, params: [unmarked_run_folder],
        )

        with pytest.raises(ValueError, match="No artifact windows have been selected"):
            removeArtifactsFromSignal(base_input_parameters)

        assert recorded_calls == []

import json
import os

import h5py
import numpy as np
import pytest

from guppy.orchestration.psth import (
    _validate_fiber_recording_sites_consistent_for_group,
    _validate_psth_window_parameters,
    execute_average_for_group,
    execute_compute_cross_correlation,
    execute_compute_psth,
    execute_compute_psth_peak_and_area,
    run_psth_step,
)
from guppy.utils.progress import StepProgress, _current_step


@pytest.fixture
def psth_output_dir(tmp_path):
    """Create a minimal output directory with a z_score_DMS.hdf5 for glob discovery."""
    run_folder = tmp_path / "session1" / "session1_output_1"
    run_folder.mkdir(parents=True)
    with h5py.File(str(run_folder / "z_score_DMS.hdf5"), "w") as hdf5_file:
        hdf5_file.create_dataset("data", data=np.zeros(100))
    return run_folder


# ---------------------------------------------------------------------------
# execute_compute_psth — early-exit for control/signal events
# ---------------------------------------------------------------------------


def test_execute_compute_psth_returns_zero_for_control_event(psth_output_dir, base_input_parameters):
    result = execute_compute_psth(str(psth_output_dir), "control_DMS", base_input_parameters)
    assert result == 0


def test_execute_compute_psth_returns_zero_for_signal_event(psth_output_dir, base_input_parameters):
    result = execute_compute_psth(str(psth_output_dir), "signal_DMS", base_input_parameters)
    assert result == 0


def test_execute_compute_psth_returns_zero_for_mixed_case_control(psth_output_dir, base_input_parameters):
    result = execute_compute_psth(str(psth_output_dir), "Control_DMS", base_input_parameters)
    assert result == 0


# ---------------------------------------------------------------------------
# execute_compute_psth_peak_and_area — early-exit for control/signal events
# ---------------------------------------------------------------------------


def test_execute_compute_psth_peak_and_area_returns_zero_for_control_event(psth_output_dir, base_input_parameters):
    result = execute_compute_psth_peak_and_area(str(psth_output_dir), "control_DMS", base_input_parameters)
    assert result == 0


def test_execute_compute_psth_peak_and_area_returns_zero_for_signal_event(psth_output_dir, base_input_parameters):
    result = execute_compute_psth_peak_and_area(str(psth_output_dir), "signal_DMS", base_input_parameters)
    assert result == 0


# ---------------------------------------------------------------------------
# execute_compute_cross_correlation
# ---------------------------------------------------------------------------


def test_execute_compute_cross_correlation_no_op_when_compute_corr_false(
    psth_output_dir, base_input_parameters, monkeypatch
):
    get_corr_calls = []
    monkeypatch.setattr(
        "guppy.orchestration.psth.getCorrCombinations",
        lambda filepath, inputParameters: get_corr_calls.append(filepath) or ([], []),
    )

    base_input_parameters["computeCorr"] = False
    execute_compute_cross_correlation(str(psth_output_dir), "lever_press", base_input_parameters)

    assert len(get_corr_calls) == 0


def _record_artifact_provenance(run_folder, *, remove_artifacts, artifacts_removal_method):
    """Write the artifact provenance the Remove Artifacts step would have left behind."""
    with open(os.path.join(str(run_folder), "GuPPyParamtersUsed.json"), "w") as parameters_file:
        json.dump(
            {"removeArtifacts": remove_artifacts, "artifactsRemovalMethod": artifacts_removal_method},
            parameters_file,
        )


def test_execute_compute_cross_correlation_raises_when_run_was_concatenated(psth_output_dir, base_input_parameters):
    """The guard reads what was actually applied to this run, not a live form value."""
    base_input_parameters["computeCorr"] = True
    _record_artifact_provenance(psth_output_dir, remove_artifacts=True, artifacts_removal_method="concatenate")

    with pytest.raises(ValueError, match=r"cannot run on concatenated data"):
        execute_compute_cross_correlation(str(psth_output_dir), "lever_press", base_input_parameters)


def test_execute_compute_cross_correlation_allows_run_removed_with_nan(
    psth_output_dir, base_input_parameters, monkeypatch
):
    """A run removed with 'replace with NaN' passes the guard and proceeds to the correlation itself."""
    base_input_parameters["computeCorr"] = True
    _record_artifact_provenance(psth_output_dir, remove_artifacts=True, artifacts_removal_method="replace with NaN")
    get_corr_calls = []
    monkeypatch.setattr(
        "guppy.orchestration.psth.getCorrCombinations",
        lambda filepath, params: get_corr_calls.append(filepath) or (["dms", "vms"], ["z_score"]),
    )
    monkeypatch.setattr("guppy.orchestration.psth.read_Df", lambda *args, **kwargs: None)

    execute_compute_cross_correlation(str(psth_output_dir), "control_DMS", base_input_parameters)

    assert get_corr_calls == [str(psth_output_dir)]


def test_execute_average_for_group_raises_for_empty_folders(base_input_parameters):
    base_input_parameters["group_session_folders"] = []
    with pytest.raises(ValueError, match="No folders selected for group averaging"):
        execute_average_for_group(base_input_parameters)


def test_execute_compute_cross_correlation_returns_early_for_control_event(
    psth_output_dir, base_input_parameters, monkeypatch
):
    read_df_calls = []
    # corr_info with 2 entries so the loop would run if not for the early return
    monkeypatch.setattr(
        "guppy.orchestration.psth.getCorrCombinations",
        lambda filepath, inputParameters: (["region_1", "region_2"], ["z_score"]),
    )
    monkeypatch.setattr(
        "guppy.orchestration.psth.read_Df",
        lambda *args, **kwargs: read_df_calls.append(args),
    )

    base_input_parameters["computeCorr"] = True
    execute_compute_cross_correlation(str(psth_output_dir), "control_DMS", base_input_parameters)

    assert len(read_df_calls) == 0


def test_execute_compute_cross_correlation_raises_for_single_recording_site(
    psth_output_dir, base_input_parameters, monkeypatch
):
    """When computeCorr=True but only one signal recording site is present, a ValueError is raised."""
    monkeypatch.setattr(
        "guppy.orchestration.psth.getCorrCombinations",
        lambda filepath, inputParameters: (["dms"], ["z_score"]),
    )

    base_input_parameters["computeCorr"] = True

    with pytest.raises(ValueError, match="only one was found: 'dms'"):
        execute_compute_cross_correlation(str(psth_output_dir), "lever_press", base_input_parameters)


def test_execute_compute_cross_correlation_raises_for_no_recording_sites(
    psth_output_dir, base_input_parameters, monkeypatch
):
    """When computeCorr=True but no signal recording sites are found, a ValueError is raised."""
    monkeypatch.setattr(
        "guppy.orchestration.psth.getCorrCombinations",
        lambda filepath, inputParameters: ([], ["z_score"]),
    )

    base_input_parameters["computeCorr"] = True

    with pytest.raises(ValueError, match="no signal recording sites were found"):
        execute_compute_cross_correlation(str(psth_output_dir), "lever_press", base_input_parameters)


# ---------------------------------------------------------------------------
# _validate_fiber_recording_sites_consistent_for_group
# ---------------------------------------------------------------------------


def _write_stores_list(run_folder, store_ids):
    """Write a minimal storesList.csv with 'raw' labels in row 0 and store_ids in row 1."""
    run_folder.mkdir(parents=True, exist_ok=True)
    raw_labels = [f"raw{i}" for i in range(len(store_ids))]
    rows = [",".join(raw_labels), ",".join(store_ids)]
    (run_folder / "storesList.csv").write_text("\n".join(rows) + "\n")


def test_validate_fiber_recording_sites_consistent_passes_when_all_match(tmp_path):
    output_1 = tmp_path / "session1" / "session1_output_1"
    output_2 = tmp_path / "session2" / "session2_output_1"
    _write_stores_list(output_1, ["control_DMS", "signal_DMS", "port_entries"])
    _write_stores_list(output_2, ["control_DMS", "signal_DMS", "port_entries"])

    # Should not raise
    _validate_fiber_recording_sites_consistent_for_group(np.array([str(output_1), str(output_2)]))


def test_validate_fiber_recording_sites_consistent_allows_reordered_stores(tmp_path):
    output_1 = tmp_path / "session1" / "session1_output_1"
    output_2 = tmp_path / "session2" / "session2_output_1"
    _write_stores_list(output_1, ["control_DMS", "signal_DMS", "port_entries"])
    _write_stores_list(output_2, ["signal_DMS", "port_entries", "control_DMS"])

    _validate_fiber_recording_sites_consistent_for_group(np.array([str(output_1), str(output_2)]))


def test_validate_fiber_recording_sites_allows_same_recording_site_different_event(tmp_path):
    # Issue #368: sessions recorded from the same fiber recording site (DMS) but under
    # different behavioral conditions must be allowed to average together.
    output_1 = tmp_path / "session1" / "session1_output_1"
    output_2 = tmp_path / "session2" / "session2_output_1"
    _write_stores_list(output_1, ["control_DMS", "signal_DMS", "novelobject"])
    _write_stores_list(output_2, ["control_DMS", "signal_DMS", "novelfemale1"])

    # Fiber store_ids match; only the event differs → should not raise
    _validate_fiber_recording_sites_consistent_for_group(np.array([str(output_1), str(output_2)]))


def test_validate_fiber_recording_sites_raises_for_non_overlapping_fibers(tmp_path):
    output_1 = tmp_path / "session1" / "session1_output_1"
    output_2 = tmp_path / "session2" / "session2_output_1"
    _write_stores_list(output_1, ["control_DMS_A", "signal_DMS_A", "port_entries_A"])
    _write_stores_list(output_2, ["control_DMS_B", "signal_DMS_B", "port_entries_B"])

    with pytest.raises(ValueError, match="mismatched control/signal store_ids"):
        _validate_fiber_recording_sites_consistent_for_group(np.array([str(output_1), str(output_2)]))


def test_validate_fiber_recording_sites_raises_for_mismatched_recording_site_labels(tmp_path):
    output_1 = tmp_path / "session1" / "session1_output_1"
    output_2 = tmp_path / "session2" / "session2_output_1"
    _write_stores_list(output_1, ["control_region1", "signal_region1", "port_entries1"])
    _write_stores_list(output_2, ["control_region2", "signal_region2", "port_entries2"])

    with pytest.raises(ValueError, match="mismatched control/signal store_ids"):
        _validate_fiber_recording_sites_consistent_for_group(np.array([str(output_1), str(output_2)]))


def test_validate_fiber_recording_sites_error_message_lists_session_name_and_store_ids(tmp_path):
    output_1 = tmp_path / "session1" / "session1_output_1"
    output_2 = tmp_path / "session2" / "session2_output_1"
    _write_stores_list(output_1, ["control_region1", "signal_region1"])
    _write_stores_list(output_2, ["control_region2", "signal_region2"])

    with pytest.raises(ValueError) as exc_info:
        _validate_fiber_recording_sites_consistent_for_group(np.array([str(output_1), str(output_2)]))
    message = str(exc_info.value)
    # Session folder names
    assert "session1" in message
    assert "session2" in message
    # Store IDs for each session
    assert "control_region1" in message
    assert "signal_region1" in message
    assert "control_region2" in message
    assert "signal_region2" in message


def test_validate_fiber_recording_sites_single_session_does_not_raise(tmp_path):
    output_1 = tmp_path / "session1" / "session1_output_1"
    _write_stores_list(output_1, ["control_DMS", "signal_DMS"])

    # Single session → trivially consistent
    _validate_fiber_recording_sites_consistent_for_group(np.array([str(output_1)]))


# ---------------------------------------------------------------------------
# _validate_psth_window_parameters — upfront step-4 input validation
# ---------------------------------------------------------------------------


@pytest.fixture
def psth_window_inputs():
    """Minimal inputParameters dict for _validate_psth_window_parameters tests.

    Uses production-shaped values: nSecPrev is the negative-side bound and
    baselineCorrectionStart is negative-relative-to-event.
    """
    return {
        "peak_startPoint": [0.0, np.nan],
        "peak_endPoint": [3.0, np.nan],
        "nSecPrev": -10,
        "nSecPost": 20,
        "baselineCorrectionStart": -5,
        "baselineCorrectionEnd": 0,
    }


def test_validate_psth_window_parameters_passes_with_valid_inputs(psth_window_inputs):
    # Should not raise.
    _validate_psth_window_parameters(psth_window_inputs)


def test_validate_psth_window_parameters_skips_when_baseline_correction_off(psth_window_inputs):
    # (0, 0) is the documented sentinel for "skip baseline correction".
    psth_window_inputs["baselineCorrectionStart"] = 0
    psth_window_inputs["baselineCorrectionEnd"] = 0
    _validate_psth_window_parameters(psth_window_inputs)


def test_validate_psth_window_parameters_raises_for_inverted_peak_window(psth_window_inputs):
    psth_window_inputs["peak_startPoint"] = [3.0]
    psth_window_inputs["peak_endPoint"] = [1.0]
    with pytest.raises(ValueError, match=r"Peak End Time is less than or equal to Peak Start Time"):
        _validate_psth_window_parameters(psth_window_inputs)


def test_validate_psth_window_parameters_raises_for_inverted_baseline_window(psth_window_inputs):
    psth_window_inputs["baselineCorrectionStart"] = 0
    psth_window_inputs["baselineCorrectionEnd"] = -5
    with pytest.raises(
        ValueError, match=r"baselineCorrectionStart=0 must be strictly less than baselineCorrectionEnd=-5"
    ):
        _validate_psth_window_parameters(psth_window_inputs)


def test_validate_psth_window_parameters_raises_when_baseline_outside_psth_window(psth_window_inputs):
    # baselineCorrectionEnd=25 exceeds nSecPost=20.
    psth_window_inputs["baselineCorrectionStart"] = -5
    psth_window_inputs["baselineCorrectionEnd"] = 25
    with pytest.raises(ValueError, match=r"PSTH window is \[-10, 20\]s"):
        _validate_psth_window_parameters(psth_window_inputs)


def test_validate_psth_window_parameters_raises_for_unequal_peak_array_lengths(psth_window_inputs):
    psth_window_inputs["peak_startPoint"] = [0.0, 1.0]
    psth_window_inputs["peak_endPoint"] = [2.0]
    with pytest.raises(ValueError, match=r"unequal \(start: 2, end: 1\)"):
        _validate_psth_window_parameters(psth_window_inputs)


# ---------------------------------------------------------------------------
# run_psth_step — chains PSTH into transient analysis in-process
# ---------------------------------------------------------------------------


class TestRunPsthStep:
    @pytest.fixture
    def recorded_calls(self, monkeypatch):
        """Record the order in which main() invokes the two step-4 workers."""
        calls = []

        def fake_psth_for_each_store(inputParameters):
            calls.append(("psthForEachStore", inputParameters))
            return inputParameters

        def fake_execute_find_freq_and_amp(inputParameters):
            calls.append(("executeFindFreqAndAmp", inputParameters))

        monkeypatch.setattr("guppy.orchestration.psth.psthForEachStore", fake_psth_for_each_store)
        monkeypatch.setattr("guppy.orchestration.psth.executeFindFreqAndAmp", fake_execute_find_freq_and_amp)
        return calls

    def test_runs_psth_then_transients_on_the_same_parameters(self, recorded_calls):
        input_parameters = {"session_folders": ["/tmp/session1"]}

        run_psth_step(input_parameters)

        assert [name for name, _ in recorded_calls] == ["psthForEachStore", "executeFindFreqAndAmp"]
        assert recorded_calls[0][1] is input_parameters
        assert recorded_calls[1][1] is input_parameters

    def test_transients_failure_is_reported_through_the_progress_channel(self, recorded_calls, monkeypatch):
        """A failure in the chained transients step must reach the progress error channel,
        which is how the GUI surfaces it — previously it was lost in a grandchild process."""
        step = StepProgress()
        token = _current_step.set(step)

        def failing_transients(inputParameters):
            raise ValueError("transientsThresh=0 must be positive")

        monkeypatch.setattr("guppy.orchestration.psth.executeFindFreqAndAmp", failing_transients)

        try:
            with pytest.raises(ValueError, match="transientsThresh=0 must be positive"):
                run_psth_step({})
        finally:
            _current_step.reset(token)

        assert step.error_message == "transientsThresh=0 must be positive"

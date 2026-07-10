import h5py
import numpy as np
import pytest

from guppy.orchestration.psth import (
    _validate_fiber_regions_consistent_for_group,
    _validate_psth_window_parameters,
    execute_average_for_group,
    execute_compute_cross_correlation,
    execute_compute_psth,
    execute_compute_psth_peak_and_area,
)


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


def test_execute_compute_cross_correlation_raises_when_concatenate_and_remove_artifacts(
    psth_output_dir, base_input_parameters
):
    base_input_parameters["computeCorr"] = True
    base_input_parameters["removeArtifacts"] = True
    base_input_parameters["artifactsRemovalMethod"] = "concatenate"

    with pytest.raises(ValueError, match=r"must be 'replace with NaNs' and not 'concatenate'"):
        execute_compute_cross_correlation(str(psth_output_dir), "lever_press", base_input_parameters)


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
    base_input_parameters["removeArtifacts"] = False
    execute_compute_cross_correlation(str(psth_output_dir), "control_DMS", base_input_parameters)

    assert len(read_df_calls) == 0


def test_execute_compute_cross_correlation_raises_for_single_region(
    psth_output_dir, base_input_parameters, monkeypatch
):
    """When computeCorr=True but only one signal region is present, a ValueError is raised."""
    monkeypatch.setattr(
        "guppy.orchestration.psth.getCorrCombinations",
        lambda filepath, inputParameters: (["dms"], ["z_score"]),
    )

    base_input_parameters["computeCorr"] = True
    base_input_parameters["removeArtifacts"] = False

    with pytest.raises(ValueError, match="only one was found: 'dms'"):
        execute_compute_cross_correlation(str(psth_output_dir), "lever_press", base_input_parameters)


def test_execute_compute_cross_correlation_raises_for_no_regions(psth_output_dir, base_input_parameters, monkeypatch):
    """When computeCorr=True but no signal regions are found, a ValueError is raised."""
    monkeypatch.setattr(
        "guppy.orchestration.psth.getCorrCombinations",
        lambda filepath, inputParameters: ([], ["z_score"]),
    )

    base_input_parameters["computeCorr"] = True
    base_input_parameters["removeArtifacts"] = False

    with pytest.raises(ValueError, match="no signal regions were found"):
        execute_compute_cross_correlation(str(psth_output_dir), "lever_press", base_input_parameters)


# ---------------------------------------------------------------------------
# _validate_fiber_regions_consistent_for_group
# ---------------------------------------------------------------------------


def _write_stores_list(run_folder, store_ids):
    """Write a minimal storesList.csv with 'raw' labels in row 0 and store_ids in row 1."""
    run_folder.mkdir(parents=True, exist_ok=True)
    raw_labels = [f"raw{i}" for i in range(len(store_ids))]
    rows = [",".join(raw_labels), ",".join(store_ids)]
    (run_folder / "storesList.csv").write_text("\n".join(rows) + "\n")


def test_validate_fiber_regions_consistent_passes_when_all_match(tmp_path):
    output_1 = tmp_path / "session1" / "session1_output_1"
    output_2 = tmp_path / "session2" / "session2_output_1"
    _write_stores_list(output_1, ["control_DMS", "signal_DMS", "port_entries"])
    _write_stores_list(output_2, ["control_DMS", "signal_DMS", "port_entries"])

    # Should not raise
    _validate_fiber_regions_consistent_for_group(np.array([str(output_1), str(output_2)]))


def test_validate_fiber_regions_consistent_allows_reordered_storenames(tmp_path):
    output_1 = tmp_path / "session1" / "session1_output_1"
    output_2 = tmp_path / "session2" / "session2_output_1"
    _write_stores_list(output_1, ["control_DMS", "signal_DMS", "port_entries"])
    _write_stores_list(output_2, ["signal_DMS", "port_entries", "control_DMS"])

    _validate_fiber_regions_consistent_for_group(np.array([str(output_1), str(output_2)]))


def test_validate_fiber_regions_allows_same_region_different_event(tmp_path):
    # Issue #368: sessions recorded from the same fiber region (DMS) but under
    # different behavioral conditions must be allowed to average together.
    output_1 = tmp_path / "session1" / "session1_output_1"
    output_2 = tmp_path / "session2" / "session2_output_1"
    _write_stores_list(output_1, ["control_DMS", "signal_DMS", "novelobject"])
    _write_stores_list(output_2, ["control_DMS", "signal_DMS", "novelfemale1"])

    # Fiber store_ids match; only the event differs → should not raise
    _validate_fiber_regions_consistent_for_group(np.array([str(output_1), str(output_2)]))


def test_validate_fiber_regions_raises_for_non_overlapping_fibers(tmp_path):
    output_1 = tmp_path / "session1" / "session1_output_1"
    output_2 = tmp_path / "session2" / "session2_output_1"
    _write_stores_list(output_1, ["control_DMS_A", "signal_DMS_A", "port_entries_A"])
    _write_stores_list(output_2, ["control_DMS_B", "signal_DMS_B", "port_entries_B"])

    with pytest.raises(ValueError, match="mismatched control/signal store_ids"):
        _validate_fiber_regions_consistent_for_group(np.array([str(output_1), str(output_2)]))


def test_validate_fiber_regions_raises_for_mismatched_region_labels(tmp_path):
    output_1 = tmp_path / "session1" / "session1_output_1"
    output_2 = tmp_path / "session2" / "session2_output_1"
    _write_stores_list(output_1, ["control_region1", "signal_region1", "port_entries1"])
    _write_stores_list(output_2, ["control_region2", "signal_region2", "port_entries2"])

    with pytest.raises(ValueError, match="mismatched control/signal store_ids"):
        _validate_fiber_regions_consistent_for_group(np.array([str(output_1), str(output_2)]))


def test_validate_fiber_regions_error_message_lists_session_name_and_storenames(tmp_path):
    output_1 = tmp_path / "session1" / "session1_output_1"
    output_2 = tmp_path / "session2" / "session2_output_1"
    _write_stores_list(output_1, ["control_region1", "signal_region1"])
    _write_stores_list(output_2, ["control_region2", "signal_region2"])

    with pytest.raises(ValueError) as exc_info:
        _validate_fiber_regions_consistent_for_group(np.array([str(output_1), str(output_2)]))
    message = str(exc_info.value)
    # Session folder names
    assert "session1" in message
    assert "session2" in message
    # Store IDs for each session
    assert "control_region1" in message
    assert "signal_region1" in message
    assert "control_region2" in message
    assert "signal_region2" in message


def test_validate_fiber_regions_single_session_does_not_raise(tmp_path):
    output_1 = tmp_path / "session1" / "session1_output_1"
    _write_stores_list(output_1, ["control_DMS", "signal_DMS"])

    # Single session → trivially consistent
    _validate_fiber_regions_consistent_for_group(np.array([str(output_1)]))


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

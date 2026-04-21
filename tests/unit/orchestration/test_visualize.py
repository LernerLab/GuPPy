import os

import numpy as np
import pytest

from guppy.orchestration.visualize import _validate_store_names_across_sessions, visualizeResults


# ---------------------------------------------------------------------------
# _validate_store_names_across_sessions — unit tests
# ---------------------------------------------------------------------------


def test_validate_returns_empty_when_all_store_names_present():
    sessions_stores = [
        ("/path/to/session_A_output_1", {"signal_DMS", "control_DMS", "lever_press"}),
        ("/path/to/session_B_output_1", {"signal_DMS", "control_DMS", "lever_press"}),
    ]
    combined = ["control_DMS", "lever_press", "signal_DMS"]

    result = _validate_store_names_across_sessions(sessions_stores, combined)

    assert result == []


def test_validate_detects_store_name_missing_in_one_session():
    sessions_stores = [
        ("/path/to/session_A_output_1", {"signal_DMS", "control_DMS", "lever_press"}),
        ("/path/to/session_B_output_1", {"signal_DMS", "control_DMS"}),
    ]
    combined = ["control_DMS", "lever_press", "signal_DMS"]

    result = _validate_store_names_across_sessions(sessions_stores, combined)

    assert len(result) == 1
    assert result[0] == ("/path/to/session_B_output_1", "lever_press")


def test_validate_detects_multiple_missing_cells():
    sessions_stores = [
        ("/path/to/session_A_output_1", {"signal_DMS", "control_DMS", "event1"}),
        ("/path/to/session_B_output_1", {"signal_DMS", "control_DMS", "event2"}),
    ]
    combined = ["control_DMS", "event1", "event2", "signal_DMS"]

    result = _validate_store_names_across_sessions(sessions_stores, combined)

    missing_pairs = set(result)
    assert ("/path/to/session_A_output_1", "event2") in missing_pairs
    assert ("/path/to/session_B_output_1", "event1") in missing_pairs
    assert len(result) == 2


def test_validate_returns_empty_for_single_session():
    sessions_stores = [
        ("/path/to/session_A_output_1", {"signal_DMS", "control_DMS", "lever_press"}),
    ]
    combined = ["control_DMS", "lever_press", "signal_DMS"]

    result = _validate_store_names_across_sessions(sessions_stores, combined)

    assert result == []


# ---------------------------------------------------------------------------
# visualizeResults — combine_data path raises ValueError on store name mismatch
# ---------------------------------------------------------------------------


def _write_stores_list(directory, raw_names, semantic_names):
    """Write a storesList.csv into *directory* and return the directory path."""
    os.makedirs(directory, exist_ok=True)
    rows = np.array([raw_names, semantic_names], dtype=str)
    np.savetxt(os.path.join(directory, "storesList.csv"), rows, fmt="%s", delimiter=",")
    return directory


def _make_combine_data_params(base_dir, folder_names):
    return {
        "folderNames": folder_names,
        "folderNamesForAvg": [],
        "combine_data": True,
        "visualizeAverageResults": False,
        "visualize_zscore_or_dff": "z_score",
        "abspath": base_dir,
        "nSecPrev": 5.0,
        "nSecPost": 10.0,
    }


def test_visualize_results_raises_on_mismatched_store_names(tmp_path, monkeypatch):
    """visualizeResults should raise ValueError when sessions have different store names."""
    # Session A: has lever_press; Session B: does not
    session_a_out = str(tmp_path / "session_A" / "session_A_output_1")
    session_b_out = str(tmp_path / "session_B" / "session_B_output_1")

    _write_stores_list(session_a_out, ["Dv1A", "Dv2A", "PulA"], ["control_DMS", "signal_DMS", "lever_press"])
    _write_stores_list(session_b_out, ["Dv1A", "Dv2A"], ["control_DMS", "signal_DMS"])

    session_a = str(tmp_path / "session_A")
    session_b = str(tmp_path / "session_B")

    # Patch get_all_stores_for_combining_data to return the two output dirs as one group
    monkeypatch.setattr(
        "guppy.orchestration.visualize.get_all_stores_for_combining_data",
        lambda paths: [[session_a_out, session_b_out]],
    )

    params = _make_combine_data_params(str(tmp_path), [session_a, session_b])

    with pytest.raises(ValueError, match="Store name mismatch"):
        visualizeResults(params)


def test_visualize_results_raises_lists_missing_session_and_name(tmp_path, monkeypatch):
    """The ValueError message identifies which session and which store name are missing."""
    session_a_out = str(tmp_path / "session_A" / "session_A_output_1")
    session_b_out = str(tmp_path / "session_B" / "session_B_output_1")

    _write_stores_list(session_a_out, ["Dv1A", "Dv2A", "PulA"], ["control_DMS", "signal_DMS", "lever_press"])
    _write_stores_list(session_b_out, ["Dv1A", "Dv2A"], ["control_DMS", "signal_DMS"])

    monkeypatch.setattr(
        "guppy.orchestration.visualize.get_all_stores_for_combining_data",
        lambda paths: [[session_a_out, session_b_out]],
    )

    params = _make_combine_data_params(str(tmp_path), [str(tmp_path / "session_A"), str(tmp_path / "session_B")])

    with pytest.raises(ValueError) as exc_info:
        visualizeResults(params)

    message = str(exc_info.value)
    assert "lever_press" in message
    assert session_b_out in message


def test_visualize_results_combine_data_proceeds_when_store_names_match(tmp_path, monkeypatch):
    """visualizeResults does NOT raise when all sessions share the same store names."""
    session_a_out = str(tmp_path / "session_A" / "session_A_output_1")
    session_b_out = str(tmp_path / "session_B" / "session_B_output_1")

    _write_stores_list(session_a_out, ["Dv1A", "Dv2A", "PulA"], ["control_DMS", "signal_DMS", "lever_press"])
    _write_stores_list(session_b_out, ["Dv1A", "Dv2A", "PulA"], ["control_DMS", "signal_DMS", "lever_press"])

    monkeypatch.setattr(
        "guppy.orchestration.visualize.get_all_stores_for_combining_data",
        lambda paths: [[session_a_out, session_b_out]],
    )

    create_plots_calls = []
    monkeypatch.setattr(
        "guppy.orchestration.visualize.createPlots",
        lambda filepath, event, inputParameters: create_plots_calls.append(filepath),
    )

    params = _make_combine_data_params(str(tmp_path), [str(tmp_path / "session_A"), str(tmp_path / "session_B")])
    visualizeResults(params)

    assert len(create_plots_calls) == 1, "createPlots should have been called once"
    assert create_plots_calls[0] == session_a_out
"""
Unit tests for guppy.orchestration.visualize._validate_metric_against_step5_outputs.
"""

import re

import pytest

from guppy.orchestration.visualize import _validate_metric_against_step5_outputs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_dir(tmp_path, name="session1"):
    """Return a session directory that contains one output sub-directory."""
    session_dir = tmp_path / name
    session_dir.mkdir(parents=True, exist_ok=True)
    output_dir = session_dir / f"{name}_output_1"
    output_dir.mkdir(parents=True, exist_ok=True)
    return session_dir, output_dir


def _base_params(session_dir, *, visualize_zscore_or_dff="z_score"):
    """Minimal inputParameters dict for _validate_metric_against_step5_outputs."""
    return {
        "folderNames": [str(session_dir)],
        "folderNamesForAvg": [],
        "visualizeAverageResults": False,
        "combine_data": False,
        "visualize_zscore_or_dff": visualize_zscore_or_dff,
    }


# ---------------------------------------------------------------------------
# z_score metric
# ---------------------------------------------------------------------------


def test_passes_when_z_score_psth_files_present(tmp_path):
    session_dir, output_dir = _make_session_dir(tmp_path)
    # Simulate a step-5 PSTH output file for z_score
    (output_dir / "ttl_region_z_score_region.h5").write_bytes(b"")
    params = _base_params(session_dir, visualize_zscore_or_dff="z_score")
    # Should not raise
    _validate_metric_against_step5_outputs(params)


def test_raises_when_z_score_psth_files_missing(tmp_path):
    session_dir, output_dir = _make_session_dir(tmp_path)
    # Only dff PSTH files exist
    (output_dir / "ttl_region_dff_region.h5").write_bytes(b"")
    params = _base_params(session_dir, visualize_zscore_or_dff="z_score")
    with pytest.raises(ValueError, match="z_score"):
        _validate_metric_against_step5_outputs(params)


def test_raises_names_missing_session_in_message(tmp_path):
    session_dir, output_dir = _make_session_dir(tmp_path)
    (output_dir / "ttl_region_dff_region.h5").write_bytes(b"")
    params = _base_params(session_dir, visualize_zscore_or_dff="z_score")
    with pytest.raises(ValueError, match=re.escape(str(output_dir))):
        _validate_metric_against_step5_outputs(params)


def test_raises_suggests_alternative_metric(tmp_path):
    session_dir, output_dir = _make_session_dir(tmp_path)
    (output_dir / "ttl_region_dff_region.h5").write_bytes(b"")
    params = _base_params(session_dir, visualize_zscore_or_dff="z_score")
    with pytest.raises(ValueError, match="dff"):
        _validate_metric_against_step5_outputs(params)


def test_raises_suggests_rerun_step5(tmp_path):
    session_dir, output_dir = _make_session_dir(tmp_path)
    (output_dir / "ttl_region_dff_region.h5").write_bytes(b"")
    params = _base_params(session_dir, visualize_zscore_or_dff="z_score")
    with pytest.raises(ValueError, match="Re-run step 5"):
        _validate_metric_against_step5_outputs(params)


# ---------------------------------------------------------------------------
# dff metric
# ---------------------------------------------------------------------------


def test_passes_when_dff_psth_files_present(tmp_path):
    session_dir, output_dir = _make_session_dir(tmp_path)
    (output_dir / "ttl_region_dff_region.h5").write_bytes(b"")
    params = _base_params(session_dir, visualize_zscore_or_dff="dff")
    _validate_metric_against_step5_outputs(params)


def test_raises_when_dff_psth_files_missing(tmp_path):
    session_dir, output_dir = _make_session_dir(tmp_path)
    (output_dir / "ttl_region_z_score_region.h5").write_bytes(b"")
    params = _base_params(session_dir, visualize_zscore_or_dff="dff")
    with pytest.raises(ValueError, match="dff"):
        _validate_metric_against_step5_outputs(params)


# ---------------------------------------------------------------------------
# Step-4 .hdf5 files must NOT count as PSTH outputs
# ---------------------------------------------------------------------------


def test_step4_hdf5_files_do_not_satisfy_check(tmp_path):
    """z_score_region.hdf5 (step-4 output) must not be mistaken for a PSTH file."""
    session_dir, output_dir = _make_session_dir(tmp_path)
    # Only the step-4 intermediate file exists, no step-5 PSTH .h5 file
    (output_dir / "z_score_region.hdf5").write_bytes(b"")
    params = _base_params(session_dir, visualize_zscore_or_dff="z_score")
    with pytest.raises(ValueError, match="z_score"):
        _validate_metric_against_step5_outputs(params)


# ---------------------------------------------------------------------------
# "Both" metric produced in step 5
# ---------------------------------------------------------------------------


def test_passes_for_z_score_when_both_psth_files_present(tmp_path):
    session_dir, output_dir = _make_session_dir(tmp_path)
    (output_dir / "ttl_region_z_score_region.h5").write_bytes(b"")
    (output_dir / "ttl_region_dff_region.h5").write_bytes(b"")
    params = _base_params(session_dir, visualize_zscore_or_dff="z_score")
    _validate_metric_against_step5_outputs(params)


def test_passes_for_dff_when_both_psth_files_present(tmp_path):
    session_dir, output_dir = _make_session_dir(tmp_path)
    (output_dir / "ttl_region_z_score_region.h5").write_bytes(b"")
    (output_dir / "ttl_region_dff_region.h5").write_bytes(b"")
    params = _base_params(session_dir, visualize_zscore_or_dff="dff")
    _validate_metric_against_step5_outputs(params)


# ---------------------------------------------------------------------------
# Multiple sessions — partial failure
# ---------------------------------------------------------------------------


def test_raises_only_missing_sessions_are_reported(tmp_path):
    session1_dir, output1_dir = _make_session_dir(tmp_path, "session1")
    session2_dir, output2_dir = _make_session_dir(tmp_path, "session2")

    # session1 has z_score psth; session2 does not
    (output1_dir / "ttl_region_z_score_region.h5").write_bytes(b"")
    (output2_dir / "ttl_region_dff_region.h5").write_bytes(b"")

    params = {
        "folderNames": [str(session1_dir), str(session2_dir)],
        "folderNamesForAvg": [],
        "visualizeAverageResults": False,
        "combine_data": False,
        "visualize_zscore_or_dff": "z_score",
    }

    with pytest.raises(ValueError) as exc_info:
        _validate_metric_against_step5_outputs(params)

    message = str(exc_info.value)
    assert str(output2_dir) in message
    assert str(output1_dir) not in message
    assert "1 session" in message


# ---------------------------------------------------------------------------
# No output directories found — graceful no-op
# ---------------------------------------------------------------------------


def test_no_op_when_no_output_directories(tmp_path):
    """When no *_output_* dirs exist the function returns silently."""
    session_dir = tmp_path / "empty_session"
    session_dir.mkdir()
    params = _base_params(session_dir, visualize_zscore_or_dff="z_score")
    # Should not raise
    _validate_metric_against_step5_outputs(params)

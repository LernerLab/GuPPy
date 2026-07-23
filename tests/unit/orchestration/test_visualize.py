"""
Unit tests for guppy.orchestration.visualize._validate_metric_against_step4_outputs.
"""

import re

import pytest

from guppy.orchestration.visualize import (
    _validate_average_visualization_preconditions,
    _validate_metric_against_step4_outputs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_dir(tmp_path, name="session1"):
    """Return a session directory that contains one output sub-directory."""
    session_dir = tmp_path / name
    session_dir.mkdir(parents=True, exist_ok=True)
    run_folder = session_dir / f"{name}_output_1"
    run_folder.mkdir(parents=True, exist_ok=True)
    # select_run_folders validates that picked outputs have a storesList.csv (re-run step 1 if missing).
    (run_folder / "storesList.csv").write_text("")
    return session_dir, run_folder


def _base_params(session_dir, *, visualize_zscore_or_dff="z_score", selected_runs=("1",)):
    """Minimal inputParameters dict for _validate_metric_against_step4_outputs."""
    return {
        "session_folders": [str(session_dir)],
        "group_session_folders": [],
        "visualizeAverageResults": False,
        "combine_data": False,
        "visualize_zscore_or_dff": visualize_zscore_or_dff,
        "selected_runs": {str(session_dir): list(selected_runs)},
    }


# ---------------------------------------------------------------------------
# z_score metric
# ---------------------------------------------------------------------------


def test_passes_when_z_score_psth_files_present(tmp_path):
    session_dir, run_folder = _make_session_dir(tmp_path)
    # Simulate a step-4 PSTH output file for z_score
    (run_folder / "ttl_region_z_score_region.h5").write_bytes(b"")
    params = _base_params(session_dir, visualize_zscore_or_dff="z_score")
    # Should not raise
    _validate_metric_against_step4_outputs(params)


def test_raises_when_z_score_psth_files_missing(tmp_path):
    session_dir, run_folder = _make_session_dir(tmp_path)
    # Only dff PSTH files exist
    (run_folder / "ttl_region_dff_region.h5").write_bytes(b"")
    params = _base_params(session_dir, visualize_zscore_or_dff="z_score")
    with pytest.raises(ValueError, match="z_score"):
        _validate_metric_against_step4_outputs(params)


def test_raises_names_missing_session_in_message(tmp_path):
    session_dir, run_folder = _make_session_dir(tmp_path)
    (run_folder / "ttl_region_dff_region.h5").write_bytes(b"")
    params = _base_params(session_dir, visualize_zscore_or_dff="z_score")
    with pytest.raises(ValueError, match=re.escape(str(run_folder))):
        _validate_metric_against_step4_outputs(params)


def test_raises_suggests_alternative_metric(tmp_path):
    session_dir, run_folder = _make_session_dir(tmp_path)
    (run_folder / "ttl_region_dff_region.h5").write_bytes(b"")
    params = _base_params(session_dir, visualize_zscore_or_dff="z_score")
    with pytest.raises(ValueError, match="dff"):
        _validate_metric_against_step4_outputs(params)


def test_raises_suggests_rerun_step4(tmp_path):
    session_dir, run_folder = _make_session_dir(tmp_path)
    (run_folder / "ttl_region_dff_region.h5").write_bytes(b"")
    params = _base_params(session_dir, visualize_zscore_or_dff="z_score")
    with pytest.raises(ValueError, match="Re-run step 4"):
        _validate_metric_against_step4_outputs(params)


# ---------------------------------------------------------------------------
# dff metric
# ---------------------------------------------------------------------------


def test_passes_when_dff_psth_files_present(tmp_path):
    session_dir, run_folder = _make_session_dir(tmp_path)
    (run_folder / "ttl_region_dff_region.h5").write_bytes(b"")
    params = _base_params(session_dir, visualize_zscore_or_dff="dff")
    _validate_metric_against_step4_outputs(params)


def test_raises_when_dff_psth_files_missing(tmp_path):
    session_dir, run_folder = _make_session_dir(tmp_path)
    (run_folder / "ttl_region_z_score_region.h5").write_bytes(b"")
    params = _base_params(session_dir, visualize_zscore_or_dff="dff")
    with pytest.raises(ValueError, match="dff"):
        _validate_metric_against_step4_outputs(params)


# ---------------------------------------------------------------------------
# Step-3 .hdf5 files must NOT count as PSTH outputs
# ---------------------------------------------------------------------------


def test_step3_hdf5_files_do_not_satisfy_check(tmp_path):
    """z_score_region.hdf5 (step-3 output) must not be mistaken for a PSTH file."""
    session_dir, run_folder = _make_session_dir(tmp_path)
    # Only the step-3 intermediate file exists, no step-4 PSTH .h5 file
    (run_folder / "z_score_region.hdf5").write_bytes(b"")
    params = _base_params(session_dir, visualize_zscore_or_dff="z_score")
    with pytest.raises(ValueError, match="z_score"):
        _validate_metric_against_step4_outputs(params)


# ---------------------------------------------------------------------------
# "Both" metric produced in step 4
# ---------------------------------------------------------------------------


def test_passes_for_z_score_when_both_psth_files_present(tmp_path):
    session_dir, run_folder = _make_session_dir(tmp_path)
    (run_folder / "ttl_region_z_score_region.h5").write_bytes(b"")
    (run_folder / "ttl_region_dff_region.h5").write_bytes(b"")
    params = _base_params(session_dir, visualize_zscore_or_dff="z_score")
    _validate_metric_against_step4_outputs(params)


def test_passes_for_dff_when_both_psth_files_present(tmp_path):
    session_dir, run_folder = _make_session_dir(tmp_path)
    (run_folder / "ttl_region_z_score_region.h5").write_bytes(b"")
    (run_folder / "ttl_region_dff_region.h5").write_bytes(b"")
    params = _base_params(session_dir, visualize_zscore_or_dff="dff")
    _validate_metric_against_step4_outputs(params)


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
        "session_folders": [str(session1_dir), str(session2_dir)],
        "group_session_folders": [],
        "visualizeAverageResults": False,
        "combine_data": False,
        "visualize_zscore_or_dff": "z_score",
        "selected_runs": {str(session1_dir): ["1"], str(session2_dir): ["1"]},
    }

    with pytest.raises(ValueError) as exc_info:
        _validate_metric_against_step4_outputs(params)

    message = str(exc_info.value)
    assert str(output2_dir) in message
    assert str(output1_dir) not in message
    assert "1 session" in message


# ---------------------------------------------------------------------------
# No output directories found — graceful no-op
# ---------------------------------------------------------------------------


def test_no_op_when_no_output_directories(tmp_path):
    """When no *_output_* dirs exist the function returns silently.

    The homepage gate `validate_selected_runs_for_consumers` skips sessions with
    no output dirs on disk, so they reach orchestration with no entry in
    selected_runs. The validator must skip such sessions instead of erroring.
    """
    session_dir = tmp_path / "empty_session"
    session_dir.mkdir()
    params = _base_params(session_dir, visualize_zscore_or_dff="z_score", selected_runs=())
    # Empty selected_runs simulates "session not picked in the FileSelector" — should not raise.
    _validate_metric_against_step4_outputs(params)


# ---------------------------------------------------------------------------
# _validate_average_visualization_preconditions
# ---------------------------------------------------------------------------


def _avg_params(
    tmp_path,
    *,
    visualize_average_results=True,
    folder_names_for_avg=None,
    visualize_zscore_or_dff="z_score",
):
    return {
        "abspath": str(tmp_path),
        "visualizeAverageResults": visualize_average_results,
        "group_session_folders": folder_names_for_avg if folder_names_for_avg is not None else [],
        "visualize_zscore_or_dff": visualize_zscore_or_dff,
    }


def test_precondition_noop_when_visualize_average_false(tmp_path):
    """When visualizeAverageResults=False, no checks are run — even without folders."""
    params = _avg_params(tmp_path, visualize_average_results=False)
    _validate_average_visualization_preconditions(params)


def test_precondition_raises_when_no_folders_selected_for_avg(tmp_path):
    """visualizeAverageResults=True with empty group_session_folders → actionable error."""
    params = _avg_params(tmp_path, folder_names_for_avg=[])
    with pytest.raises(ValueError, match="no folders are selected"):
        _validate_average_visualization_preconditions(params)


def test_precondition_raises_when_average_folder_missing(tmp_path):
    """visualizeAverageResults=True but no `average/` folder on disk → actionable error."""
    session_dir = tmp_path / "session1"
    session_dir.mkdir()
    params = _avg_params(tmp_path, folder_names_for_avg=[str(session_dir)])
    with pytest.raises(ValueError, match="no 'average' directory"):
        _validate_average_visualization_preconditions(params)


def test_precondition_raises_when_average_folder_empty_of_psth_outputs(tmp_path):
    """`average/` exists but has no PSTH .h5 outputs for the requested metric → actionable error."""
    session_dir = tmp_path / "session1"
    session_dir.mkdir()
    (tmp_path / "average").mkdir()
    params = _avg_params(tmp_path, folder_names_for_avg=[str(session_dir)])
    with pytest.raises(ValueError, match="no PSTH outputs for the 'z_score' metric"):
        _validate_average_visualization_preconditions(params)


def test_precondition_passes_when_average_folder_has_z_score_psth(tmp_path):
    session_dir = tmp_path / "session1"
    session_dir.mkdir()
    average_dir = tmp_path / "average"
    average_dir.mkdir()
    (average_dir / "ttl_region_z_score_region.h5").write_bytes(b"")
    params = _avg_params(tmp_path, folder_names_for_avg=[str(session_dir)], visualize_zscore_or_dff="z_score")
    _validate_average_visualization_preconditions(params)


def test_precondition_passes_when_average_folder_has_dff_psth(tmp_path):
    session_dir = tmp_path / "session1"
    session_dir.mkdir()
    average_dir = tmp_path / "average"
    average_dir.mkdir()
    (average_dir / "ttl_region_dff_region.h5").write_bytes(b"")
    params = _avg_params(tmp_path, folder_names_for_avg=[str(session_dir)], visualize_zscore_or_dff="dff")
    _validate_average_visualization_preconditions(params)

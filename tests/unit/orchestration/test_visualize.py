"""
Unit tests for guppy.orchestration.visualize._validate_metric_against_step5_outputs.
"""

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
    with pytest.raises(ValueError, match=str(output_dir)):
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

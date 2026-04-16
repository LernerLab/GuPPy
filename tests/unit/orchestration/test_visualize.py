"""Unit tests for the visualization metric validation in visualize.py."""

import json
import os

import pytest

from guppy.orchestration.visualize import _validate_visualization_metric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_params(folder: str, select_for_compute_psth: str) -> None:
    """Write a minimal GuPPyParamtersUsed.json into *folder*."""
    params = {
        "combine_data": False,
        "isosbestic_control": True,
        "timeForLightsTurnOn": 1,
        "filter_window": 100,
        "removeArtifacts": False,
        "noChannels": 2,
        "zscore_method": "standard z-score",
        "baselineWindowStart": 0,
        "baselineWindowEnd": 0,
        "nSecPrev": -10,
        "nSecPost": 20,
        "timeInterval": 2,
        "bin_psth_trials": 0,
        "use_time_or_trials": "Time (min)",
        "baselineCorrectionStart": -5,
        "baselineCorrectionEnd": 0,
        "peak_startPoint": [0.0],
        "peak_endPoint": [5.0],
        "selectForComputePsth": select_for_compute_psth,
        "selectForTransientsComputation": select_for_compute_psth,
        "moving_window": 15,
        "highAmpFilt": 2,
        "transientsThresh": 3,
    }
    with open(os.path.join(folder, "GuPPyParamtersUsed.json"), "w") as fh:
        json.dump(params, fh)


def _make_output_dir(session_dir: str) -> str:
    """Create a ``<session>_output_1`` subdirectory and return its path."""
    output_dir = os.path.join(session_dir, f"{os.path.basename(session_dir)}_output_1")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _base_params(session_dir: str, *, visualize: str, average: bool = False) -> dict:
    return {
        "folderNames": [session_dir],
        "folderNamesForAvg": [],
        "visualize_zscore_or_dff": visualize,
        "visualizeAverageResults": average,
        "combine_data": False,
    }


# ---------------------------------------------------------------------------
# Tests: JSON-based primary check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "select_for_psth, visualize, should_raise",
    [
        # Matching selections — no error
        ("z_score", "z_score", False),
        ("dff", "dff", False),
        # "Both" computed — any visualization choice is valid
        ("Both", "z_score", False),
        ("Both", "dff", False),
        # Mismatch — must raise
        ("z_score", "dff", True),
        ("dff", "z_score", True),
    ],
)
def test_json_based_check(tmp_path, select_for_psth, visualize, should_raise):
    """_validate_visualization_metric raises iff the metric is missing."""
    session = tmp_path / "session1"
    session.mkdir()
    _make_output_dir(str(session))
    _write_params(str(session), select_for_psth)

    params = _base_params(str(session), visualize=visualize)

    if should_raise:
        with pytest.raises(ValueError) as exc_info:
            _validate_visualization_metric(params)
        msg = str(exc_info.value)
        # The message must identify the session
        assert str(session) in msg
        # The message must name the requested metric
        assert visualize in msg
        # The message must provide actionable instructions
        assert "step 5" in msg.lower()
        assert "z_score and/or" in msg or "re-run" in msg.lower()
    else:
        _validate_visualization_metric(params)  # should not raise


def test_error_message_contains_other_metric_hint(tmp_path):
    """The error message suggests the correct alternative metric."""
    session = tmp_path / "session1"
    session.mkdir()
    _make_output_dir(str(session))
    _write_params(str(session), "z_score")

    params = _base_params(str(session), visualize="dff")

    with pytest.raises(ValueError) as exc_info:
        _validate_visualization_metric(params)
    msg = str(exc_info.value)
    # Should suggest switching to 'z_score' (what was computed)
    assert "'z_score'" in msg


def test_error_lists_all_missing_sessions(tmp_path):
    """All affected session paths are listed in the error message."""
    sessions = []
    for name in ("sess_a", "sess_b"):
        s = tmp_path / name
        s.mkdir()
        _make_output_dir(str(s))
        _write_params(str(s), "z_score")
        sessions.append(str(s))

    params = {
        "folderNames": sessions,
        "folderNamesForAvg": [],
        "visualize_zscore_or_dff": "dff",
        "visualizeAverageResults": False,
        "combine_data": False,
    }

    with pytest.raises(ValueError) as exc_info:
        _validate_visualization_metric(params)
    msg = str(exc_info.value)
    for s in sessions:
        assert s in msg


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


def test_no_output_dirs_skips_validation(tmp_path):
    """If step 5 has not been run (no output dirs), validation is skipped."""
    session = tmp_path / "session1"
    session.mkdir()
    # Do NOT create an output dir — step 5 was never run.
    _write_params(str(session), "z_score")

    params = _base_params(str(session), visualize="dff")
    _validate_visualization_metric(params)  # must not raise


def test_no_json_file_falls_back_to_file_check_ok(tmp_path):
    """When JSON is absent, PSTH files matching the metric are accepted."""
    session = tmp_path / "session1"
    session.mkdir()
    output_dir = _make_output_dir(str(session))
    # Simulate a z_score PSTH file (event_region_z_score_region.h5)
    open(os.path.join(output_dir, "ttl_region_z_score_region.h5"), "w").close()

    params = _base_params(str(session), visualize="z_score")
    _validate_visualization_metric(params)  # must not raise


def test_no_json_file_falls_back_to_file_check_raises(tmp_path):
    """When JSON is absent and no metric PSTH files exist, an error is raised."""
    session = tmp_path / "session1"
    session.mkdir()
    output_dir = _make_output_dir(str(session))
    # Only dff PSTH files exist (no z_score PSTH files)
    open(os.path.join(output_dir, "ttl_region_dff_region.h5"), "w").close()

    params = _base_params(str(session), visualize="z_score")
    with pytest.raises(ValueError) as exc_info:
        _validate_visualization_metric(params)
    assert "z_score" in str(exc_info.value)


def test_raw_metric_files_not_mistaken_for_psth(tmp_path):
    """Raw z_score_*.h5 files (not PSTH) must not satisfy the file-based check."""
    session = tmp_path / "session1"
    session.mkdir()
    output_dir = _make_output_dir(str(session))
    # Raw file starts with the metric prefix — not a PSTH file
    open(os.path.join(output_dir, "z_score_region.h5"), "w").close()

    params = _base_params(str(session), visualize="z_score")
    # The fallback check should NOT accept this raw file as a PSTH file.
    with pytest.raises(ValueError):
        _validate_visualization_metric(params)


def test_unknown_metric_value_skips_validation(tmp_path):
    """Unrecognised metric values are silently ignored (no crash)."""
    session = tmp_path / "session1"
    session.mkdir()
    _make_output_dir(str(session))
    _write_params(str(session), "z_score")

    params = _base_params(str(session), visualize="unknown_value")
    _validate_visualization_metric(params)  # must not raise


def test_corrupted_json_falls_back_to_file_check(tmp_path):
    """A corrupted JSON file causes the fallback file check to be used."""
    session = tmp_path / "session1"
    session.mkdir()
    output_dir = _make_output_dir(str(session))
    # Write a broken JSON file
    with open(os.path.join(str(session), "GuPPyParamtersUsed.json"), "w") as fh:
        fh.write("{not valid json")
    # Simulate a z_score PSTH file so the file-based check passes
    open(os.path.join(output_dir, "ttl_region_z_score_region.h5"), "w").close()

    params = _base_params(str(session), visualize="z_score")
    _validate_visualization_metric(params)  # must not raise


# ---------------------------------------------------------------------------
# Tests: average visualisation uses folderNamesForAvg
# ---------------------------------------------------------------------------


def test_average_mode_checks_folder_names_for_avg(tmp_path):
    """In average mode, folderNamesForAvg folders are checked, not folderNames."""
    # Session used for average analysis — has correct metric
    avg_session = tmp_path / "avg_sess"
    avg_session.mkdir()
    _make_output_dir(str(avg_session))
    _write_params(str(avg_session), "z_score")

    # Individual session listed in folderNames — has WRONG metric
    # (should NOT trigger a validation error when average=True)
    ind_session = tmp_path / "ind_sess"
    ind_session.mkdir()
    _make_output_dir(str(ind_session))
    _write_params(str(ind_session), "dff")

    params = {
        "folderNames": [str(ind_session)],
        "folderNamesForAvg": [str(avg_session)],
        "visualize_zscore_or_dff": "z_score",
        "visualizeAverageResults": True,
        "combine_data": False,
    }
    _validate_visualization_metric(params)  # must not raise (avg_session is fine)

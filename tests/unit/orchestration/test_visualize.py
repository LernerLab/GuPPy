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

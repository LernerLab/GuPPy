import json
import os
import types

import numpy as np
import pytest

from guppy.orchestration.storenames import _fetchValues, _save, make_dir, show_dir

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_widget(value):
    """Return a lightweight mock Panel widget with only a .value attribute."""
    return types.SimpleNamespace(value=value)


class FakePath:
    """Replaces pathlib.Path in storenames so cache writes go to tmp_path."""

    _home = None

    @classmethod
    def home(cls):
        return cls._home


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Redirect Path.home() to tmp_path so _save never touches ~/.storesList.json."""
    FakePath._home = tmp_path
    monkeypatch.setattr("guppy.orchestration.storenames.Path", FakePath)
    return tmp_path


# ---------------------------------------------------------------------------
# show_dir
# ---------------------------------------------------------------------------


def test_show_dir_returns_output_1_when_no_existing_dirs(tmp_path):
    session = tmp_path / "session1"
    session.mkdir()

    result = show_dir(str(session))

    assert result == str(session / "session1_output_1")


def test_show_dir_returns_output_3_when_output_1_and_2_exist(tmp_path):
    session = tmp_path / "session1"
    session.mkdir()
    (session / "session1_output_1").mkdir()
    (session / "session1_output_2").mkdir()

    result = show_dir(str(session))

    assert result == str(session / "session1_output_3")


def test_show_dir_sequential_numbering_no_gap_filling(tmp_path):
    """show_dir counts up from 1; if only _output_1 exists it returns _output_2."""
    session = tmp_path / "session1"
    session.mkdir()
    (session / "session1_output_1").mkdir()

    result = show_dir(str(session))

    assert result == str(session / "session1_output_2")


# ---------------------------------------------------------------------------
# make_dir
# ---------------------------------------------------------------------------


def test_make_dir_creates_directory(tmp_path):
    session = tmp_path / "session1"
    session.mkdir()

    result = make_dir(str(session))

    assert os.path.isdir(result)


def test_make_dir_returns_correct_path(tmp_path):
    session = tmp_path / "session1"
    session.mkdir()

    result = make_dir(str(session))

    assert result == str(session / "session1_output_1")


def test_make_dir_increments_when_previous_exists(tmp_path):
    session = tmp_path / "session1"
    session.mkdir()
    (session / "session1_output_1").mkdir()

    result = make_dir(str(session))

    assert result == str(session / "session1_output_2")
    assert os.path.isdir(result)


# ---------------------------------------------------------------------------
# _save
# ---------------------------------------------------------------------------


def test_save_writes_storeslist_csv(isolated_cache):
    select_location = str(isolated_cache / "session1_output_1")
    storenames_data = {
        "storenames": ["Dv1A", "Dv2A", "PulA"],
        "names_for_storenames": ["control_DMS", "signal_DMS", "event1"],
    }

    result = _save(storenames_data, select_location)

    assert result == "#### No alerts !!"
    assert os.path.exists(os.path.join(select_location, "storesList.csv"))


def test_save_csv_content_matches_input(isolated_cache):
    select_location = str(isolated_cache / "session1_output_1")
    storenames_data = {
        "storenames": ["Dv1A", "Dv2A", "PulA"],
        "names_for_storenames": ["control_DMS", "signal_DMS", "event1"],
    }

    _save(storenames_data, select_location)

    loaded = np.loadtxt(os.path.join(select_location, "storesList.csv"), delimiter=",", dtype=str)
    np.testing.assert_array_equal(loaded[0], ["Dv1A", "Dv2A", "PulA"])
    np.testing.assert_array_equal(loaded[1], ["control_DMS", "signal_DMS", "event1"])


def test_save_returns_alert_when_shapes_mismatch(isolated_cache):
    select_location = str(isolated_cache / "session1_output_1")
    storenames_data = {
        "storenames": ["Dv1A", "Dv2A"],
        "names_for_storenames": ["control_DMS"],  # length mismatch
    }

    result = _save(storenames_data, select_location)

    assert "Alert" in result


def test_save_returns_alert_when_empty_string_in_names(isolated_cache):
    select_location = str(isolated_cache / "session1_output_1")
    storenames_data = {
        "storenames": ["Dv1A", "Dv2A"],
        "names_for_storenames": ["control_DMS", ""],  # empty string
    }

    result = _save(storenames_data, select_location)

    assert "Alert" in result


def test_save_updates_cache_file(isolated_cache):
    select_location = str(isolated_cache / "session1_output_1")
    storenames_data = {
        "storenames": ["Dv1A", "Dv2A"],
        "names_for_storenames": ["control_DMS", "signal_DMS"],
    }

    _save(storenames_data, select_location)

    cache_path = isolated_cache / ".storesList.json"
    assert cache_path.exists()
    with open(cache_path) as file:
        cache = json.load(file)
    assert "Dv1A" in cache
    assert "control_DMS" in cache["Dv1A"]
    assert "Dv2A" in cache
    assert "signal_DMS" in cache["Dv2A"]


# ---------------------------------------------------------------------------
# _fetchValues
# ---------------------------------------------------------------------------


def _build_fetchValues_args(dropdown_values, textbox_values, text_value=None):
    """Build the storename_dropdowns, storename_textboxes, and text widget."""
    storenames = list(dropdown_values.keys())
    storename_dropdowns = {key: make_widget(dropdown_values[key]) for key in storenames}
    storename_textboxes = {key: make_widget(textbox_values[key]) for key in storenames}
    text = make_widget(text_value if text_value is not None else storenames)
    return text, storenames, storename_dropdowns, storename_textboxes


def test_fetchValues_returns_alert_when_storenames_empty():
    text = make_widget([])
    storenames = []
    storename_dropdowns = {}
    storename_textboxes = {}
    result = _fetchValues(text, storenames, storename_dropdowns, storename_textboxes, {})
    assert "Alert" in result


def test_fetchValues_returns_alert_when_whitespace_in_textbox():
    text, storenames, dropdowns, textboxes = _build_fetchValues_args(
        dropdown_values={"Dv1A": "control"},
        textbox_values={"Dv1A": "DMS region"},  # space in name
    )
    result = _fetchValues(text, storenames, dropdowns, textboxes, {})
    assert "Alert" in result


def test_fetchValues_returns_alert_when_required_textbox_empty():
    text, storenames, dropdowns, textboxes = _build_fetchValues_args(
        dropdown_values={"Dv1A": "control"},
        textbox_values={"Dv1A": ""},  # empty textbox for required role
    )
    result = _fetchValues(text, storenames, dropdowns, textboxes, {})
    assert "Alert" in result


def test_fetchValues_returns_alert_when_underscore_in_region_name():
    text, storenames, dropdowns, textboxes = _build_fetchValues_args(
        dropdown_values={"Dv1A": "control"},
        textbox_values={"Dv1A": "DMS_region"},  # underscore in region name
    )
    result = _fetchValues(text, storenames, dropdowns, textboxes, {})
    assert "Alert" in result


def test_fetchValues_valid_control_entry_sets_correct_name():
    text, storenames, dropdowns, textboxes = _build_fetchValues_args(
        dropdown_values={"Dv1A": "control"},
        textbox_values={"Dv1A": "DMS"},
        text_value=["Dv1A"],
    )
    result_dict = {}
    result = _fetchValues(text, storenames, dropdowns, textboxes, result_dict)
    assert result == "#### No alerts !!"
    assert result_dict["names_for_storenames"] == ["control_DMS"]


def test_fetchValues_valid_signal_entry_sets_correct_name():
    text, storenames, dropdowns, textboxes = _build_fetchValues_args(
        dropdown_values={"Dv2A": "signal"},
        textbox_values={"Dv2A": "DMS"},
        text_value=["Dv2A"],
    )
    result_dict = {}
    result = _fetchValues(text, storenames, dropdowns, textboxes, result_dict)
    assert result == "#### No alerts !!"
    assert result_dict["names_for_storenames"] == ["signal_DMS"]


def test_fetchValues_valid_event_ttls_uses_textbox_value():
    text, storenames, dropdowns, textboxes = _build_fetchValues_args(
        dropdown_values={"PulA": "event TTLs"},
        textbox_values={"PulA": "lever_press"},
        text_value=["PulA"],
    )
    result_dict = {}
    result = _fetchValues(text, storenames, dropdowns, textboxes, result_dict)
    assert result == "#### No alerts !!"
    assert result_dict["names_for_storenames"] == ["lever_press"]


def test_fetchValues_non_standard_dropdown_uses_dropdown_value():
    """Dropdown values other than control/signal/event TTLs use the dropdown value directly."""
    text, storenames, dropdowns, textboxes = _build_fetchValues_args(
        dropdown_values={"PulA": "exclude"},
        textbox_values={"PulA": ""},  # empty textbox — not required for non-standard roles
        text_value=["PulA"],
    )
    result_dict = {}
    result = _fetchValues(text, storenames, dropdowns, textboxes, result_dict)
    assert result == "#### No alerts !!"
    assert result_dict["names_for_storenames"] == ["exclude"]


def test_fetchValues_populates_storenames_from_text_value():
    text_value = ["Dv1A", "Dv2A"]
    text, storenames, dropdowns, textboxes = _build_fetchValues_args(
        dropdown_values={"Dv1A": "control", "Dv2A": "signal"},
        textbox_values={"Dv1A": "DMS", "Dv2A": "DMS"},
        text_value=text_value,
    )
    result_dict = {}
    _fetchValues(text, storenames, dropdowns, textboxes, result_dict)
    assert result_dict["storenames"] == text_value

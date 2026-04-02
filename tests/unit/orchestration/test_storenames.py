import json
import os
import types

import numpy as np
import panel as pn
import pytest

import guppy.orchestration.storenames as storenames_module
from guppy.orchestration.storenames import (
    _fetchValues,
    _save,
    build_storenames_template,
    make_dir,
    show_dir,
)

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


# ---------------------------------------------------------------------------
# build_storenames_template on-click closures
# ---------------------------------------------------------------------------


class CapturingStorenamesSelector:
    """Minimal fake StorenamesSelector that captures on-click closures without rendering Panel."""

    def __init__(self, allnames):
        self.text = types.SimpleNamespace(value=[])
        self.callbacks = {}
        self.alert_message = None
        self.select_location_options = None
        self.change_widgets_value = None
        self.path_value = None
        self._literal_input_2 = {}
        self._cross_selector_value = []
        self._take_widgets = [[], []]
        self._select_location_value = ""
        self.widget = types.SimpleNamespace()
        self.configure_storenames_calls = []

    def set_select_location_options(self, options):
        self.select_location_options = options

    def set_alert_message(self, message):
        self.alert_message = message

    def get_literal_input_2(self):
        return self._literal_input_2

    def set_literal_input_2(self, d):
        self._literal_input_2 = d

    def get_take_widgets(self):
        return self._take_widgets

    def set_change_widgets(self, value):
        self.change_widgets_value = value

    def get_cross_selector(self):
        return self._cross_selector_value

    def get_select_location(self):
        return self._select_location_value

    def set_path(self, value):
        self.path_value = value

    def attach_callbacks(self, button_name_to_onclick_fn):
        self.callbacks = button_name_to_onclick_fn

    def configure_storenames(self, storename_dropdowns, storename_textboxes, storenames, storenames_cache):
        self.configure_storenames_calls.append(
            {"storenames": list(storenames), "storenames_cache": dict(storenames_cache)}
        )


class FakeStorenamesInstructions:
    def __init__(self, folder_path=None):
        self.widget = types.SimpleNamespace()


class FakeBootstrapTemplate:
    def __init__(self, title=""):
        self.main = types.SimpleNamespace(append=lambda item: None)


@pytest.fixture
def storenames_closures(tmp_path, monkeypatch, panel_extension):
    """Build build_storenames_template with faked UI; return (selector, folder_path).

    selector.callbacks maps button names to their on-click closure functions.
    """
    FakePath._home = tmp_path
    monkeypatch.setattr("guppy.orchestration.storenames.Path", FakePath)

    folder = tmp_path / "my_session"
    folder.mkdir()

    captured_selector = None

    class TrackingSelector(CapturingStorenamesSelector):
        def __init__(self, allnames):
            super().__init__(allnames)
            nonlocal captured_selector
            captured_selector = self

    monkeypatch.setattr("guppy.orchestration.storenames.StorenamesSelector", TrackingSelector)
    monkeypatch.setattr("guppy.orchestration.storenames.StorenamesInstructions", FakeStorenamesInstructions)
    monkeypatch.setattr(pn.template, "BootstrapTemplate", FakeBootstrapTemplate)
    monkeypatch.setattr(pn, "Row", lambda *args, **kwargs: None)

    build_storenames_template(["Dv1A", "Dv2A", "PulA"], [], str(folder))

    return captured_selector, str(folder)


# ---------------------------------------------------------------------------
# overwrite_button_actions
# ---------------------------------------------------------------------------


def test_overwrite_button_actions_create_new_file_sets_next_output_dir(storenames_closures):
    selector, folder_path = storenames_closures
    overwrite_button_actions = selector.callbacks["overwrite_button"]

    overwrite_button_actions(types.SimpleNamespace(new="create_new_file"))

    expected = os.path.join(folder_path, "my_session_output_1")
    assert selector.select_location_options == [expected]


def test_overwrite_button_actions_over_write_file_returns_existing_output_dirs(storenames_closures):
    selector, folder_path = storenames_closures
    overwrite_button_actions = selector.callbacks["overwrite_button"]

    output_dir = os.path.join(folder_path, "my_session_output_1")
    os.mkdir(output_dir)

    overwrite_button_actions(types.SimpleNamespace(new="over_write_file"))

    assert selector.select_location_options == [output_dir]


# ---------------------------------------------------------------------------
# fetchValues (show_config_button)
# ---------------------------------------------------------------------------


def test_fetch_values_sets_alert_message_when_no_storenames_configured(storenames_closures, monkeypatch):
    selector, _ = storenames_closures
    monkeypatch.setattr(storenames_module, "storenames", [], raising=False)

    selector.callbacks["show_config_button"](types.SimpleNamespace())

    assert "Alert" in selector.alert_message


def test_fetch_values_always_calls_set_literal_input_2(storenames_closures, monkeypatch):
    selector, _ = storenames_closures
    monkeypatch.setattr(storenames_module, "storenames", [], raising=False)

    selector.callbacks["show_config_button"](types.SimpleNamespace())

    # set_literal_input_2 must be called unconditionally (with an empty dict on alert path)
    assert selector._literal_input_2 == {}


def test_fetch_values_delegates_to_fetch_values_function(storenames_closures, monkeypatch):
    """fetchValues passes the module-level storenames to _fetchValues and relays results to the selector."""
    selector, _ = storenames_closures
    monkeypatch.setattr(storenames_module, "storenames", ["Dv1A"], raising=False)

    captured_args = {}

    def fake_fetch_values(text, storenames, storename_dropdowns, storename_textboxes, d):
        captured_args["storenames"] = list(storenames)
        d["storenames"] = storenames
        d["names_for_storenames"] = ["control_DMS"]
        return "#### No alerts !!"

    monkeypatch.setattr("guppy.orchestration.storenames._fetchValues", fake_fetch_values)

    selector.callbacks["show_config_button"](types.SimpleNamespace())

    assert captured_args["storenames"] == ["Dv1A"]
    assert selector.alert_message == "#### No alerts !!"
    assert selector._literal_input_2["names_for_storenames"] == ["control_DMS"]


# ---------------------------------------------------------------------------
# update_values (update_options button)
# ---------------------------------------------------------------------------


def test_update_values_sets_storenames_from_cross_selector(storenames_closures):
    selector, _ = storenames_closures
    selector._cross_selector_value = ["Dv1A", "Dv2A"]
    selector._take_widgets = [[], []]  # no repeated storenames

    selector.callbacks["update_options"](types.SimpleNamespace())

    assert selector.change_widgets_value == ["Dv1A", "Dv2A"]


def test_update_values_includes_repeated_storenames(storenames_closures):
    selector, _ = storenames_closures
    selector._cross_selector_value = ["Dv2A"]
    selector._take_widgets = [["Dv1A"], [2]]  # Dv1A repeated twice

    selector.callbacks["update_options"](types.SimpleNamespace())

    assert selector.change_widgets_value == ["Dv2A", "Dv1A", "Dv1A"]


def test_update_values_calls_configure_storenames_with_selected_storenames(storenames_closures):
    selector, _ = storenames_closures
    selector._cross_selector_value = ["Dv1A", "PulA"]
    selector._take_widgets = [[], []]

    selector.callbacks["update_options"](types.SimpleNamespace())

    assert len(selector.configure_storenames_calls) == 1
    assert selector.configure_storenames_calls[0]["storenames"] == ["Dv1A", "PulA"]


def test_update_values_loads_storenames_cache_when_json_file_exists(storenames_closures, tmp_path):
    selector, _ = storenames_closures
    selector._cross_selector_value = ["Dv1A"]
    selector._take_widgets = [[], []]

    cache = {"Dv1A": ["control_DMS"]}
    cache_file = tmp_path / ".storesList.json"
    with open(cache_file, "w") as f:
        json.dump(cache, f)

    selector.callbacks["update_options"](types.SimpleNamespace())

    assert selector.configure_storenames_calls[0]["storenames_cache"] == cache


def test_update_values_passes_empty_cache_when_no_json_file_exists(storenames_closures, tmp_path):
    selector, _ = storenames_closures
    selector._cross_selector_value = ["Dv1A"]
    selector._take_widgets = [[], []]
    # Ensure no cache file exists
    cache_file = tmp_path / ".storesList.json"
    cache_file.unlink(missing_ok=True)

    selector.callbacks["update_options"](types.SimpleNamespace())

    assert selector.configure_storenames_calls[0]["storenames_cache"] == {}


# ---------------------------------------------------------------------------
# save_button
# ---------------------------------------------------------------------------


def test_save_button_writes_storeslist_and_updates_path(storenames_closures, tmp_path):
    selector, _ = storenames_closures
    output_dir = str(tmp_path / "my_session_output_1")
    os.mkdir(output_dir)

    selector._literal_input_2 = {
        "storenames": ["Dv1A", "Dv2A"],
        "names_for_storenames": ["control_DMS", "signal_DMS"],
    }
    selector._select_location_value = output_dir

    selector.callbacks["save"](None)

    assert selector.alert_message == "#### No alerts !!"
    assert selector.path_value == os.path.join(output_dir, "storesList.csv")
    assert os.path.exists(os.path.join(output_dir, "storesList.csv"))


def test_save_button_sets_alert_on_mismatched_lengths(storenames_closures, tmp_path):
    selector, _ = storenames_closures
    output_dir = str(tmp_path / "my_session_output_1")
    os.mkdir(output_dir)

    selector._literal_input_2 = {
        "storenames": ["Dv1A"],
        "names_for_storenames": ["control_DMS", "signal_DMS"],  # length mismatch
    }
    selector._select_location_value = output_dir

    selector.callbacks["save"](None)

    assert "Alert" in selector.alert_message

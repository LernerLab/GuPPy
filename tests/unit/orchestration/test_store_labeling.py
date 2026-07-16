import json
import os
import types

import numpy as np
import panel as pn
import pytest

from guppy.extractors import NpmRecordingExtractor
from guppy.orchestration.store_labeling import (
    _compute_npm_channel_previews,
    _fetchValues,
    _save,
    build_store_labeling_template,
    make_dir,
    show_dir,
)
from guppy_test_data import STUBBED_TESTING_DATA

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_widget(value):
    """Return a lightweight mock Panel widget with only a .value attribute."""
    return types.SimpleNamespace(value=value)


class FakePath:
    """Replaces pathlib.Path in store_ids so cache writes go to tmp_path."""

    _home = None

    @classmethod
    def home(cls):
        return cls._home


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Redirect Path.home() to tmp_path so _save never touches ~/.storesList.json."""
    FakePath._home = tmp_path
    monkeypatch.setattr("guppy.orchestration.store_labeling.Path", FakePath)
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


def test_make_dir_with_explicit_run_name_creates_named_directory(tmp_path):
    session = tmp_path / "session1"
    session.mkdir()

    result = make_dir(str(session), run_name="baseline")

    assert result == str(session / "session1_output_baseline")
    assert os.path.isdir(result)


def test_make_dir_create_policy_raises_on_existing_directory(tmp_path):
    session = tmp_path / "session1"
    session.mkdir()
    (session / "session1_output_baseline").mkdir()

    with pytest.raises(ValueError, match="already exists"):
        make_dir(str(session), run_name="baseline", run_name_policy="create")


def test_make_dir_overwrite_policy_replaces_existing_directory(tmp_path):
    session = tmp_path / "session1"
    session.mkdir()
    existing = session / "session1_output_baseline"
    existing.mkdir()
    (existing / "stale.txt").write_text("stale")

    result = make_dir(str(session), run_name="baseline", run_name_policy="overwrite")

    assert result == str(existing)
    assert os.path.isdir(result)
    assert not (existing / "stale.txt").exists()


def test_make_dir_invalid_policy_raises():
    with pytest.raises(ValueError, match="run_name_policy"):
        make_dir("/anywhere", run_name="x", run_name_policy="bogus")


def test_show_dir_with_explicit_run_name_returns_named_path(tmp_path):
    session = tmp_path / "session1"
    session.mkdir()

    result = show_dir(str(session), run_name="strict")

    assert result == str(session / "session1_output_strict")
    assert not os.path.exists(result)


def test_show_dir_invalid_run_name_raises(tmp_path):
    session = tmp_path / "session1"
    session.mkdir()

    with pytest.raises(ValueError, match="forbidden character"):
        show_dir(str(session), run_name="bad/name")


# ---------------------------------------------------------------------------
# _save
# ---------------------------------------------------------------------------


def test_save_writes_storeslist_csv(isolated_cache):
    select_location = str(isolated_cache / "session1_output_1")
    store_labeling_config = {
        "store_ids": ["Dv1A", "Dv2A", "PulA"],
        "store_labels": ["control_DMS", "signal_DMS", "event1"],
    }

    result = _save(store_labeling_config, select_location)

    assert result == "#### No alerts !!"
    assert os.path.exists(os.path.join(select_location, "storesList.csv"))


def test_save_csv_content_matches_input(isolated_cache):
    select_location = str(isolated_cache / "session1_output_1")
    store_labeling_config = {
        "store_ids": ["Dv1A", "Dv2A", "PulA"],
        "store_labels": ["control_DMS", "signal_DMS", "event1"],
    }

    _save(store_labeling_config, select_location)

    loaded = np.loadtxt(os.path.join(select_location, "storesList.csv"), delimiter=",", dtype=str)
    np.testing.assert_array_equal(loaded[0], ["Dv1A", "Dv2A", "PulA"])
    np.testing.assert_array_equal(loaded[1], ["control_DMS", "signal_DMS", "event1"])


def test_save_returns_alert_when_shapes_mismatch(isolated_cache):
    select_location = str(isolated_cache / "session1_output_1")
    store_labeling_config = {
        "store_ids": ["Dv1A", "Dv2A"],
        "store_labels": ["control_DMS"],  # length mismatch
    }

    result = _save(store_labeling_config, select_location)

    assert "Alert" in result
    # Both lengths should be reported in the alert
    assert "(2)" in result
    assert "(1)" in result


def test_save_returns_alert_when_empty_string_in_names(isolated_cache):
    select_location = str(isolated_cache / "session1_output_1")
    store_labeling_config = {
        "store_ids": ["Dv1A", "Dv2A"],
        "store_labels": ["control_DMS", ""],  # empty string at index 1
    }

    result = _save(store_labeling_config, select_location)

    assert "Alert" in result
    # Alert should name the offending index and store_id
    assert "index 1" in result
    assert "Dv2A" in result


def test_save_returns_alert_listing_multiple_empty_indices(isolated_cache):
    select_location = str(isolated_cache / "session1_output_1")
    store_labeling_config = {
        "store_ids": ["Dv1A", "Dv2A", "Dv3A"],
        "store_labels": ["", "control_DMS", ""],
    }

    result = _save(store_labeling_config, select_location)

    assert "Alert" in result
    # Multiple indices listed
    assert "[0, 2]" in result
    assert "Dv1A" in result
    assert "Dv3A" in result


def test_save_updates_cache_file(isolated_cache):
    select_location = str(isolated_cache / "session1_output_1")
    store_labeling_config = {
        "store_ids": ["Dv1A", "Dv2A"],
        "store_labels": ["control_DMS", "signal_DMS"],
    }

    _save(store_labeling_config, select_location)

    cache_path = isolated_cache / ".storesList.json"
    assert cache_path.exists()
    with open(cache_path) as file:
        cache = json.load(file)
    assert "Dv1A" in cache
    assert "control_DMS" in cache["Dv1A"]
    assert "Dv2A" in cache
    assert "signal_DMS" in cache["Dv2A"]


# ---------------------------------------------------------------------------
# _save — overwrite mode clears output directory
# ---------------------------------------------------------------------------


def test_save_overwrites_clears_all_files_in_existing_dir(isolated_cache):
    """When select_location already exists, all files are deleted before saving."""
    select_location = isolated_cache / "session1_output_1"
    select_location.mkdir()

    # Representative derived data files written by earlier pipeline steps.
    stale_files = ["signal_DMS.hdf5", "control_DMS.hdf5", "z_score_DMS.hdf5", "storesList.csv", "old_data.h5"]
    for filename in stale_files:
        (select_location / filename).write_bytes(b"stale")

    store_labeling_config = {
        "store_ids": ["Dv3A", "Dv4A"],
        "store_labels": ["signal_NAc", "control_NAc"],
    }

    result = _save(store_labeling_config, str(select_location))

    assert result == "#### No alerts !!"
    # Only the freshly written storesList.csv should remain.
    remaining = set(os.listdir(str(select_location)))
    assert remaining == {"storesList.csv"}, f"Expected only storesList.csv, found: {remaining}"


def test_save_overwrites_removes_subdirectories(isolated_cache):
    """When select_location already exists, subdirectories are also removed."""
    select_location = isolated_cache / "session1_output_1"
    select_location.mkdir()

    subdir = select_location / "cross_correlation_output"
    subdir.mkdir()
    (subdir / "corr_event1.hdf5").write_bytes(b"stale")

    store_labeling_config = {
        "store_ids": ["Dv3A", "Dv4A"],
        "store_labels": ["signal_NAc", "control_NAc"],
    }

    _save(store_labeling_config, str(select_location))

    assert not subdir.exists(), "Subdirectory should have been removed on overwrite"


def test_save_new_dir_creates_directory(isolated_cache):
    """When select_location does not exist, _save creates it (new mode, not overwrite)."""
    select_location = isolated_cache / "session1_output_1"
    assert not select_location.exists()

    store_labeling_config = {
        "store_ids": ["Dv1A"],
        "store_labels": ["signal_DMS"],
    }

    result = _save(store_labeling_config, str(select_location))

    assert result == "#### No alerts !!"
    assert select_location.is_dir()


# ---------------------------------------------------------------------------
# _fetchValues
# ---------------------------------------------------------------------------


def _build_fetchValues_args(dropdown_values, textbox_values=None, control_refs=None, text_value=None):
    """Build the store_id_dropdowns, store_id_textboxes, store_id_control_refs, and text widget.

    ``control_refs`` maps a control store's key → the signal store's key it pairs with.
    """
    store_ids = list(dropdown_values.keys())
    textbox_values = textbox_values or {}
    control_refs = control_refs or {}
    store_id_dropdowns = {key: make_widget(dropdown_values[key]) for key in store_ids}
    store_id_textboxes = {key: make_widget(textbox_values.get(key, "")) for key in store_ids}
    store_id_control_refs = {key: make_widget(control_refs.get(key, "")) for key in store_ids}
    text = make_widget(text_value if text_value is not None else store_ids)
    return text, store_ids, store_id_dropdowns, store_id_textboxes, store_id_control_refs


def _fetch_isosbestic(*args, **kwargs):
    """Call _fetchValues with isosbestic_control=True (the configuration where pair checks apply)."""
    kwargs.setdefault("isosbestic_control", True)
    return _fetchValues(*args, **kwargs)


def test_fetchValues_returns_alert_when_store_ids_empty():
    text = make_widget([])
    result = _fetchValues(text, [], {}, {}, {}, {})
    assert "Alert" in result


def test_fetchValues_returns_alert_when_whitespace_in_signal_name():
    text, store_ids, dropdowns, textboxes, control_refs = _build_fetchValues_args(
        dropdown_values={"Dv2A": "signal"},
        textbox_values={"Dv2A": "DMS region"},  # space in name
    )
    result = _fetchValues(text, store_ids, dropdowns, textboxes, control_refs, {})
    assert "Alert" in result


def test_fetchValues_returns_alert_when_required_textbox_empty():
    text, store_ids, dropdowns, textboxes, control_refs = _build_fetchValues_args(
        dropdown_values={"Dv2A": "signal"},
        textbox_values={"Dv2A": ""},  # empty name for required role
    )
    result = _fetchValues(text, store_ids, dropdowns, textboxes, control_refs, {})
    assert "Alert" in result


def test_fetchValues_allows_underscore_in_recording_site_name():
    """Underscores in a recording-site name are now allowed (issue #383)."""
    text, store_ids, dropdowns, textboxes, control_refs = _build_fetchValues_args(
        dropdown_values={"Dv2A": "signal"},
        textbox_values={"Dv2A": "left_hemisphere"},
        text_value=["Dv2A"],
    )
    result_dict = {}
    result = _fetchValues(text, store_ids, dropdowns, textboxes, control_refs, result_dict)
    assert result == "#### No alerts !!"
    assert result_dict["store_labels"] == ["signal_left_hemisphere"]


def test_fetchValues_returns_alert_when_control_has_no_signal_assigned():
    text, store_ids, dropdowns, textboxes, control_refs = _build_fetchValues_args(
        dropdown_values={"Dv1A": "control"},
        control_refs={"Dv1A": ""},  # no signal selected
    )
    result = _fetchValues(text, store_ids, dropdowns, textboxes, control_refs, {})
    assert "Alert" in result
    assert "control" in result.lower()


def test_fetchValues_control_inherits_referenced_signal_name():
    text, store_ids, dropdowns, textboxes, control_refs = _build_fetchValues_args(
        dropdown_values={"Dv2A": "signal", "Dv1A": "control"},
        textbox_values={"Dv2A": "DMS"},
        control_refs={"Dv1A": "Dv2A"},
        text_value=["Dv2A", "Dv1A"],
    )
    result_dict = {}
    result = _fetchValues(text, store_ids, dropdowns, textboxes, control_refs, result_dict)
    assert result == "#### No alerts !!"
    assert result_dict["store_labels"] == ["signal_DMS", "control_DMS"]


def test_fetchValues_control_inherits_underscore_signal_name():
    """A control pairs with a signal whose name contains underscores, with no double entry."""
    text, store_ids, dropdowns, textboxes, control_refs = _build_fetchValues_args(
        dropdown_values={"Dv2A": "signal", "Dv1A": "control"},
        textbox_values={"Dv2A": "d_ms"},
        control_refs={"Dv1A": "Dv2A"},
        text_value=["Dv2A", "Dv1A"],
    )
    result_dict = {}
    result = _fetchValues(text, store_ids, dropdowns, textboxes, control_refs, result_dict)
    assert result == "#### No alerts !!"
    assert result_dict["store_labels"] == ["signal_d_ms", "control_d_ms"]


def test_fetchValues_valid_signal_entry_sets_correct_name():
    text, store_ids, dropdowns, textboxes, control_refs = _build_fetchValues_args(
        dropdown_values={"Dv2A": "signal"},
        textbox_values={"Dv2A": "DMS"},
        text_value=["Dv2A"],
    )
    result_dict = {}
    result = _fetchValues(text, store_ids, dropdowns, textboxes, control_refs, result_dict)
    assert result == "#### No alerts !!"
    assert result_dict["store_labels"] == ["signal_DMS"]


def test_fetchValues_valid_event_ttls_uses_textbox_value():
    text, store_ids, dropdowns, textboxes, control_refs = _build_fetchValues_args(
        dropdown_values={"PulA": "event TTLs"},
        textbox_values={"PulA": "lever_press"},
        text_value=["PulA"],
    )
    result_dict = {}
    result = _fetchValues(text, store_ids, dropdowns, textboxes, control_refs, result_dict)
    assert result == "#### No alerts !!"
    assert result_dict["store_labels"] == ["lever_press"]


def test_fetchValues_non_standard_dropdown_uses_dropdown_value():
    """Dropdown values other than control/signal/event TTLs use the dropdown value directly."""
    text, store_ids, dropdowns, textboxes, control_refs = _build_fetchValues_args(
        dropdown_values={"PulA": "exclude"},
        text_value=["PulA"],
    )
    result_dict = {}
    result = _fetchValues(text, store_ids, dropdowns, textboxes, control_refs, result_dict)
    assert result == "#### No alerts !!"
    assert result_dict["store_labels"] == ["exclude"]


def test_fetchValues_returns_alert_when_duplicate_signal_names():
    text, store_ids, dropdowns, textboxes, control_refs = _build_fetchValues_args(
        dropdown_values={"Dv3A": "signal", "Dv4A": "signal"},
        textbox_values={"Dv3A": "DMS", "Dv4A": "DMS"},
        text_value=["Dv3A", "Dv4A"],
    )
    result = _fetchValues(text, store_ids, dropdowns, textboxes, control_refs, {})
    assert "Alert" in result
    assert "Duplicate signal" in result
    assert "DMS" in result


def test_fetchValues_returns_alert_when_two_controls_share_a_signal():
    text, store_ids, dropdowns, textboxes, control_refs = _build_fetchValues_args(
        dropdown_values={"Dv2A": "signal", "Dv1A": "control", "Dv0A": "control"},
        textbox_values={"Dv2A": "DMS"},
        control_refs={"Dv1A": "Dv2A", "Dv0A": "Dv2A"},
        text_value=["Dv2A", "Dv1A", "Dv0A"],
    )
    result = _fetchValues(text, store_ids, dropdowns, textboxes, control_refs, {})
    assert "Alert" in result
    assert "Duplicate" in result
    assert "control_DMS" in result


def test_fetchValues_returns_alert_when_duplicate_event_ttls():
    text, store_ids, dropdowns, textboxes, control_refs = _build_fetchValues_args(
        dropdown_values={"PulA": "event TTLs", "PulB": "event TTLs"},
        textbox_values={"PulA": "lever_press", "PulB": "lever_press"},
        text_value=["PulA", "PulB"],
    )
    result = _fetchValues(text, store_ids, dropdowns, textboxes, control_refs, {})
    assert "Alert" in result
    assert "Duplicate" in result
    assert "lever_press" in result


def test_fetchValues_isosbestic_alert_when_signal_has_no_control():
    text, store_ids, dropdowns, textboxes, control_refs = _build_fetchValues_args(
        dropdown_values={"Dv1A": "control", "Dv2A": "signal", "Dv3A": "signal"},
        textbox_values={"Dv2A": "DMS", "Dv3A": "NAc"},
        control_refs={"Dv1A": "Dv2A"},  # only DMS has a control
        text_value=["Dv1A", "Dv2A", "Dv3A"],
    )
    result = _fetch_isosbestic(text, store_ids, dropdowns, textboxes, control_refs, {})
    assert "Alert" in result
    assert "no control" in result
    assert "NAc" in result


def test_fetchValues_isosbestic_matched_pairs_pass():
    text, store_ids, dropdowns, textboxes, control_refs = _build_fetchValues_args(
        dropdown_values={"Dv1A": "control", "Dv2A": "signal", "Dv3A": "control", "Dv4A": "signal"},
        textbox_values={"Dv2A": "DMS", "Dv4A": "NAc"},
        control_refs={"Dv1A": "Dv2A", "Dv3A": "Dv4A"},
        text_value=["Dv1A", "Dv2A", "Dv3A", "Dv4A"],
    )
    result_dict = {}
    result = _fetch_isosbestic(text, store_ids, dropdowns, textboxes, control_refs, result_dict)
    assert result == "#### No alerts !!"
    assert result_dict["store_labels"] == ["control_DMS", "signal_DMS", "control_NAc", "signal_NAc"]


def test_fetchValues_non_isosbestic_allows_signal_only():
    """When isosbestic_control is False, a signal without a matching control is valid."""
    text, store_ids, dropdowns, textboxes, control_refs = _build_fetchValues_args(
        dropdown_values={"Dv2A": "signal", "Dv3A": "signal"},
        textbox_values={"Dv2A": "DMS", "Dv3A": "NAc"},
        text_value=["Dv2A", "Dv3A"],
    )
    result_dict = {}
    result = _fetchValues(text, store_ids, dropdowns, textboxes, control_refs, result_dict, isosbestic_control=False)
    assert result == "#### No alerts !!"
    assert result_dict["store_labels"] == ["signal_DMS", "signal_NAc"]


def test_fetchValues_populates_store_ids_from_text_value():
    text_value = ["Dv2A", "Dv1A"]
    text, store_ids, dropdowns, textboxes, control_refs = _build_fetchValues_args(
        dropdown_values={"Dv2A": "signal", "Dv1A": "control"},
        textbox_values={"Dv2A": "DMS"},
        control_refs={"Dv1A": "Dv2A"},
        text_value=text_value,
    )
    result_dict = {}
    _fetchValues(text, store_ids, dropdowns, textboxes, control_refs, result_dict)
    assert result_dict["store_ids"] == text_value


# ---------------------------------------------------------------------------
# build_store_labeling_template on-click closures
# ---------------------------------------------------------------------------


class CapturingStoreLabelingSelector:
    """Minimal fake StoreLabelingSelector that captures on-click closures without rendering Panel."""

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
        self.store_ids = []
        self.store_id_dropdowns = {}
        self.store_id_textboxes = {}
        self.store_id_control_refs = {}
        self.configure_store_ids_calls = []

    def set_select_location_options(self, options):
        self.select_location_options = options

    def set_alert_message(self, message):
        self.alert_message = message

    def get_literal_input_2(self):
        return self._literal_input_2

    def set_literal_input_2(self, store_labeling_config):
        self._literal_input_2 = store_labeling_config

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

    def attach_run_name_watcher(self, callback):
        self.run_name_callback = callback

    def get_run_name(self):
        return getattr(self, "_run_name_value", "")

    def get_overwrite_mode(self):
        return getattr(self, "_overwrite_mode_value", "create_new_file")

    def configure_store_ids(self, store_id_to_store_labels):
        self.configure_store_ids_calls.append(
            {"store_ids": list(self.store_ids), "store_id_to_store_labels": dict(store_id_to_store_labels)}
        )


class FakeStoreLabelingInstructions:
    def __init__(self, folder_path=None):
        self.widget = types.SimpleNamespace()


class FakeBootstrapTemplate:
    def __init__(self, title=""):
        self.main = types.SimpleNamespace(append=lambda item: None)


@pytest.fixture
def store_labeling_closures(tmp_path, monkeypatch, panel_extension):
    """Build build_store_labeling_template with faked UI; return (selector, folder_path).

    selector.callbacks maps button names to their on-click closure functions.
    """
    FakePath._home = tmp_path
    monkeypatch.setattr("guppy.orchestration.store_labeling.Path", FakePath)

    folder = tmp_path / "my_session"
    folder.mkdir()

    captured_selector = None

    class TrackingSelector(CapturingStoreLabelingSelector):
        def __init__(self, allnames):
            super().__init__(allnames)
            nonlocal captured_selector
            captured_selector = self

    monkeypatch.setattr("guppy.orchestration.store_labeling.StoreLabelingSelector", TrackingSelector)
    monkeypatch.setattr("guppy.orchestration.store_labeling.StoreLabelingInstructions", FakeStoreLabelingInstructions)
    monkeypatch.setattr(pn.template, "BootstrapTemplate", FakeBootstrapTemplate)
    monkeypatch.setattr(pn, "Row", lambda *args, **kwargs: None)

    build_store_labeling_template(["Dv1A", "Dv2A", "PulA"], [], str(folder))

    return captured_selector, str(folder)


# ---------------------------------------------------------------------------
# overwrite_button_actions
# ---------------------------------------------------------------------------


def test_overwrite_button_actions_create_new_file_sets_next_output_dir(store_labeling_closures):
    selector, folder_path = store_labeling_closures
    overwrite_button_actions = selector.callbacks["overwrite_button"]

    overwrite_button_actions(types.SimpleNamespace(new="create_new_file"))

    expected = os.path.join(folder_path, "my_session_output_1")
    assert selector.select_location_options == [expected]


def test_overwrite_button_actions_over_write_file_returns_existing_output_dirs(store_labeling_closures):
    selector, folder_path = store_labeling_closures
    overwrite_button_actions = selector.callbacks["overwrite_button"]

    run_folder = os.path.join(folder_path, "my_session_output_1")
    os.mkdir(run_folder)

    overwrite_button_actions(types.SimpleNamespace(new="over_write_file"))

    assert selector.select_location_options == [run_folder]


# ---------------------------------------------------------------------------
# run_name_input_changed
# ---------------------------------------------------------------------------


def test_run_name_input_changed_no_op_when_not_create_new_file(store_labeling_closures):
    selector, _ = store_labeling_closures
    selector._overwrite_mode_value = "over_write_file"
    selector.select_location_options = "untouched"

    selector.run_name_callback(types.SimpleNamespace(new="myrun"))

    assert selector.select_location_options == "untouched"
    assert selector.alert_message is None


def test_run_name_input_changed_updates_select_location_options(store_labeling_closures):
    selector, folder_path = store_labeling_closures
    selector._overwrite_mode_value = "create_new_file"

    selector.run_name_callback(types.SimpleNamespace(new="myrun"))

    expected = os.path.join(folder_path, "my_session_output_myrun")
    assert selector.select_location_options == [expected]
    assert selector.alert_message == "#### No alerts !!"


def test_run_name_input_changed_empty_string_falls_back_to_numeric(store_labeling_closures):
    selector, folder_path = store_labeling_closures
    selector._overwrite_mode_value = "create_new_file"

    selector.run_name_callback(types.SimpleNamespace(new=""))

    expected = os.path.join(folder_path, "my_session_output_1")
    assert selector.select_location_options == [expected]


def test_run_name_input_changed_invalid_run_name_sets_alert(store_labeling_closures):
    selector, _ = store_labeling_closures
    selector._overwrite_mode_value = "create_new_file"

    selector.run_name_callback(types.SimpleNamespace(new="bad/name"))

    assert "Alert" in selector.alert_message
    # When show_dir raises, select_location_options is not updated.
    assert selector.select_location_options is None


# ---------------------------------------------------------------------------
# fetchValues (show_config_button)
# ---------------------------------------------------------------------------


def test_fetch_values_sets_alert_message_when_no_store_ids_configured(store_labeling_closures):
    selector, _ = store_labeling_closures
    selector.store_ids = []

    selector.callbacks["show_config_button"](types.SimpleNamespace())

    assert "Alert" in selector.alert_message


def test_fetch_values_always_calls_set_literal_input_2(store_labeling_closures):
    selector, _ = store_labeling_closures
    selector.store_ids = []

    selector.callbacks["show_config_button"](types.SimpleNamespace())

    # set_literal_input_2 must be called unconditionally (with an empty dict on alert path)
    assert selector._literal_input_2 == {}


def test_fetch_values_delegates_to_fetch_values_function(store_labeling_closures, monkeypatch):
    """fetchValues passes the selector's store_ids to _fetchValues and relays results to the selector."""
    selector, _ = store_labeling_closures
    selector.store_ids = ["Dv1A"]

    captured_args = {}

    def fake_fetch_values(text, store_ids, store_id_dropdowns, store_id_textboxes, store_labeling_config, **kwargs):
        captured_args["store_ids"] = list(store_ids)
        store_labeling_config["store_ids"] = store_ids
        store_labeling_config["store_labels"] = ["control_DMS"]
        return "#### No alerts !!"

    monkeypatch.setattr("guppy.orchestration.store_labeling._fetchValues", fake_fetch_values)

    selector.callbacks["show_config_button"](types.SimpleNamespace())

    assert captured_args["store_ids"] == ["Dv1A"]
    assert selector.alert_message == "#### No alerts !!"
    assert selector._literal_input_2["store_labels"] == ["control_DMS"]


# ---------------------------------------------------------------------------
# update_values (update_options button)
# ---------------------------------------------------------------------------


def test_update_values_sets_store_ids_from_cross_selector(store_labeling_closures):
    selector, _ = store_labeling_closures
    selector._cross_selector_value = ["Dv1A", "Dv2A"]
    selector._take_widgets = [[], []]  # no repeated store_ids

    selector.callbacks["update_options"](types.SimpleNamespace())

    assert selector.change_widgets_value == ["Dv1A", "Dv2A"]


def test_update_values_includes_repeated_store_ids(store_labeling_closures):
    selector, _ = store_labeling_closures
    selector._cross_selector_value = ["Dv2A"]
    selector._take_widgets = [["Dv1A"], [2]]  # Dv1A repeated twice

    selector.callbacks["update_options"](types.SimpleNamespace())

    assert selector.change_widgets_value == ["Dv2A", "Dv1A", "Dv1A"]


def test_update_values_calls_configure_store_ids_with_selected_store_ids(store_labeling_closures):
    selector, _ = store_labeling_closures
    selector._cross_selector_value = ["Dv1A", "PulA"]
    selector._take_widgets = [[], []]

    selector.callbacks["update_options"](types.SimpleNamespace())

    assert len(selector.configure_store_ids_calls) == 1
    assert selector.configure_store_ids_calls[0]["store_ids"] == ["Dv1A", "PulA"]


def test_update_values_loads_store_ids_cache_when_json_file_exists(store_labeling_closures, tmp_path):
    selector, _ = store_labeling_closures
    selector._cross_selector_value = ["Dv1A"]
    selector._take_widgets = [[], []]

    cache = {"Dv1A": ["control_DMS"]}
    cache_file = tmp_path / ".storesList.json"
    with open(cache_file, "w") as f:
        json.dump(cache, f)

    selector.callbacks["update_options"](types.SimpleNamespace())

    assert selector.configure_store_ids_calls[0]["store_id_to_store_labels"] == cache


def test_update_values_passes_empty_cache_when_no_json_file_exists(store_labeling_closures, tmp_path):
    selector, _ = store_labeling_closures
    selector._cross_selector_value = ["Dv1A"]
    selector._take_widgets = [[], []]
    # Ensure no cache file exists
    cache_file = tmp_path / ".storesList.json"
    cache_file.unlink(missing_ok=True)

    selector.callbacks["update_options"](types.SimpleNamespace())

    assert selector.configure_store_ids_calls[0]["store_id_to_store_labels"] == {}


# ---------------------------------------------------------------------------
# save_button
# ---------------------------------------------------------------------------


def test_save_button_writes_storeslist_and_updates_path(store_labeling_closures, tmp_path):
    selector, _ = store_labeling_closures
    run_folder = str(tmp_path / "my_session_output_1")
    os.mkdir(run_folder)

    selector._literal_input_2 = {
        "store_ids": ["Dv1A", "Dv2A"],
        "store_labels": ["control_DMS", "signal_DMS"],
    }
    selector._select_location_value = run_folder

    selector.callbacks["save"](None)

    assert selector.alert_message == "#### No alerts !!"
    assert selector.path_value == os.path.join(run_folder, "storesList.csv")
    assert os.path.exists(os.path.join(run_folder, "storesList.csv"))


def test_save_button_sets_alert_on_mismatched_lengths(store_labeling_closures, tmp_path):
    selector, _ = store_labeling_closures
    run_folder = str(tmp_path / "my_session_output_1")
    os.mkdir(run_folder)

    selector._literal_input_2 = {
        "store_ids": ["Dv1A"],
        "store_labels": ["control_DMS", "signal_DMS"],  # length mismatch
    }
    selector._select_location_value = run_folder

    selector.callbacks["save"](None)

    assert "Alert" in selector.alert_message


# ---------------------------------------------------------------------------
# _compute_npm_channel_previews
# ---------------------------------------------------------------------------


def test_compute_npm_channel_previews_aligns_ragged_channel_lengths():
    # sampleData_NPM_4 interleaves unevenly: chod has one more sample than chev, so chod
    # borrows chev's (shorter) timestamps. The preview must align x/y to equal length,
    # otherwise hv.Curve raises a DataError in the Step-1 GUI.
    folder_path = os.path.join(STUBBED_TESTING_DATA, "npm", "sampleData_NPM_4")
    input_parameters = {"noChannels": 2}

    # Confirm the ragged scenario is real: at least one channel stream has unequal
    # timestamps/data lengths, which is exactly what the alignment guards against.
    streams = NpmRecordingExtractor(folder_path, num_ch=2).decompose()
    ragged = [
        name
        for name, stream in streams.items()
        if "data" in stream and len(stream["timestamps"]) != len(stream["data"])
    ]
    assert ragged, "Expected at least one ragged chod/chpr channel in sampleData_NPM_4"

    previews = _compute_npm_channel_previews(input_parameters, folder_path)

    assert previews, "Expected chev/chod/chpr previews for an NPM session"
    for name, preview in previews.items():
        assert len(preview["x"]) == len(preview["y"]), f"Unequal x/y lengths for preview {name!r}"

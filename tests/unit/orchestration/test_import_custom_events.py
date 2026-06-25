import os

import pandas as pd
import pytest

from guppy.orchestration.import_custom_events import build_custom_events_template


@pytest.fixture
def template(panel_extension, tmp_path):
    return build_custom_events_template(str(tmp_path))


def _set_row(config, index, name, text):
    while len(config.rows) <= index:
        config.add_event_row()
    config.rows[index][0].value = name
    config.rows[index][1].value = text


class TestBuildCustomEventsTemplate:
    def test_exposes_hooks_and_widgets(self, template):
        assert "save_button" in template._hooks
        assert "config" in template._widgets

    def test_save_writes_csv(self, template, tmp_path):
        config = template._widgets["config"]
        _set_row(config, 0, "movement_onset", "0.5\n1.5\n2.5")
        template._hooks["save_button"](None)

        df = pd.read_csv(os.path.join(str(tmp_path), "movement_onset.csv"))
        assert df["timestamps"].tolist() == [0.5, 1.5, 2.5]
        assert "Saved" in config.alert.object

    def test_empty_rows_write_nothing(self, template, tmp_path):
        config = template._widgets["config"]
        template._hooks["save_button"](None)

        assert os.listdir(str(tmp_path)) == []
        assert "No events to save" in config.alert.object

    def test_unsorted_writes_csv_with_warning(self, template, tmp_path):
        config = template._widgets["config"]
        _set_row(config, 0, "ev", "3.0\n1.0\n2.0")
        template._hooks["save_button"](None)

        df = pd.read_csv(os.path.join(str(tmp_path), "ev.csv"))
        assert df["timestamps"].tolist() == [3.0, 1.0, 2.0]
        assert "Warning" in config.alert.object

    def test_non_numeric_sets_alert_and_writes_no_csv(self, template, tmp_path):
        config = template._widgets["config"]
        _set_row(config, 0, "ev", "0.1\nabc")
        template._hooks["save_button"](None)

        assert os.listdir(str(tmp_path)) == []
        assert "abc" in config.alert.object

    def test_named_event_without_timestamps_errors(self, template, tmp_path):
        config = template._widgets["config"]
        _set_row(config, 0, "ev", "   ")
        template._hooks["save_button"](None)

        assert os.listdir(str(tmp_path)) == []
        assert "no timestamps" in config.alert.object

    def test_collision_warns_then_overwrites(self, template, tmp_path):
        config = template._widgets["config"]
        _set_row(config, 0, "ev", "1.0")
        template._hooks["save_button"](None)

        # Second save of the same name without overwrite: warn, keep original.
        _set_row(config, 0, "ev", "9.0\n8.0")
        template._hooks["save_button"](None)
        assert "already exists" in config.alert.object
        assert pd.read_csv(os.path.join(str(tmp_path), "ev.csv"))["timestamps"].tolist() == [1.0]

        # Enable overwrite and save again: file is replaced.
        config.overwrite.value = True
        template._hooks["save_button"](None)
        assert pd.read_csv(os.path.join(str(tmp_path), "ev.csv"))["timestamps"].tolist() == [9.0, 8.0]

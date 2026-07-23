import holoviews as hv
import numpy as np
import pytest

from guppy.frontend.store_labeling_instructions import (
    StoreLabelingInstructions,
    StoreLabelingInstructionsNPM,
    _validate_timestamp_configuration,
)


class TestValidateTimestampConfiguration:
    def test_returns_none_when_both_fields_populated(self):
        result = _validate_timestamp_configuration(timestamp_column_name="Timestamp", time_unit="seconds")
        assert result is None

    def test_raises_when_timestamp_column_blank(self):
        with pytest.raises(ValueError, match="'Select which timestamps to use'"):
            _validate_timestamp_configuration(timestamp_column_name="", time_unit="seconds")

    def test_raises_when_time_unit_blank(self):
        with pytest.raises(ValueError, match="'Select timestamps unit'"):
            _validate_timestamp_configuration(timestamp_column_name="Timestamp", time_unit="")

    def test_raises_when_both_blank_and_lists_both_fields(self):
        with pytest.raises(ValueError) as exception_info:
            _validate_timestamp_configuration(timestamp_column_name="", time_unit="")
        message = str(exception_info.value)
        assert "'Select which timestamps to use'" in message
        assert "'Select timestamps unit'" in message


class TestStoreLabelingInstructions:
    def test_widget_first_object_contains_basename(self, panel_extension, tmp_path):
        session_dir = tmp_path / "my_session"
        session_dir.mkdir()
        instructions = StoreLabelingInstructions(folder_path=str(session_dir))
        first_object = instructions.widget.objects[0]
        # Panel wraps bare strings in Markdown panes; read the .object attribute for content
        content = first_object.object if hasattr(first_object, "object") else str(first_object)
        assert "my_session" in content

    def test_markdown_contains_select_stores_text(self, panel_extension, tmp_path):
        session_dir = tmp_path / "session"
        session_dir.mkdir()
        instructions = StoreLabelingInstructions(folder_path=str(session_dir))
        assert "Select Stores" in instructions.mark_down.object


class TestStoreLabelingInstructionsNPM:
    @staticmethod
    def _preview():
        """Return an in-memory channel preview ({"x", "y"} arrays)."""
        timestamps = np.linspace(0, 10, 50)
        return {"x": timestamps, "y": np.sin(timestamps)}

    @pytest.fixture
    def one_file_instructions(self, tmp_path, panel_extension):
        return StoreLabelingInstructionsNPM(
            folder_path=str(tmp_path / "npm_session"),
            channel_previews={"chev1": self._preview()},
        )

    @pytest.fixture
    def two_file_instructions(self, tmp_path, panel_extension):
        return StoreLabelingInstructionsNPM(
            folder_path=str(tmp_path / "npm_session"),
            channel_previews={"chev1": self._preview(), "chev2": self._preview()},
        )

    def test_plot_select_options_match_basenames(self, two_file_instructions):
        expected_basenames = sorted(["chev1", "chev2"])
        actual_options = sorted(two_file_instructions.plot_select.options)
        assert actual_options == expected_basenames

    def test_make_plot_returns_hv_curve(self, one_file_instructions):
        plot = one_file_instructions._make_plot("chev1")
        assert isinstance(plot, hv.Curve)

    def test_plot_select_change_updates_plot_pane(self, two_file_instructions):
        original_plot = two_file_instructions.plot_pane.object
        other_key = [
            key for key in two_file_instructions.plot_select.options if key != two_file_instructions.plot_select.value
        ][0]
        two_file_instructions.plot_select.value = other_key
        assert two_file_instructions.plot_pane.object is not original_plot


class TestStoreLabelingInstructionsNPMConfigForm:
    """The on-page NPM configuration form (split-events and timestamp column/unit)."""

    @pytest.fixture
    def config_form(self, tmp_path, panel_extension):
        # File 0 needs both split-events and timestamp-unit input; file 1 needs neither.
        return StoreLabelingInstructionsNPM(
            folder_path=str(tmp_path / "npm_session"),
            channel_previews={},
            multiple_event_ttls=[True, False],
            ts_unit_needs=[True, False],
            col_names_ts=["", "Timestamp", "ComputerTimestamp"],
        )

    def test_confirm_button_created_in_interactive_mode(self, config_form):
        assert config_form.confirm_button is not None

    def test_widgets_only_created_for_files_that_need_them(self, config_form):
        assert set(config_form.split_event_checkboxes.keys()) == {0}
        assert set(config_form.timestamp_column_selects.keys()) == {0}
        assert set(config_form.time_unit_selects.keys()) == {0}

    def test_get_npm_split_events_defaults_false_for_non_multiple(self, config_form):
        # File 0 checkbox unchecked -> False; file 1 has no checkbox -> False.
        assert config_form.get_npm_split_events() == [False, False]

    def test_get_npm_split_events_reflects_checkbox(self, config_form):
        config_form.split_event_checkboxes[0].value = True
        assert config_form.get_npm_split_events() == [True, False]

    def test_get_timestamp_configuration_reflects_selections_and_defaults(self, config_form):
        config_form.timestamp_column_selects[0].value = "Timestamp"
        config_form.time_unit_selects[0].value = "milliseconds"
        ts_units, column_names = config_form.get_timestamp_configuration()
        # File 0 uses the selected values; file 1 (not needed) defaults to seconds/None.
        assert ts_units == ["milliseconds", "seconds"]
        assert column_names == ["Timestamp", None]

    def test_get_timestamp_configuration_raises_when_unset(self, config_form):
        # File 0 needs input but the selectors start empty.
        with pytest.raises(ValueError, match="Select"):
            config_form.get_timestamp_configuration()

    def test_set_channel_previews_populates_plot_after_confirm(self, config_form):
        assert config_form.plot_select is None
        config_form.set_channel_previews(
            channel_previews={"chev1": {"x": np.array([0.0, 1.0]), "y": np.array([2.0, 3.0])}}
        )
        assert config_form.plot_select.options == ["chev1"]
        assert isinstance(config_form._make_plot("chev1"), hv.Curve)

    def test_non_interactive_mode_has_no_confirm_button(self, tmp_path, panel_extension):
        instructions = StoreLabelingInstructionsNPM(
            folder_path=str(tmp_path / "npm_session"),
            channel_previews={"chev1": {"x": np.array([0.0, 1.0]), "y": np.array([2.0, 3.0])}},
        )
        assert instructions.confirm_button is None
        assert instructions.plot_select.options == ["chev1"]

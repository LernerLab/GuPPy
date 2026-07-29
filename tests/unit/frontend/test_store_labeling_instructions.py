import holoviews as hv
import numpy as np
import pytest

from guppy.frontend.store_labeling_instructions import (
    StoreLabelingInstructions,
    StoreLabelingInstructionsNPM,
)


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
        # col_names_ts repeats names (one entry per file) and leads with an empty string,
        # mirroring what NpmRecordingExtractor.needs_ts_unit returns.
        return StoreLabelingInstructionsNPM(
            folder_path=str(tmp_path / "npm_session"),
            channel_previews={},
            multiple_event_ttls=[True, False],
            ts_unit_needs=[True, False],
            col_names_ts=["", "Timestamp", "ComputerTimestamp", "Timestamp", "ComputerTimestamp"],
        )

    def test_confirm_button_created_in_interactive_mode(self, config_form):
        assert config_form.confirm_button is not None

    def test_widgets_only_created_for_files_that_need_them(self, config_form):
        assert set(config_form.split_event_checkboxes.keys()) == {0}
        assert set(config_form.timestamp_column_selects.keys()) == {0}
        assert set(config_form.time_unit_selects.keys()) == {0}

    def test_column_options_are_deduped_with_no_empty_option(self, config_form):
        assert config_form.timestamp_column_selects[0].options == ["Timestamp", "ComputerTimestamp"]

    def test_unit_options_have_no_empty_option(self, config_form):
        assert config_form.time_unit_selects[0].options == ["seconds", "milliseconds", "microseconds"]

    def test_get_npm_split_events_defaults_false_for_non_multiple(self, config_form):
        # File 0 checkbox unchecked -> False; file 1 has no checkbox -> False.
        assert config_form.get_npm_split_events() == [False, False]

    def test_get_npm_split_events_reflects_checkbox(self, config_form):
        config_form.split_event_checkboxes[0].value = True
        assert config_form.get_npm_split_events() == [True, False]

    def test_get_timestamp_configuration_uses_defaults(self, config_form):
        # File 0 selectors default to the first option; file 1 (not needed) defaults to seconds/None.
        ts_units, column_names = config_form.get_timestamp_configuration()
        assert ts_units == ["seconds", "seconds"]
        assert column_names == ["Timestamp", None]

    def test_get_timestamp_configuration_reflects_selections(self, config_form):
        config_form.timestamp_column_selects[0].value = "ComputerTimestamp"
        config_form.time_unit_selects[0].value = "milliseconds"
        ts_units, column_names = config_form.get_timestamp_configuration()
        # File 0 uses the selected values; file 1 (not needed) defaults to seconds/None.
        assert ts_units == ["milliseconds", "seconds"]
        assert column_names == ["ComputerTimestamp", None]

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

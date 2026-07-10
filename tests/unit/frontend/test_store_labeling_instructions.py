import holoviews as hv
import numpy as np
import pytest

from guppy.frontend.store_labeling_instructions import (
    StoreLabelingInstructions,
    StoreLabelingInstructionsNPM,
)


class TestStorenamesInstructions:
    def test_widget_first_object_contains_basename(self, panel_extension, tmp_path):
        session_dir = tmp_path / "my_session"
        session_dir.mkdir()
        instructions = StoreLabelingInstructions(folder_path=str(session_dir))
        first_object = instructions.widget.objects[0]
        # Panel wraps bare strings in Markdown panes; read the .object attribute for content
        content = first_object.object if hasattr(first_object, "object") else str(first_object)
        assert "my_session" in content

    def test_markdown_contains_select_storenames_text(self, panel_extension, tmp_path):
        session_dir = tmp_path / "session"
        session_dir.mkdir()
        instructions = StoreLabelingInstructions(folder_path=str(session_dir))
        assert "Select Stores" in instructions.mark_down.object


class TestStorenamesInstructionsNPM:
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

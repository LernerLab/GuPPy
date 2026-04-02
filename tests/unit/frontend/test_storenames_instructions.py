import os

import holoviews as hv
import numpy as np
import pandas as pd
import pytest

from guppy.frontend.storenames_instructions import (
    StorenamesInstructions,
    StorenamesInstructionsNPM,
)


class TestStorenamesInstructions:
    def test_widget_first_object_contains_basename(self, panel_extension, tmp_path):
        session_dir = tmp_path / "my_session"
        session_dir.mkdir()
        instructions = StorenamesInstructions(folder_path=str(session_dir))
        first_object = instructions.widget.objects[0]
        # Panel wraps bare strings in Markdown panes; read the .object attribute for content
        content = first_object.object if hasattr(first_object, "object") else str(first_object)
        assert "my_session" in content

    def test_markdown_contains_select_storenames_text(self, panel_extension, tmp_path):
        session_dir = tmp_path / "session"
        session_dir.mkdir()
        instructions = StorenamesInstructions(folder_path=str(session_dir))
        assert "Select Storenames" in instructions.mark_down.object


class TestStorenamesInstructionsNPM:
    def _write_stub_chev_csv(self, directory, filename):
        """Write a minimal CSV with timestamps and data columns to directory."""
        timestamps = np.linspace(0, 10, 50)
        data = np.sin(timestamps)
        df = pd.DataFrame({"timestamps": timestamps, "data": data})
        file_path = os.path.join(str(directory), filename)
        df.to_csv(file_path, index=False)
        return file_path

    @pytest.fixture
    def one_file_instructions(self, tmp_path, panel_extension):
        session_dir = tmp_path / "npm_session"
        session_dir.mkdir()
        self._write_stub_chev_csv(session_dir, "chev1.csv")
        return StorenamesInstructionsNPM(folder_path=str(session_dir))

    @pytest.fixture
    def two_file_instructions(self, tmp_path, panel_extension):
        session_dir = tmp_path / "npm_session"
        session_dir.mkdir()
        self._write_stub_chev_csv(session_dir, "chev1.csv")
        self._write_stub_chev_csv(session_dir, "chev2.csv")
        return StorenamesInstructionsNPM(folder_path=str(session_dir))

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

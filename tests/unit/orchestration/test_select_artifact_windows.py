import panel as pn
import pytest

from guppy.orchestration import select_artifact_windows
from guppy.utils import page_links
from guppy.utils.page_links import Page


@pytest.fixture
def preprocessed_run_folder(tmp_path):
    """A run folder holding the Step-3 outputs the marking page reads."""
    run_folder = tmp_path / "session_output_1"
    run_folder.mkdir()
    (run_folder / "cntrl_sig_fit_dms.hdf5").touch()
    return str(run_folder)


class TestOrchestrateSelectArtifactWindows:
    def test_offers_the_page_for_the_selected_sessions(self, panel_extension, monkeypatch, preprocessed_run_folder):
        monkeypatch.setattr(
            select_artifact_windows, "resolve_run_folders", lambda session_folders, params: [preprocessed_run_folder]
        )
        window_page = pn.pane.Markdown("windows")
        calls = []

        def build_artifact_window_page(*, run_folders):
            calls.append(run_folders)
            return window_page

        monkeypatch.setattr(select_artifact_windows, "build_artifact_window_page", build_artifact_window_page)

        with page_links.collect() as pages:
            select_artifact_windows.orchestrate_select_artifact_windows(
                {"session_folders": ["/a"], "combine_data": False}
            )

        assert calls == [[preprocessed_run_folder]]
        assert pages == [Page(name="Select Artifact Windows", content=window_page)]

    def test_raises_without_offering_when_preprocessing_has_not_run(self, monkeypatch, tmp_path):
        empty_run_folder = tmp_path / "session_output_1"
        empty_run_folder.mkdir()
        monkeypatch.setattr(
            select_artifact_windows, "resolve_run_folders", lambda session_folders, params: [str(empty_run_folder)]
        )

        with page_links.collect() as pages:
            with pytest.raises(ValueError, match="No preprocessing outputs found"):
                select_artifact_windows.orchestrate_select_artifact_windows(
                    {"session_folders": ["/a"], "combine_data": False}
                )

        assert pages == []

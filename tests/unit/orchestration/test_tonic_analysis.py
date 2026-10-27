import panel as pn
import pytest

from guppy.orchestration import tonic_analysis
from guppy.utils import page_links
from guppy.utils.page_links import Page


@pytest.fixture
def preprocessed_run_folder(tmp_path):
    """A run folder holding the Step-3 outputs the epoch page reads."""
    run_folder = tmp_path / "session_output_1"
    run_folder.mkdir()
    (run_folder / "cntrl_sig_fit_dms.hdf5").touch()
    return str(run_folder)


class TestOrchestrateTonicAnalysis:
    def test_offers_the_page_for_the_selected_sessions(self, panel_extension, monkeypatch, preprocessed_run_folder):
        monkeypatch.setattr(
            tonic_analysis, "resolve_run_folders", lambda session_folders, params: [preprocessed_run_folder]
        )
        epoch_page = pn.pane.Markdown("epochs")
        calls = []

        def build_tonic_epoch_page(*, run_folders):
            calls.append(run_folders)
            return epoch_page

        monkeypatch.setattr(tonic_analysis, "build_tonic_epoch_page", build_tonic_epoch_page)

        with page_links.collect() as pages:
            tonic_analysis.orchestrate_tonic_analysis({"session_folders": ["/a"], "combine_data": False})

        assert calls == [[preprocessed_run_folder]]
        assert pages == [Page(name="Tonic Analysis", content=epoch_page)]

    def test_raises_without_offering_when_preprocessing_has_not_run(self, monkeypatch, tmp_path):
        empty_run_folder = tmp_path / "session_output_1"
        empty_run_folder.mkdir()
        monkeypatch.setattr(
            tonic_analysis, "resolve_run_folders", lambda session_folders, params: [str(empty_run_folder)]
        )

        with page_links.collect() as pages:
            with pytest.raises(ValueError, match="Run Step 3 \\(Preprocess\\) before running tonic analysis"):
                tonic_analysis.orchestrate_tonic_analysis({"session_folders": ["/a"], "combine_data": False})

        assert pages == []

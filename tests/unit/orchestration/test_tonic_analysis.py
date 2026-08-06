import pytest

from guppy.orchestration import tonic_analysis


@pytest.fixture
def preprocessed_run_folder(tmp_path):
    """A run folder holding the Step-3 outputs the epoch page reads."""
    run_folder = tmp_path / "session_output_1"
    run_folder.mkdir()
    (run_folder / "cntrl_sig_fit_dms.hdf5").touch()
    return str(run_folder)


def test_build_page_resolves_folders_and_composes(monkeypatch, preprocessed_run_folder):
    monkeypatch.setattr(
        tonic_analysis, "resolve_run_folders", lambda session_folders, params: [preprocessed_run_folder]
    )
    calls = []
    monkeypatch.setattr(tonic_analysis, "build_tonic_epoch_page", lambda *, run_folders: calls.append(run_folders))

    tonic_analysis._build_page(["/a"], {"combine_data": False})

    assert calls == [[preprocessed_run_folder]]


class TestOrchestrateDefineTonicEpochs:
    def test_opens_the_view_for_the_selected_sessions(self, monkeypatch, preprocessed_run_folder):
        monkeypatch.setattr(
            tonic_analysis, "resolve_run_folders", lambda session_folders, params: [preprocessed_run_folder]
        )
        opened = []
        monkeypatch.setattr(
            tonic_analysis,
            "open_tonic_analysis_view",
            lambda session_folders, params: opened.append(session_folders),
        )
        input_parameters = {"session_folders": ["/a"], "combine_data": False}

        tonic_analysis.orchestrate_tonic_analysis(input_parameters)

        assert opened == [["/a"]]

    def test_raises_without_opening_when_preprocessing_has_not_run(self, monkeypatch, tmp_path):
        empty_run_folder = tmp_path / "session_output_1"
        empty_run_folder.mkdir()
        monkeypatch.setattr(
            tonic_analysis, "resolve_run_folders", lambda session_folders, params: [str(empty_run_folder)]
        )
        opened = []
        monkeypatch.setattr(
            tonic_analysis,
            "open_tonic_analysis_view",
            lambda session_folders, params: opened.append(session_folders),
        )

        with pytest.raises(ValueError, match="Run Step 3 \\(Preprocess\\) before running tonic analysis"):
            tonic_analysis.orchestrate_tonic_analysis({"session_folders": ["/a"], "combine_data": False})

        assert opened == []


def test_exposes_route_factory_and_open_helper():
    assert callable(tonic_analysis.build_tonic_analysis_view)
    assert callable(tonic_analysis.open_tonic_analysis_view)

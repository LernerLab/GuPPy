import pytest

from guppy.orchestration import define_tonic_epochs


@pytest.fixture
def preprocessed_run_folder(tmp_path):
    """A run folder holding the Step-3 outputs the epoch page reads."""
    run_folder = tmp_path / "session_output_1"
    run_folder.mkdir()
    (run_folder / "cntrl_sig_fit_dms.hdf5").touch()
    return str(run_folder)


def test_build_page_resolves_folders_and_composes(monkeypatch, preprocessed_run_folder):
    monkeypatch.setattr(
        define_tonic_epochs, "resolve_run_folders", lambda session_folders, params: [preprocessed_run_folder]
    )
    calls = []
    monkeypatch.setattr(define_tonic_epochs, "build_tonic_epoch_page", lambda *, run_folders: calls.append(run_folders))

    define_tonic_epochs._build_page(["/a"], {"combine_data": False})

    assert calls == [[preprocessed_run_folder]]


class TestOrchestrateDefineTonicEpochs:
    def test_opens_the_view_for_the_selected_sessions(self, monkeypatch, preprocessed_run_folder):
        monkeypatch.setattr(
            define_tonic_epochs, "resolve_run_folders", lambda session_folders, params: [preprocessed_run_folder]
        )
        opened = []
        monkeypatch.setattr(
            define_tonic_epochs,
            "open_define_tonic_epochs_view",
            lambda session_folders, params: opened.append(session_folders),
        )
        input_parameters = {"session_folders": ["/a"], "combine_data": False}

        define_tonic_epochs.orchestrate_define_tonic_epochs(input_parameters)

        assert opened == [["/a"]]

    def test_raises_without_opening_when_preprocessing_has_not_run(self, monkeypatch, tmp_path):
        empty_run_folder = tmp_path / "session_output_1"
        empty_run_folder.mkdir()
        monkeypatch.setattr(
            define_tonic_epochs, "resolve_run_folders", lambda session_folders, params: [str(empty_run_folder)]
        )
        opened = []
        monkeypatch.setattr(
            define_tonic_epochs,
            "open_define_tonic_epochs_view",
            lambda session_folders, params: opened.append(session_folders),
        )

        with pytest.raises(ValueError, match="Run Step 3 \\(Preprocess\\) before defining tonic epochs"):
            define_tonic_epochs.orchestrate_define_tonic_epochs({"session_folders": ["/a"], "combine_data": False})

        assert opened == []


def test_exposes_route_factory_and_open_helper():
    assert callable(define_tonic_epochs.build_define_tonic_epochs_view)
    assert callable(define_tonic_epochs.open_define_tonic_epochs_view)

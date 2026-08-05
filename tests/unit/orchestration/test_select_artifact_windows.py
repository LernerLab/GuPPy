import pytest

from guppy.orchestration import select_artifact_windows


@pytest.fixture
def preprocessed_run_folder(tmp_path):
    """A run folder holding the Step-3 outputs the marking page reads."""
    run_folder = tmp_path / "session_output_1"
    run_folder.mkdir()
    (run_folder / "cntrl_sig_fit_dms.hdf5").touch()
    return str(run_folder)


def test_build_page_resolves_folders_and_composes(monkeypatch, preprocessed_run_folder):
    monkeypatch.setattr(
        select_artifact_windows, "resolve_run_folders", lambda session_folders, params: [preprocessed_run_folder]
    )
    calls = []
    monkeypatch.setattr(
        select_artifact_windows, "build_artifact_window_page", lambda *, run_folders: calls.append(run_folders)
    )

    select_artifact_windows._build_page(["/a"], {"combine_data": False})

    assert calls == [[preprocessed_run_folder]]


class TestOrchestrateSelectArtifactWindows:
    def test_opens_the_view_for_the_selected_sessions(self, monkeypatch, preprocessed_run_folder):
        monkeypatch.setattr(
            select_artifact_windows, "resolve_run_folders", lambda session_folders, params: [preprocessed_run_folder]
        )
        opened = []
        monkeypatch.setattr(
            select_artifact_windows,
            "open_select_artifact_windows_view",
            lambda session_folders, params: opened.append(session_folders),
        )
        input_parameters = {"session_folders": ["/a"], "combine_data": False}

        select_artifact_windows.orchestrate_select_artifact_windows(input_parameters)

        assert opened == [["/a"]]

    def test_raises_without_opening_when_preprocessing_has_not_run(self, monkeypatch, tmp_path):
        empty_run_folder = tmp_path / "session_output_1"
        empty_run_folder.mkdir()
        monkeypatch.setattr(
            select_artifact_windows, "resolve_run_folders", lambda session_folders, params: [str(empty_run_folder)]
        )
        opened = []
        monkeypatch.setattr(
            select_artifact_windows,
            "open_select_artifact_windows_view",
            lambda session_folders, params: opened.append(session_folders),
        )

        with pytest.raises(ValueError, match="No preprocessing outputs found"):
            select_artifact_windows.orchestrate_select_artifact_windows(
                {"session_folders": ["/a"], "combine_data": False}
            )

        assert opened == []


def test_exposes_route_factory_and_open_helper():
    assert callable(select_artifact_windows.build_select_artifact_windows_view)
    assert callable(select_artifact_windows.open_select_artifact_windows_view)

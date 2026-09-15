"""Unit tests for the settings GuPPy remembers between launches."""

import json

import pytest

from guppy.settings import load_settings, remember_root_folders, remembered_root_folders


@pytest.fixture
def roots(tmp_path):
    """An input root folder and an output root folder that both exist on disk."""
    input_root_folder = tmp_path / "raw"
    output_root_folder = tmp_path / "analysis"
    input_root_folder.mkdir()
    output_root_folder.mkdir()
    return input_root_folder, output_root_folder


class TestRememberRootFolders:
    def test_nothing_is_remembered_before_anything_is_stored(self):
        assert load_settings() == {}
        assert remembered_root_folders() == (None, None)

    def test_a_stored_pair_comes_back(self, roots):
        input_root_folder, output_root_folder = roots

        remember_root_folders(input_root_folder=str(input_root_folder), output_root_folder=str(output_root_folder))

        assert remembered_root_folders() == (str(input_root_folder), str(output_root_folder))

    def test_storing_again_replaces_the_pair(self, roots, tmp_path):
        input_root_folder, output_root_folder = roots
        later_input_root = tmp_path / "raw2"
        later_input_root.mkdir()

        remember_root_folders(input_root_folder=str(input_root_folder), output_root_folder=str(output_root_folder))
        remember_root_folders(input_root_folder=str(later_input_root), output_root_folder=str(output_root_folder))

        assert remembered_root_folders() == (str(later_input_root), str(output_root_folder))

    def test_a_folder_that_has_since_been_deleted_is_not_offered(self, roots):
        input_root_folder, output_root_folder = roots
        remember_root_folders(input_root_folder=str(input_root_folder), output_root_folder=str(output_root_folder))

        input_root_folder.rmdir()

        # A stale setting leaves the form empty rather than pointing at nothing.
        assert remembered_root_folders() == (None, str(output_root_folder))

    def test_an_unreadable_settings_file_is_ignored(self, isolated_settings):
        isolated_settings.parent.mkdir(parents=True, exist_ok=True)
        isolated_settings.write_text("{ this is not json")

        # The file is editable by hand, so a damaged one must not stop GuPPy from starting.
        assert load_settings() == {}
        assert remembered_root_folders() == (None, None)

    def test_the_file_is_written_as_readable_json(self, roots, isolated_settings):
        input_root_folder, output_root_folder = roots

        remember_root_folders(input_root_folder=str(input_root_folder), output_root_folder=str(output_root_folder))

        with isolated_settings.open() as settings_file:
            assert json.load(settings_file) == {
                "input_root_folder": str(input_root_folder),
                "output_root_folder": str(output_root_folder),
            }

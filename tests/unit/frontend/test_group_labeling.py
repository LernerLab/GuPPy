import json

import pytest

from guppy.frontend.group_labeling import (
    CREATE_NEW_GROUP,
    EDIT_EXISTING_GROUP,
    GroupLabelingPage,
    save_group_definition,
)
from guppy.utils.utils import GROUP_MEMBERS_FILENAME, read_group_members


@pytest.fixture
def member_run_folder(tmp_path):
    """A usable group member: an ``_output_`` directory holding storesList.csv."""
    session = tmp_path / "sessionA"
    session.mkdir()
    run_folder = session / "sessionA_output_1"
    run_folder.mkdir()
    (run_folder / "storesList.csv").write_text("Dv1A,Dv2A\ncontrol_dms,signal_dms\n")
    return run_folder


@pytest.fixture
def defined_group(tmp_path, member_run_folder):
    """An existing group directory holding only its manifest."""
    group_folder = tmp_path / "saline_group"
    save_group_definition(group_folder=str(group_folder), member_run_folders=[str(member_run_folder)])
    return group_folder


@pytest.fixture
def page(tmp_path, panel_extension):
    return GroupLabelingPage(start_path=str(tmp_path), selected_group_folders=[])


class TestSaveGroupDefinition:
    def test_writes_only_the_manifest(self, tmp_path, member_run_folder):
        group_folder = tmp_path / "saline_group"

        save_group_definition(group_folder=str(group_folder), member_run_folders=[str(member_run_folder)])

        assert [entry.name for entry in group_folder.iterdir()] == [GROUP_MEMBERS_FILENAME]
        with open(group_folder / GROUP_MEMBERS_FILENAME) as manifest:
            assert json.load(manifest) == {"member_run_folders": [str(member_run_folder)]}

    def test_overwrites_the_membership_of_an_existing_group(self, tmp_path, member_run_folder):
        group_folder = tmp_path / "saline_group"
        save_group_definition(group_folder=str(group_folder), member_run_folders=[str(member_run_folder)])

        save_group_definition(group_folder=str(group_folder), member_run_folders=[])

        assert read_group_members(group_folder=str(group_folder)) == []


class TestGroupLabelingPage:
    def test_starts_in_create_mode_with_the_edit_column_hidden(self, page):
        assert page._current_mode == CREATE_NEW_GROUP
        assert page.edit_column.visible is False
        assert page.destination_column.visible is True

    def test_save_writes_the_group_and_reports_its_path(self, tmp_path, page, member_run_folder):
        page.group_name.value = "saline"
        page.destination_selector.value = [str(tmp_path)]
        page.members_selector.value = [str(member_run_folder)]

        page._on_save()

        assert page.alert.visible is False
        assert page.path.value == str(tmp_path / "saline_group")
        assert read_group_members(group_folder=str(tmp_path / "saline_group")) == [str(member_run_folder)]

    def test_save_surfaces_an_invalid_name_on_the_alert(self, tmp_path, page, member_run_folder):
        page.group_name.value = "bad/name"
        page.destination_selector.value = [str(tmp_path)]
        page.members_selector.value = [str(member_run_folder)]

        page._on_save()

        assert page.alert.visible is True
        assert "forbidden character" in page.alert.object

    def test_save_surfaces_a_missing_destination_on_the_alert(self, page, member_run_folder):
        page.group_name.value = "saline"
        page.members_selector.value = [str(member_run_folder)]

        page._on_save()

        assert page.alert.visible is True
        assert "exactly one destination directory" in page.alert.object

    def test_save_surfaces_a_session_folder_picked_as_a_member(self, tmp_path, page):
        session = tmp_path / "sessionA"
        page.group_name.value = "saline"
        page.destination_selector.value = [str(tmp_path)]
        page.members_selector.value = [str(session)]

        page._on_save()

        assert page.alert.visible is True
        assert "must be output directories" in page.alert.object

    def test_a_saved_group_leaves_the_edit_browser_pointing_at_it(self, tmp_path, page, member_run_folder):
        page.group_name.value = "saline"
        page.destination_selector.value = [str(tmp_path)]
        page.members_selector.value = [str(member_run_folder)]

        page._on_save()

        assert page.group_to_edit_selector.directory == str(tmp_path)


class TestGroupLabelingPageEditMode:
    @pytest.fixture
    def editing_page(self, tmp_path, defined_group, panel_extension):
        page = GroupLabelingPage(start_path=str(tmp_path), selected_group_folders=[str(defined_group)])
        page._on_mode_change(type("Event", (), {"new": EDIT_EXISTING_GROUP})())
        return page

    def test_edit_mode_shows_the_group_browser_and_hides_the_destination_column(self, editing_page):
        assert editing_page.edit_column.visible is True
        assert editing_page.destination_column.visible is False

    def test_browsing_to_a_group_loads_its_recorded_members(self, editing_page, defined_group, member_run_folder):
        editing_page.group_to_edit_selector.value = [str(defined_group)]

        assert editing_page.members_selector.value == [str(member_run_folder)]

    def test_loaded_members_are_visible_in_the_browsers_selected_pane(
        self, editing_page, defined_group, member_run_folder
    ):
        """Loading must show in the browser, not just in its value."""
        editing_page.group_to_edit_selector.value = [str(defined_group)]

        assert editing_page.members_selector._selector.value == [str(member_run_folder)]

    def test_a_group_can_be_edited_with_nothing_selected_on_the_homepage(
        self, tmp_path, defined_group, panel_extension
    ):
        """Editing must not depend on a prior homepage selection."""
        page = GroupLabelingPage(start_path=str(tmp_path), selected_group_folders=[])
        page._on_mode_change(type("Event", (), {"new": EDIT_EXISTING_GROUP})())

        page.group_to_edit_selector.value = [str(defined_group)]

        assert page.members_selector.value == read_group_members(group_folder=str(defined_group))

    def test_saving_rewrites_the_membership_in_place(self, editing_page, defined_group, tmp_path):
        other_session = tmp_path / "sessionB"
        other_run = other_session / "sessionB_output_1"
        other_run.mkdir(parents=True)
        (other_run / "storesList.csv").write_text("Dv1A,Dv2A\ncontrol_dms,signal_dms\n")

        editing_page.group_to_edit_selector.value = [str(defined_group)]
        editing_page.members_selector.value = [str(other_run)]
        editing_page._on_save()

        assert editing_page.alert.visible is False
        assert read_group_members(group_folder=str(defined_group)) == [str(other_run)]

    def test_saving_without_choosing_a_group_surfaces_an_alert(self, editing_page, member_run_folder):
        editing_page.members_selector.value = [str(member_run_folder)]

        editing_page._on_save()

        assert editing_page.alert.visible is True
        assert "Select exactly one group directory to edit" in editing_page.alert.object

"""Label Groups page: define a group of output runs to average, without computing anything.

The page writes only ``group_members.json`` into ``<destination>/<name>_group/``, mirroring how
Label Stores writes only ``storesList.csv`` into a new run folder. Averaging into the group is
the separate Group Analysis step, which operates on the group directories selected on the
homepage.
"""

import logging
import os

import panel as pn

from .frontend_utils import default_root_path
from ..utils.utils import (
    GROUP_MEMBERS_FILENAME,
    group_folder_for_group,
    is_group_folder,
    read_group_members,
    validate_group_name,
    write_group_members,
)
from ..utils.validation import (
    validate_group_definitions,
    validate_group_member_run_folders,
)

logger = logging.getLogger(__name__)

CREATE_NEW_GROUP = "create_new_group"
EDIT_EXISTING_GROUP = "edit_existing_group"


class GroupLabelingPage:
    """Panel page for defining one group: its name, destination, and member runs.

    Parameters
    ----------
    start_path : str
        Directory the file browsers open in.
    selected_group_folders : list of str
        Group directories selected on the homepage. Used only to open the edit browser
        near a group already in play; editing never depends on this being non-empty.
    """

    def __init__(self, *, start_path: str, selected_group_folders: list[str]) -> None:
        self.start_path = start_path if start_path and os.path.isdir(start_path) else default_root_path()
        self.selected_group_folders = list(selected_group_folders)
        self.edit_start_path = (
            os.path.dirname(self.selected_group_folders[0]) if self.selected_group_folders else self.start_path
        )
        self.setup_widgets()
        self.attach_callbacks()

    def setup_widgets(self) -> None:
        """Build every widget on the page and store it as an instance attribute."""
        self.mode_markdown = pn.pane.Markdown(
            """
### Label Groups

A group collects output runs so their PSTH, peak/AUC, transient and cross-correlation
results can be averaged together. Saving here only records *which* runs belong to the
group — run **Group Analysis** afterwards to compute the averages.

- **create_new_group** — define a new group. Name it and choose where it goes below.
- **edit_existing_group** — change the members of a group selected on the homepage.
            """,
            width=620,
        )
        self.mode_button = pn.widgets.MenuButton(
            name="create a new group or edit an existing one?  ",
            items=[CREATE_NEW_GROUP, EDIT_EXISTING_GROUP],
            button_type="default",
            split=True,
            width=620,
        )
        self._current_mode = CREATE_NEW_GROUP

        self.edit_markdown = pn.pane.Markdown(
            "### 2 · Which group are you editing?\n\n"
            "Browse to the `<name>_group` directory whose members you want to change. Its recorded "
            "members load into the browser on the left, ready to adjust.",
            width=620,
        )
        self.group_to_edit_selector = pn.widgets.FileSelector(
            self.edit_start_path, root_directory="/", name="Group directory to edit", width=620
        )

        self.destination_markdown = pn.pane.Markdown(
            "### 2 · Where does the group go?\n\n"
            "Name the group and pick **one** directory to write it into. The group is created at "
            "`<directory below>/<name>_group/`.",
            width=620,
        )
        self.group_name = pn.widgets.TextInput(name="Group name", value="", placeholder="e.g. saline", width=620)
        self.destination_selector = pn.widgets.FileSelector(
            self.start_path, root_directory="/", name="Destination directory for the new group", width=620
        )

        self.members_markdown = pn.pane.Markdown(
            "### 1 · What goes into the group?\n\n"
            "Pick the **output run directories** to average — the `<session>_output_<run>` folders "
            "*inside* each session, not the session folders themselves. Each must already have "
            "Step-4 results.",
            width=620,
        )
        self.members_selector = pn.widgets.FileSelector(
            self.start_path, root_directory="/", name="Member runs to average", width=620
        )

        self.save = pn.widgets.Button(name="Save group definition", button_type="primary", width=1300)
        self.alert = pn.pane.Alert("", alert_type="danger", visible=False, width=1300)
        self.path = pn.widgets.TextInput(name="Group directory", width=1300, disabled=True)

        self.destination_column = pn.Column(
            self.destination_markdown,
            self.group_name,
            self.destination_selector,
            width=640,
        )
        self.edit_column = pn.Column(
            self.edit_markdown,
            self.group_to_edit_selector,
            width=640,
            visible=False,
        )
        self.members_column = pn.Column(
            self.members_markdown,
            self.members_selector,
            width=640,
        )
        self.widget = pn.Column(
            self.mode_markdown,
            self.mode_button,
            pn.layout.Divider(),
            pn.Row(self.members_column, pn.Spacer(width=20), self.destination_column, self.edit_column),
            pn.layout.Divider(),
            self.save,
            self.alert,
            self.path,
        )

    def attach_callbacks(self) -> None:
        """Wire the page's widget callbacks."""
        self.mode_button.on_click(self._on_mode_change)
        self.group_to_edit_selector.param.watch(self._on_group_to_edit_change, "value")
        self.save.on_click(self._on_save)

    def _on_mode_change(self, event: object) -> None:
        """Show the fields that belong to the chosen mode."""
        self._current_mode = event.new
        creating = self._current_mode == CREATE_NEW_GROUP
        self.destination_column.visible = creating
        self.edit_column.visible = not creating

    def _on_group_to_edit_change(self, event: object) -> None:
        """Load the chosen group's recorded members into the member selector."""
        selected = list(event.new or [])
        if len(selected) != 1 or not is_group_folder(selected[0]):
            return
        if not os.path.exists(os.path.join(selected[0], GROUP_MEMBERS_FILENAME)):
            return
        self._show_members(read_group_members(group_folder=selected[0]))

    def _show_members(self, member_run_folders: list[str]) -> None:
        """Load members into the browser so they appear in its selected pane.

        Setting ``value`` alone leaves the pane empty, because the browser's selected list
        is driven by its own cross-selector rather than the other way round. Navigating to
        the members' directory first puts them among the browser's options, so driving the
        cross-selector both selects them and shows them.
        """
        if not member_run_folders:
            self.members_selector.value = []
            return
        self.members_selector.directory = os.path.dirname(member_run_folders[0])
        self.members_selector._update_files()
        self.members_selector._selector.value = list(member_run_folders)

    def _resolve_group_folder(self) -> str:
        """Return the group directory the save targets, validating the mode's inputs.

        Raises
        ------
        ValueError
            When the group name, destination or edit selection is unusable.
        """
        if self._current_mode == EDIT_EXISTING_GROUP:
            selected = list(self.group_to_edit_selector.value or [])
            if len(selected) != 1:
                raise ValueError(
                    f"Select exactly one group directory to edit; got {len(selected)}. Browse to a "
                    f"'<name>_group' directory, or switch to '{CREATE_NEW_GROUP}' to define a new group."
                )
            validate_group_definitions(group_folders=selected)
            return selected[0]

        validate_group_name(self.group_name.value)
        destinations = self.destination_selector.value or []
        if len(destinations) != 1:
            raise ValueError(
                f"Select exactly one destination directory for the group; got {len(destinations)}. "
                "The group is written to '<destination>/<group_name>_group'."
            )
        return group_folder_for_group(destination_directory=destinations[0], group_name=self.group_name.value)

    def _on_save(self, event: object = None) -> None:
        """Write the group's manifest, surfacing input problems on the page's alert."""
        try:
            group_folder = self._resolve_group_folder()
            member_run_folders = list(self.members_selector.value or [])
            validate_group_member_run_folders(member_run_folders=member_run_folders)
            save_group_definition(group_folder=group_folder, member_run_folders=member_run_folders)
        except ValueError as error:
            self.alert.object = str(error)
            self.alert.visible = True
            return

        self.alert.visible = False
        self.path.value = group_folder
        # Point the edit browser at the group just written, so switching to edit mode lands
        # on it rather than back at the starting directory.
        self.group_to_edit_selector.directory = os.path.dirname(group_folder)
        self.group_to_edit_selector._update_files()

    def build_template(self) -> pn.template.BootstrapTemplate:
        """Return the Panel template hosting this page.

        The page is a fixed width, so it is flanked by stretching spacers to keep it
        centred rather than pinned to the left on a wide monitor.
        """
        template = pn.template.BootstrapTemplate(title="GuPPy — Label Groups")
        template.main.append(pn.Row(pn.HSpacer(), self.widget, pn.HSpacer(), sizing_mode="stretch_width"))
        return template


def save_group_definition(*, group_folder: str, member_run_folders: list[str]) -> None:
    """Create a group directory holding only its membership manifest.

    The directory carries no averaged results until the Group Analysis step runs against
    it, mirroring a run folder that holds ``storesList.csv`` before Step 2.

    Parameters
    ----------
    group_folder : str
        Path of the group output directory.
    member_run_folders : list of str
        Output (run) directories to record as the group's members.
    """
    os.makedirs(group_folder, exist_ok=True)
    write_group_members(group_folder=group_folder, member_run_folders=member_run_folders)
    logger.info(f"Group definition saved at {group_folder} with {len(member_run_folders)} member run(s).")

import json
import shutil
from pathlib import Path

import pytest

from guppy.testing import default_output_root_folder
from guppy.testing.api import label_groups, step1, step2, step3, step4
from guppy_test_data import STUBBED_TESTING_DATA

SESSION_SUBDIRS = [
    "tdt/Photo_048_392-200728-121222",
    "tdt/Photo_63_207-181030-103332",
]
STORE_ID_TO_STORE_LABEL = {
    "Dv1A": "control_dms",
    "Dv2A": "signal_dms",
    "PrtN": "port_entries_dms",
}


@pytest.fixture(scope="module")
def labeled_sessions(tmp_path_factory):
    """Both TDT sessions labeled through the first session's "Save to all selected sessions" button."""
    base_dir = tmp_path_factory.mktemp("input_root_folder")
    selected_folders = []
    for subdir in SESSION_SUBDIRS:
        session_copy = base_dir / Path(subdir).name
        shutil.copytree(STUBBED_TESTING_DATA / subdir, session_copy)
        selected_folders.append(str(session_copy))

    step1(
        base_dir=str(base_dir),
        selected_folders=selected_folders,
        store_id_to_store_label=STORE_ID_TO_STORE_LABEL,
        save_to_all_sessions=True,
    )
    return str(base_dir), selected_folders


def test_save_to_all_sessions_labels_every_session(labeled_sessions):
    base_dir, selected_folders = labeled_sessions
    output_root_folder = Path(default_output_root_folder(base_dir=base_dir))

    for session in selected_folders:
        stores_list = output_root_folder / Path(session).name / "output_1" / "storesList.csv"
        assert stores_list.read_text().splitlines() == [
            "Dv1A,Dv2A,PrtN",
            "control_dms,signal_dms,port_entries_dms",
        ]


def test_label_groups_selects_member_runs_by_run_name(labeled_sessions):
    base_dir, selected_folders = labeled_sessions
    selected_runs = {folder: ["1"] for folder in selected_folders}
    step2(base_dir=base_dir, selected_folders=selected_folders, selected_runs=selected_runs)
    step3(base_dir=base_dir, selected_folders=selected_folders, selected_runs=selected_runs)
    step4(base_dir=base_dir, selected_folders=selected_folders, selected_runs=selected_runs)

    label_groups(
        destination_directory=base_dir,
        group_name="saline",
        base_dir=base_dir,
        selected_folders=selected_folders,
        run_names=["1"],
    )

    output_root_folder = Path(default_output_root_folder(base_dir=base_dir))
    with (Path(base_dir) / "saline_group" / "group_members.json").open() as manifest:
        assert json.load(manifest) == {
            "member_run_folders": [
                str(output_root_folder / "Photo_048_392-200728-121222" / "output_1"),
                str(output_root_folder / "Photo_63_207-181030-103332" / "output_1"),
            ]
        }

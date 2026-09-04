import glob
import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import holoviews as hv
import numpy as np
import pandas as pd
import pytest

from guppy.frontend.visualization_dashboard import VisualizationDashboard
from guppy.testing.api import (
    group_analysis,
    label_groups,
    step1,
    step2,
    step3,
    step4,
    step5,
)
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
EXPECTED_RECORDING_SITE = "dms"
EXPECTED_TTL = "port_entries_dms"

# Two sessions that share the same fiber recording site (dms) but record different behavioral
# events, so each event's group average has a single contributing session (n=1).
DISJOINT_STORE_ID_TO_STORE_LABEL = {
    "tdt/Photo_048_392-200728-121222": {
        "Dv1A": "control_dms",
        "Dv2A": "signal_dms",
        "PrtN": "rewarded_nose_pokes",
    },
    "tdt/Photo_63_207-181030-103332": {
        "Dv1A": "control_dms",
        "Dv2A": "signal_dms",
        "PrtN": "unrewarded_nose_pokes",
    },
}


@pytest.fixture
def copied_sessions(tmp_path):
    """Copy the two sample TDT sessions into a fresh base dir with prior outputs removed.

    Returns
    -------
    tuple[str, list[str]]
        ``(base_dir, selected_folders)`` ready to drive the pipeline API.
    """
    source_sessions = [STUBBED_TESTING_DATA / subdir for subdir in SESSION_SUBDIRS]
    for source_session in source_sessions:
        assert source_session.is_dir(), f"Sample data not available at expected path: {source_session}"

    temporary_base_directory = tmp_path / "data_root"
    temporary_base_directory.mkdir(parents=True, exist_ok=True)

    session_copies = []
    for source_session in source_sessions:
        session_name = source_session.name
        session_copy = temporary_base_directory / session_name
        shutil.copytree(source_session, session_copy)
        for output_directory in glob.glob(os.path.join(session_copy, f"{session_name}_output_*")):
            assert os.path.isdir(output_directory)
            shutil.rmtree(output_directory)
        parameters_path = session_copy / "GuPPyParamtersUsed.json"
        if parameters_path.exists():
            parameters_path.unlink()
        session_copies.append(session_copy)

    return str(temporary_base_directory), [str(session_copy) for session_copy in session_copies]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_group_analysis(copied_sessions):
    """
    Integration test: run the full pipeline (Steps 2-5) on two TDT sessions and then
    perform group-level averaging, asserting that the average directory and expected
    output files are created with the correct structure.
    """
    base_dir, selected_folders = copied_sessions
    temporary_base_directory = Path(base_dir)

    common_kwargs = dict(base_dir=base_dir, selected_folders=selected_folders)
    selected_runs = {folder: ["1"] for folder in selected_folders}

    step1(**common_kwargs, store_id_to_store_label=STORE_ID_TO_STORE_LABEL)
    step2(**common_kwargs, selected_runs=selected_runs)
    step3(**common_kwargs, selected_runs=selected_runs)
    step4(**common_kwargs, selected_runs=selected_runs)

    # Run group averaging pass
    label_groups(
        member_run_folders=[
            os.path.join(folder, f"{os.path.basename(folder)}_output_1") for folder in selected_folders
        ],
        destination_directory=base_dir,
        group_name="saline",
    )
    group_analysis(base_dir=base_dir, selected_group_folders=[os.path.join(base_dir, "saline_group")])

    group_directory = temporary_base_directory / "saline_group"
    assert group_directory.is_dir(), f"No group directory found under {temporary_base_directory}"

    group_psth_file_path = os.path.join(
        group_directory,
        f"{EXPECTED_TTL}_{EXPECTED_RECORDING_SITE}_z_score_{EXPECTED_RECORDING_SITE}.h5",
    )
    assert os.path.exists(group_psth_file_path), f"Missing group PSTH HDF5: {group_psth_file_path}"

    group_psth_dataframe = pd.read_hdf(group_psth_file_path, key="df")
    assert "timestamps" in group_psth_dataframe.columns, f"'timestamps' column missing in {group_psth_file_path}"
    assert "mean" in group_psth_dataframe.columns, f"'mean' column missing in {group_psth_file_path}"

    hv.extension("bokeh")
    captured_dashboards: list[VisualizationDashboard] = []
    original_init = VisualizationDashboard.__init__

    def capturing_init(self, *, plotter, basename):
        original_init(self, plotter=plotter, basename=basename)
        captured_dashboards.append(self)

    with patch.object(VisualizationDashboard, "__init__", capturing_init):
        with patch.object(VisualizationDashboard, "show", lambda self: None):
            step5(
                base_dir=base_dir,
                selected_folders=selected_folders,
                selected_runs=selected_runs,
            )

    assert len(captured_dashboards) >= 1, "step5 created no VisualizationDashboard instances"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_group_analysis_different_event_names_per_session(copied_sessions):
    """Group-average and visualize two sessions that share the same fiber recording site but
    record different behavioral events (one 'rewarded', one 'unrewarded').

    Reproduces issue #368: the sessions have non-identical store_id sets, so this
    exercises the relaxed fiber recording-site validation (averaging is no longer blocked).
    Because each event is present in only one session, its group average has a single
    contributing session (n=1), which also exercises the single-trial heatmap that
    previously blanked the visualization dashboard with a Bokeh stack overflow.
    """
    base_dir, selected_folders = copied_sessions
    temporary_base_directory = Path(base_dir)
    selected_runs = {folder: ["1"] for folder in selected_folders}

    # Step 1 is run per session so each gets a different behavioral-event store_id
    # while sharing the same control/signal (dms) fiber recording site.
    for session_folder, subdir in zip(selected_folders, SESSION_SUBDIRS, strict=True):
        step1(
            base_dir=base_dir,
            selected_folders=[session_folder],
            store_id_to_store_label=DISJOINT_STORE_ID_TO_STORE_LABEL[subdir],
        )

    common_kwargs = dict(base_dir=base_dir, selected_folders=selected_folders)
    step2(**common_kwargs, selected_runs=selected_runs)
    step3(**common_kwargs, selected_runs=selected_runs)
    step4(**common_kwargs, selected_runs=selected_runs)
    member_run_folders = [os.path.join(folder, f"{os.path.basename(folder)}_output_1") for folder in selected_folders]
    label_groups(
        member_run_folders=member_run_folders,
        destination_directory=base_dir,
        group_name="cross_condition",
    )
    group_analysis(base_dir=base_dir, selected_group_folders=[os.path.join(base_dir, "cross_condition_group")])

    # Both events must be averaged even though no session has both -- cross-condition
    # averaging that the pre-#368 validation rejected outright.
    average_directory = temporary_base_directory / "cross_condition_group"
    expected_columns_by_event = {
        "rewarded_nose_pokes": "Photo_048_392-200728-121222_output_1",
        "unrewarded_nose_pokes": "Photo_63_207-181030-103332_output_1",
    }
    for event, contributing_session in expected_columns_by_event.items():
        average_path = average_directory / f"{event}_{EXPECTED_RECORDING_SITE}_z_score_{EXPECTED_RECORDING_SITE}.h5"
        assert average_path.exists(), f"Missing group PSTH for event {event!r}: {average_path}"
        average_dataframe = pd.read_hdf(average_path, key="df")
        # n=1: exactly the one session that recorded this event contributed.
        session_columns = [c for c in average_dataframe.columns if c not in ("timestamps", "mean", "err")]
        assert session_columns == [
            contributing_session
        ], f"Event {event!r} average should aggregate only {contributing_session!r}, got {session_columns}"

    # Average visualization must build, and every single-trial heatmap must render
    # through the datashaded path rather than the old bare single-row QuadMesh that
    # overflowed Bokeh's client-side renderer and blanked the dashboard.
    hv.extension("bokeh")
    captured_dashboards: list[VisualizationDashboard] = []
    original_init = VisualizationDashboard.__init__

    def capturing_init(self, *, plotter, basename):
        original_init(self, plotter=plotter, basename=basename)
        captured_dashboards.append(self)

    with patch.object(VisualizationDashboard, "__init__", capturing_init):
        with patch.object(VisualizationDashboard, "show", lambda self: None):
            step5(
                base_dir=base_dir,
                selected_folders=selected_folders,
                selected_runs=selected_runs,
                selected_group_folders=[str(average_directory)],
            )

    assert len(captured_dashboards) >= 1, "step5 created no VisualizationDashboard instances"
    # Step 5 opens a dashboard per selected session run *and* per selected group, so pick
    # out the group's own dashboard by the directory it was built from.
    group_dashboards = [dashboard for dashboard in captured_dashboards if dashboard.basename == average_directory.name]
    assert len(group_dashboards) == 1, f"Expected exactly one group dashboard, got {len(group_dashboards)}"
    plotter = group_dashboards[0].plotter
    heatmap_events = list(plotter.param.event_selector_heatmap.objects)
    assert len(heatmap_events) == 2, f"Expected both events in the average dashboard, got {heatmap_events}"
    for event in heatmap_events:
        plotter.event_selector_heatmap = event
        image = plotter.heatmap()
        assert image is not None
        assert not isinstance(
            image, hv.QuadMesh
        ), f"Single-trial heatmap for {event!r} used the broken raw-QuadMesh path"
        hv.render(image)  # must not raise (the JS stack overflow reproduced here as a build/render error)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_group_analysis_step_writes_a_named_group_directory(copied_sessions):
    """The Group Analysis step averages selected runs into <destination>/<name>_group.

    Runs Steps 1-4 per session, then the new Group Analysis step, and asserts the group
    directory's name, manifest, provenance snapshot, stores list and averaged PSTH.
    """
    base_dir, selected_folders = copied_sessions
    common_kwargs = dict(base_dir=base_dir, selected_folders=selected_folders)
    selected_runs = {folder: ["1"] for folder in selected_folders}

    step1(**common_kwargs, store_id_to_store_label=STORE_ID_TO_STORE_LABEL)
    step2(**common_kwargs, selected_runs=selected_runs)
    step3(**common_kwargs, selected_runs=selected_runs)
    step4(**common_kwargs, selected_runs=selected_runs)

    member_run_folders = [os.path.join(folder, f"{os.path.basename(folder)}_output_1") for folder in selected_folders]
    label_groups(
        member_run_folders=member_run_folders,
        destination_directory=base_dir,
        group_name="saline",
    )
    group_analysis(base_dir=base_dir, selected_group_folders=[os.path.join(base_dir, "saline_group")])

    group_folder = Path(base_dir) / "saline_group"
    assert group_folder.is_dir(), f"No 'saline_group' directory under {base_dir}"
    # The legacy, location-derived output directory is not written any more.
    assert not (Path(base_dir) / "average").exists()

    with open(group_folder / "group_members.json") as manifest_file:
        assert json.load(manifest_file) == {"member_run_folders": member_run_folders}

    assert (group_folder / "GuPPyParamtersUsed.json").exists()

    stores_list = np.genfromtxt(group_folder / "storesList.csv", dtype="str", delimiter=",").reshape(2, -1)
    assert EXPECTED_TTL in stores_list[1, :].tolist()

    group_psth_path = group_folder / f"{EXPECTED_TTL}_{EXPECTED_RECORDING_SITE}_z_score_{EXPECTED_RECORDING_SITE}.h5"
    group_psth = pd.read_hdf(group_psth_path, key="df")
    # One column per member run, named by the run folder's basename, plus mean/err/timestamps.
    for run_folder in member_run_folders:
        assert os.path.basename(run_folder) in group_psth.columns
    assert list(group_psth.columns[-3:]) == ["timestamps", "mean", "err"]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_group_analysis_step_rebuilds_the_group_when_a_member_is_dropped(copied_sessions):
    """Re-running a group with fewer members must not leave the dropped member behind."""
    base_dir, selected_folders = copied_sessions
    common_kwargs = dict(base_dir=base_dir, selected_folders=selected_folders)
    selected_runs = {folder: ["1"] for folder in selected_folders}

    step1(**common_kwargs, store_id_to_store_label=STORE_ID_TO_STORE_LABEL)
    step2(**common_kwargs, selected_runs=selected_runs)
    step3(**common_kwargs, selected_runs=selected_runs)
    step4(**common_kwargs, selected_runs=selected_runs)

    member_run_folders = [os.path.join(folder, f"{os.path.basename(folder)}_output_1") for folder in selected_folders]
    label_groups(
        member_run_folders=member_run_folders,
        destination_directory=base_dir,
        group_name="saline",
    )
    group_analysis(base_dir=base_dir, selected_group_folders=[os.path.join(base_dir, "saline_group")])
    group_folder = Path(base_dir) / "saline_group"
    psth_path = group_folder / f"{EXPECTED_TTL}_{EXPECTED_RECORDING_SITE}_z_score_{EXPECTED_RECORDING_SITE}.h5"
    assert len(pd.read_hdf(psth_path, key="df").columns) == 5  # 2 members + timestamps/mean/err

    label_groups(
        member_run_folders=member_run_folders[:1],
        destination_directory=base_dir,
        group_name="saline",
    )
    group_analysis(base_dir=base_dir, selected_group_folders=[os.path.join(base_dir, "saline_group")])

    with open(group_folder / "group_members.json") as manifest_file:
        assert json.load(manifest_file) == {"member_run_folders": member_run_folders[:1]}
    remaining = pd.read_hdf(psth_path, key="df")
    assert os.path.basename(member_run_folders[0]) in remaining.columns
    assert os.path.basename(member_run_folders[1]) not in remaining.columns

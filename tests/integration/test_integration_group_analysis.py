import glob
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import holoviews as hv
import pandas as pd
import pytest
from conftest import STUBBED_TESTING_DATA

from guppy.frontend.visualization_dashboard import VisualizationDashboard
from guppy.testing.api import step1, step2, step3, step4, step5

SESSION_SUBDIRS = [
    "tdt/Photo_048_392-200728-121222",
    "tdt/Photo_63_207-181030-103332",
]
STORENAMES_MAP = {
    "Dv1A": "control_dms",
    "Dv2A": "signal_dms",
    "PrtN": "port_entries_dms",
}
EXPECTED_REGION = "dms"
EXPECTED_TTL = "port_entries_dms"

# Two sessions that share the same fiber region (dms) but record different behavioral
# events, so each event's group average has a single contributing session (n=1).
DISJOINT_STORENAMES = {
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

    step1(**common_kwargs, storenames_map=STORENAMES_MAP)
    step2(**common_kwargs, selected_runs=selected_runs)
    step3(**common_kwargs, selected_runs=selected_runs)
    step4(**common_kwargs, selected_runs=selected_runs)

    # Run group averaging pass
    step4(
        **common_kwargs,
        average_for_group=True,
        group_folders=selected_folders,
        selected_runs=selected_runs,
        group_selected_runs=selected_runs,
    )

    average_directory = temporary_base_directory / "average"
    assert average_directory.is_dir(), f"No average directory found under {temporary_base_directory}"

    group_psth_file_path = os.path.join(
        average_directory,
        f"{EXPECTED_TTL}_{EXPECTED_REGION}_z_score_{EXPECTED_REGION}.h5",
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
    """Group-average and visualize two sessions that share the same fiber region but
    record different behavioral events (one 'rewarded', one 'unrewarded').

    Reproduces issue #368: the sessions have non-identical storename sets, so this
    exercises the relaxed fiber-region validation (averaging is no longer blocked).
    Because each event is present in only one session, its group average has a single
    contributing session (n=1), which also exercises the single-trial heatmap that
    previously blanked the visualization dashboard with a Bokeh stack overflow.
    """
    base_dir, selected_folders = copied_sessions
    temporary_base_directory = Path(base_dir)
    selected_runs = {folder: ["1"] for folder in selected_folders}

    # Step 1 is run per session so each gets a different behavioral-event storename
    # while sharing the same control/signal (dms) fiber region.
    for session_folder, subdir in zip(selected_folders, SESSION_SUBDIRS):
        step1(base_dir=base_dir, selected_folders=[session_folder], storenames_map=DISJOINT_STORENAMES[subdir])

    common_kwargs = dict(base_dir=base_dir, selected_folders=selected_folders)
    step2(**common_kwargs, selected_runs=selected_runs)
    step3(**common_kwargs, selected_runs=selected_runs)
    step4(**common_kwargs, selected_runs=selected_runs)
    step4(
        **common_kwargs,
        average_for_group=True,
        group_folders=selected_folders,
        selected_runs=selected_runs,
        group_selected_runs=selected_runs,
    )

    # Both events must be averaged even though no session has both -- cross-condition
    # averaging that the pre-#368 validation rejected outright.
    average_directory = temporary_base_directory / "average"
    expected_columns_by_event = {
        "rewarded_nose_pokes": "Photo_048_392-200728-121222_output_1",
        "unrewarded_nose_pokes": "Photo_63_207-181030-103332_output_1",
    }
    for event, contributing_session in expected_columns_by_event.items():
        average_path = average_directory / f"{event}_{EXPECTED_REGION}_z_score_{EXPECTED_REGION}.h5"
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
                visualize_average_results=True,
                group_folders=selected_folders,
                group_selected_runs=selected_runs,
            )

    assert len(captured_dashboards) >= 1, "step5 created no VisualizationDashboard instances"
    plotter = captured_dashboards[0].plotter
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

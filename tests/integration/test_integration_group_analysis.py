import json
import shutil
from pathlib import Path
from unittest.mock import patch

import holoviews as hv
import numpy as np
import pandas as pd
import pytest

from guppy.analysis.standard_io import (
    read_covariate_correlations_from_hdf5,
    read_tonic_from_hdf5,
)
from guppy.frontend.visualization_dashboard import VisualizationDashboard
from guppy.testing import default_output_root_folder
from guppy.testing.api import (
    group_analysis,
    label_groups,
    locate_run_folder,
    step1,
    step2,
    step3,
    step4,
    step5,
    tonic_analysis,
)
from guppy.testing.covariate_session import RECORDING_SITE as COVARIATE_RECORDING_SITE
from guppy.testing.covariate_session import SESSION_NAME as COVARIATE_SESSION_NAME
from guppy.testing.covariate_session import run_covariate_session
from guppy.utils.utils import output_label_under, parse_run_name
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


def _copy_sessions(temporary_base_directory):
    """Copy the two sample TDT sessions into ``temporary_base_directory`` with prior outputs removed.

    Returns
    -------
    tuple[str, list[str]]
        ``(base_dir, selected_folders)`` ready to drive the pipeline API.
    """
    source_sessions = [STUBBED_TESTING_DATA / subdir for subdir in SESSION_SUBDIRS]
    for source_session in source_sessions:
        assert source_session.is_dir(), f"Sample data not available at expected path: {source_session}"

    session_copies = []
    for source_session in source_sessions:
        session_name = source_session.name
        session_copy = temporary_base_directory / session_name
        shutil.copytree(source_session, session_copy)
        for output_directory in list(Path(session_copy).glob(f"{session_name}_output_*")):
            assert Path(output_directory).is_dir()
            shutil.rmtree(output_directory)
        parameters_path = session_copy / "GuPPyParamtersUsed.json"
        if parameters_path.exists():
            parameters_path.unlink()
        session_copies.append(session_copy)

    return str(temporary_base_directory), [str(session_copy) for session_copy in session_copies]


@pytest.fixture
def copied_sessions(tmp_path):
    temporary_base_directory = tmp_path / "input_root_folder"
    temporary_base_directory.mkdir()
    return _copy_sessions(temporary_base_directory)


@pytest.fixture(scope="module")
def processed_sessions(tmp_path_factory):
    """Both sessions labeled alike and run through Steps 1-4, shared by the tests that group them."""
    base_dir, selected_folders = _copy_sessions(tmp_path_factory.mktemp("input_root_folder"))
    common_kwargs = dict(base_dir=base_dir, selected_folders=selected_folders)
    selected_runs = {folder: ["1"] for folder in selected_folders}

    step1(**common_kwargs, store_id_to_store_label=STORE_ID_TO_STORE_LABEL)
    step2(**common_kwargs, selected_runs=selected_runs)
    step3(**common_kwargs, selected_runs=selected_runs)
    step4(**common_kwargs, selected_runs=selected_runs)
    return base_dir, selected_folders


@pytest.fixture(scope="module")
def saline_group(processed_sessions):
    """Both processed runs grouped as ``saline`` and averaged; returns the run folders it holds."""
    base_dir, selected_folders = processed_sessions
    member_run_folders = [locate_run_folder(session=folder) for folder in selected_folders]
    label_groups(member_run_folders=member_run_folders, destination_directory=base_dir, group_name="saline")
    group_analysis(base_dir=base_dir, selected_group_folders=[Path(base_dir) / "saline_group"])
    return member_run_folders


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_group_analysis(processed_sessions, saline_group):
    """
    Integration test: run the full pipeline (Steps 2-5) on two TDT sessions and then
    perform group-level averaging, asserting that the average directory and expected
    output files are created with the correct structure.
    """
    base_dir, selected_folders = processed_sessions
    temporary_base_directory = Path(base_dir)
    selected_runs = {folder: ["1"] for folder in selected_folders}

    group_directory = temporary_base_directory / "saline_group"
    assert group_directory.is_dir(), f"No group directory found under {temporary_base_directory}"

    group_psth_file_path = Path(group_directory) / (
        f"{EXPECTED_TTL}_{EXPECTED_RECORDING_SITE}_z_score_{EXPECTED_RECORDING_SITE}.h5"
    )
    assert Path(group_psth_file_path).exists(), f"Missing group PSTH HDF5: {group_psth_file_path}"

    group_psth_dataframe = pd.read_hdf(group_psth_file_path, key="df")
    assert "timestamps" in group_psth_dataframe.columns, f"'timestamps' column missing in {group_psth_file_path}"
    assert "mean" in group_psth_dataframe.columns, f"'mean' column missing in {group_psth_file_path}"

    hv.extension("bokeh")
    captured_dashboards: list[VisualizationDashboard] = []
    original_init = VisualizationDashboard.__init__

    def capturing_init(self, **kwargs):
        original_init(self, **kwargs)
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
    member_run_folders = [locate_run_folder(session=folder) for folder in selected_folders]
    label_groups(
        member_run_folders=member_run_folders,
        destination_directory=base_dir,
        group_name="cross_condition",
    )
    group_analysis(base_dir=base_dir, selected_group_folders=[Path(base_dir) / "cross_condition_group"])

    # Both events must be averaged even though no session has both -- cross-condition
    # averaging that the pre-#368 validation rejected outright.
    average_directory = temporary_base_directory / "cross_condition_group"
    expected_columns_by_event = {
        "rewarded_nose_pokes": "Photo_048_392-200728-121222/output_1",
        "unrewarded_nose_pokes": "Photo_63_207-181030-103332/output_1",
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

    def capturing_init(self, **kwargs):
        original_init(self, **kwargs)
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
def test_group_analysis_step_writes_a_named_group_directory(processed_sessions, saline_group):
    """The Group Analysis step averages selected runs into <destination>/<name>_group.

    Runs Steps 1-4 per session, then the new Group Analysis step, and asserts the group
    directory's name, manifest, provenance snapshot, stores list and averaged PSTH.
    """
    base_dir, _ = processed_sessions
    member_run_folders = saline_group

    group_folder = Path(base_dir) / "saline_group"
    assert group_folder.is_dir(), f"No 'saline_group' directory under {base_dir}"
    # The legacy, location-derived output directory is not written any more.
    assert not (Path(base_dir) / "average").exists()

    with (group_folder / "group_members.json").open() as manifest_file:
        assert json.load(manifest_file) == {"member_run_folders": member_run_folders}

    assert (group_folder / "GuPPyParamtersUsed.json").exists()

    stores_list = np.genfromtxt(group_folder / "storesList.csv", dtype="str", delimiter=",").reshape(2, -1)
    assert EXPECTED_TTL in stores_list[1, :].tolist()

    group_psth_path = group_folder / f"{EXPECTED_TTL}_{EXPECTED_RECORDING_SITE}_z_score_{EXPECTED_RECORDING_SITE}.h5"
    group_psth = pd.read_hdf(group_psth_path, key="df")
    # One column per member run, named by its path under the output directory, plus
    # mean/err/timestamps.
    output_base = default_output_root_folder(base_dir=base_dir)
    for run_folder in member_run_folders:
        assert output_label_under(path=run_folder, root=output_base) in group_psth.columns
    assert list(group_psth.columns[-3:]) == ["timestamps", "mean", "err"]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_group_analysis_step_rebuilds_the_group_when_a_member_is_dropped(processed_sessions):
    """Re-running a group with fewer members must not leave the dropped member behind."""
    base_dir, selected_folders = processed_sessions
    # Its own group name, so rebuilding it leaves the shared saline group untouched.
    member_run_folders = [locate_run_folder(session=folder) for folder in selected_folders]
    label_groups(
        member_run_folders=member_run_folders,
        destination_directory=base_dir,
        group_name="dropped",
    )
    group_analysis(base_dir=base_dir, selected_group_folders=[Path(base_dir) / "dropped_group"])
    group_folder = Path(base_dir) / "dropped_group"
    psth_path = group_folder / f"{EXPECTED_TTL}_{EXPECTED_RECORDING_SITE}_z_score_{EXPECTED_RECORDING_SITE}.h5"
    assert len(pd.read_hdf(psth_path, key="df").columns) == 5  # 2 members + timestamps/mean/err

    label_groups(
        member_run_folders=member_run_folders[:1],
        destination_directory=base_dir,
        group_name="dropped",
    )
    group_analysis(base_dir=base_dir, selected_group_folders=[Path(base_dir) / "dropped_group"])

    with (group_folder / "group_members.json").open() as manifest_file:
        assert json.load(manifest_file) == {"member_run_folders": member_run_folders[:1]}
    remaining = pd.read_hdf(psth_path, key="df")
    output_base = default_output_root_folder(base_dir=base_dir)
    assert output_label_under(path=member_run_folders[0], root=output_base) in remaining.columns
    assert output_label_under(path=member_run_folders[1], root=output_base) not in remaining.columns


# The covariate sample session runs 600 s; one epoch in each half.
TONIC_EPOCHS = pd.DataFrame({"label": ["early", "late"], "start": [10.0, 310.0], "end": [290.0, 590.0]})


@pytest.fixture(scope="module")
def tonic_and_covariate_group(tmp_path_factory):
    """Two copies of the covariate sample session through Step 4 and Tonic Analysis, grouped.

    Returns
    -------
    tuple[Path, list[str], str]
        ``(group_folder, member_run_folders, base_dir)``.
    """
    source_directory = tmp_path_factory.mktemp("sources")
    second_source = source_directory / (COVARIATE_SESSION_NAME.removesuffix("_1") + "_2")
    shutil.copytree(
        STUBBED_TESTING_DATA / "csv" / COVARIATE_SESSION_NAME,
        second_source,
        ignore=shutil.ignore_patterns("*_output_*"),
    )
    base_directory = tmp_path_factory.mktemp("group_tonic_covariates")
    base_dir = str(base_directory)
    member_run_folders = [
        run_covariate_session(session_path=session_path, base_directory=base_directory)
        for session_path in [STUBBED_TESTING_DATA / "csv" / COVARIATE_SESSION_NAME, second_source]
    ]
    sessions = [str(base_directory / COVARIATE_SESSION_NAME), str(base_directory / second_source.name)]
    tonic_analysis(
        base_dir=base_dir,
        selected_folders=sessions,
        tonic_epochs={COVARIATE_RECORDING_SITE: TONIC_EPOCHS},
        selected_runs={
            session: [parse_run_name(run_folder)]
            for session, run_folder in zip(sessions, member_run_folders, strict=True)
        },
    )
    label_groups(member_run_folders=member_run_folders, destination_directory=base_dir, group_name="injected")
    group_folder = Path(base_dir) / "injected_group"
    group_analysis(base_dir=base_dir, selected_group_folders=[group_folder])
    return group_folder, member_run_folders, base_dir


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestGroupTonicAndCovariateTables:
    def test_tonic_member_table_stacks_each_member(self, tonic_and_covariate_group):
        group_folder, member_run_folders, base_dir = tonic_and_covariate_group
        output_base = default_output_root_folder(base_dir=base_dir)

        stacked = pd.read_hdf(group_folder / f"group_tonic_{COVARIATE_RECORDING_SITE}.h5", key="df")

        assert list(stacked.index.names) == ["member", "epoch"]
        for run_folder in member_run_folders:
            member = stacked.loc[output_label_under(path=run_folder, root=output_base)]
            pd.testing.assert_frame_equal(member, read_tonic_from_hdf5(run_folder, COVARIATE_RECORDING_SITE))
        assert (group_folder / f"group_tonic_{COVARIATE_RECORDING_SITE}.csv").exists()

    def test_tonic_summary_of_identical_members(self, tonic_and_covariate_group):
        group_folder, member_run_folders, _ = tonic_and_covariate_group

        summary = pd.read_hdf(group_folder / f"group_tonic_summary_{COVARIATE_RECORDING_SITE}.h5", key="df")

        # Both members are copies of one session, so the mean is that session's value with no spread.
        member = read_tonic_from_hdf5(member_run_folders[0], COVARIATE_RECORDING_SITE)
        assert list(summary.index) == ["early", "late"]
        np.testing.assert_allclose(summary["mean_zscore_mean"], member["mean_zscore"])
        np.testing.assert_allclose(summary["mean_zscore_sem"], [0.0, 0.0], atol=1e-12)
        np.testing.assert_array_equal(summary["mean_dff_n"], [2, 2])

    def test_covariate_summary_of_identical_members(self, tonic_and_covariate_group):
        group_folder, member_run_folders, _ = tonic_and_covariate_group

        stacked = pd.read_hdf(group_folder / f"group_covariate_correlations_{COVARIATE_RECORDING_SITE}.h5", key="df")
        summary = pd.read_hdf(
            group_folder / f"group_covariate_correlations_summary_{COVARIATE_RECORDING_SITE}.h5", key="df"
        )

        assert list(stacked.index.names) == ["member", "metric", "covariate"]
        assert list(stacked.columns) == ["pearson_r", "spearman_rho", "n_bins"]
        # The driving covariate's r for this session, as asserted in test_covariate_correlations.py.
        assert summary.loc[("mean_zscore", "akinesia"), "pearson_r_mean"] == pytest.approx(0.8436, abs=0.02)
        assert summary.loc[("mean_zscore", "akinesia"), "pearson_r_sem"] == pytest.approx(0.0, abs=1e-12)
        assert summary.loc[("mean_zscore", "akinesia"), "pearson_r_n"] == 2
        assert len(summary) == len(
            read_covariate_correlations_from_hdf5(
                filepath=member_run_folders[0], recording_site=COVARIATE_RECORDING_SITE
            )
        )

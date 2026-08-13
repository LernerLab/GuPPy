import glob
import json
import os
import shutil
from unittest.mock import patch

import holoviews as hv
import numpy as np
import pytest

from guppy.analysis.io_utils import read_hdf5
from guppy.analysis.standard_io import read_transients_from_hdf5
from guppy.frontend.visualization_dashboard import VisualizationDashboard
from guppy.testing.api import step1, step2, step3, step4, step5
from guppy.utils.utils import read_Df
from guppy_test_data import STUBBED_TESTING_DATA

SESSION_SUBDIR = "csv/sample_data_csv_1"
STORE_ID_TO_STORE_LABEL = {
    "Sample_Control_Channel": "control_region",
    "Sample_Signal_Channel": "signal_region",
    "Sample_TTL": "ttl",
}
EXPECTED_RECORDING_SITE = "region"
TRANSIENT_EVENT = f"transients_z_score_{EXPECTED_RECORDING_SITE}"
METRIC_BASENAME = f"z_score_{EXPECTED_RECORDING_SITE}"


@pytest.fixture
def run_pipeline(tmp_path):
    """Return a callable that runs Steps 1-4 on a fresh copy of the stubbed CSV session.

    Each call gets its own workspace, so a run with the spontaneous-mode toggle on can be
    compared against a run with it off. The returned dict also carries the arguments a
    follow-up Step 5 needs.
    """
    source_session = STUBBED_TESTING_DATA / SESSION_SUBDIR
    assert source_session.is_dir(), f"Sample data not available at expected path: {source_session}"

    def _run(workspace_name, **step4_kwargs):
        temporary_base_directory = tmp_path / workspace_name
        temporary_base_directory.mkdir(parents=True, exist_ok=True)
        session_name = source_session.name
        session_copy = temporary_base_directory / session_name
        shutil.copytree(source_session, session_copy)

        for output_directory in glob.glob(os.path.join(session_copy, f"{session_name}_output_*")):
            shutil.rmtree(output_directory)
        parameters_path = session_copy / "GuPPyParamtersUsed.json"
        if parameters_path.exists():
            parameters_path.unlink()

        common_kwargs = dict(base_dir=str(temporary_base_directory), selected_folders=[str(session_copy)])
        selected_runs = {str(session_copy): ["1"]}

        step1(**common_kwargs, store_id_to_store_label=STORE_ID_TO_STORE_LABEL)
        step2(**common_kwargs, selected_runs=selected_runs)
        step3(**common_kwargs, selected_runs=selected_runs)
        step4(**common_kwargs, selected_runs=selected_runs, **step4_kwargs)

        return {
            "output_directory": os.path.join(session_copy, f"{session_name}_output_1"),
            "common_kwargs": common_kwargs,
            "selected_runs": selected_runs,
        }

    return _run


@pytest.fixture
def visualized_events():
    """Return a callable that runs Step 5 headlessly and reports the events it offered."""

    def _run(pipeline_result, **step5_kwargs):
        # ParameterizedPlotter evaluates holoviews opts during VisualizationDashboard.__init__,
        # so the Bokeh backend must be registered before instantiation.
        hv.extension("bokeh")
        captured_dashboards = []
        original_init = VisualizationDashboard.__init__

        def capturing_init(self, *, plotter, basename):
            original_init(self, plotter=plotter, basename=basename)
            captured_dashboards.append(self)

        with patch.object(VisualizationDashboard, "__init__", capturing_init):
            with patch.object(VisualizationDashboard, "show", lambda self: None):
                step5(
                    **pipeline_result["common_kwargs"],
                    selected_runs=pipeline_result["selected_runs"],
                    **step5_kwargs,
                )

        return [event for dashboard in captured_dashboards for event in dashboard.plotter.event_selector_objects]

    return _run


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_transient_event_train_is_written_and_drives_the_psth(run_pipeline):
    output_directory = run_pipeline("spontaneous", use_transients_as_events=True)["output_directory"]

    event_timestamps = np.asarray(read_hdf5(TRANSIENT_EVENT, output_directory, "ts")).ravel()
    _, detector_timestamps, peaks_index = read_transients_from_hdf5(output_directory, METRIC_BASENAME)
    detected_timestamps = detector_timestamps[peaks_index]

    # The PSTH step drops transients too close to the recording start or to each other, so the
    # event train it kept is a non-empty subset of what the detector found.
    assert event_timestamps.size > 0
    assert np.isin(event_timestamps, detected_timestamps).all()

    # Every kept transient is one PSTH trial; trial columns are labeled with the event time.
    psth = read_Df(output_directory, TRANSIENT_EVENT, METRIC_BASENAME)
    trial_labels = [column for column in psth.columns if column not in ("timestamps", "mean", "err")]
    np.testing.assert_allclose(np.asarray(trial_labels, dtype=float), event_timestamps, atol=1e-6)

    assert os.path.exists(os.path.join(output_directory, f"peak_AUC_{TRANSIENT_EVENT}_{METRIC_BASENAME}.csv"))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_no_transient_event_train_when_the_toggle_is_off(run_pipeline):
    output_directory = run_pipeline("ttl_only")["output_directory"]

    assert not os.path.exists(os.path.join(output_directory, f"{TRANSIENT_EVENT}.hdf5"))
    assert not os.path.exists(os.path.join(output_directory, f"{TRANSIENT_EVENT}_{METRIC_BASENAME}.h5"))
    # The external TTL event is unaffected.
    assert os.path.exists(os.path.join(output_directory, f"ttl_{EXPECTED_RECORDING_SITE}.hdf5"))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_both_metrics_produce_two_independent_event_trains(run_pipeline):
    """z-score and dF/F transients are separate events, so neither overwrites the other."""
    output_directory = run_pipeline("both_metrics", use_transients_as_events=True, select_for_transients="Both")[
        "output_directory"
    ]

    from_z_score = np.asarray(read_hdf5(TRANSIENT_EVENT, output_directory, "ts")).ravel()
    from_dff = np.asarray(read_hdf5(f"transients_dff_{EXPECTED_RECORDING_SITE}", output_directory, "ts")).ravel()

    assert from_z_score.size > 0
    assert from_dff.size > 0
    for event in (TRANSIENT_EVENT, f"transients_dff_{EXPECTED_RECORDING_SITE}"):
        assert os.path.exists(os.path.join(output_directory, f"{event}_{METRIC_BASENAME}.h5"))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_spontaneous_mode_choice_is_recorded_in_the_parameter_snapshot(run_pipeline):
    output_directory = run_pipeline("recorded", use_transients_as_events=True)["output_directory"]

    with open(os.path.join(output_directory, "GuPPyParamtersUsed.json")) as parameters_file:
        parameters = json.load(parameters_file)
    assert parameters["useTransientsAsEvents"] is True


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_transient_event_is_offered_in_the_visualization_dashboard(run_pipeline, visualized_events):
    pipeline_result = run_pipeline("visualized", use_transients_as_events=True)

    events = visualized_events(pipeline_result, use_transients_as_events=True)

    # The external TTL is still offered alongside the transient event train.
    assert events == [f"ttl_{EXPECTED_RECORDING_SITE}", TRANSIENT_EVENT]

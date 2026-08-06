"""Contract tests for orchestrate_read_raw_data error enrichment."""

import shutil
import time

import pytest

from guppy.orchestration.read_raw_data import orchestrate_read_raw_data
from guppy.utils.progress import StepProgress, _current_step
from guppy_test_data import STUBBED_TESTING_DATA

# orchestrate_read_raw_data writes the parameter snapshot (save_parameters) into each
# selected output dir before reading, so it needs the full analysis-parameter set.
DEFAULT_ANALYSIS_PARAMETERS = {
    "combine_data": False,
    "isosbestic_control": True,
    "control_fit_method": "IRWLS",
    "controlFitWindowMode": "full trace",
    "controlFitWindowStart": 0,
    "controlFitWindowEnd": 0,
    "timeForLightsTurnOn": 1,
    "filter_window": 100,
    "removeArtifacts": False,
    "artifactsRemovalMethod": "concatenate",
    "noChannels": 2,
    "zscore_method": "standard z-score",
    "baselineWindowStart": 0,
    "baselineWindowEnd": 0,
    "nSecPrev": -10,
    "nSecPost": 20,
    "computeCorr": False,
    "timeInterval": 2,
    "bin_psth_trials": 0,
    "use_time_or_trials": "Time (min)",
    "baselineCorrectionStart": -5,
    "baselineCorrectionEnd": 0,
    "peak_startPoint": [-5.0, 0.0, 5.0],
    "peak_endPoint": [0.0, 3.0, 10.0],
    "selectForComputePsth": "z_score",
    "selectForTransientsComputation": "z_score",
    "moving_window": 15,
    "highAmpFilt": 2,
    "transientsThresh": 3,
    "visualize_zscore_or_dff": "z_score",
    "averageForGroup": False,
}


class TestOrchestrateReadRawDataErrorEnrichment:
    """Missing-event error must list the events the extractor did discover (issue #270)."""

    @pytest.fixture
    def session_with_bogus_event(self, tmp_path):
        """Copy a real stubbed Doric session and add a bogus event to storesList.csv."""
        source_folder = STUBBED_TESTING_DATA / "doric" / "sample_doric_1"
        session_folder = tmp_path / "sample_doric_1"
        shutil.copytree(source_folder, session_folder)

        run_folder = session_folder / "sample_doric_1_output_1"
        run_folder.mkdir()
        stores_list_path = run_folder / "storesList.csv"
        stores_list_path.write_text("NotARealEvent\nsignal_DMS\n")

        return str(session_folder)

    def test_missing_event_error_lists_available_events(self, session_with_bogus_event):
        input_parameters = {
            **DEFAULT_ANALYSIS_PARAMETERS,
            "session_folders": [session_with_bogus_event],
            "numberOfCores": 1,
            "noChannels": 2,
            "selected_runs": {session_with_bogus_event: ["1"]},
        }
        with pytest.raises(ValueError) as exception_info:
            orchestrate_read_raw_data(input_parameters)

        message = str(exception_info.value)
        assert "'NotARealEvent'" in message
        assert "not found in any extractor" in message
        assert "Available events:" in message
        assert "AIn-1 - Raw" in message
        assert "AIn-2 - Raw" in message
        assert "DI--O-1" in message


@pytest.fixture
def bound_step():
    """Bind a StepProgress for the duration of one test, as ``home.py`` does per step run."""
    step = StepProgress()
    token = _current_step.set(step)
    yield step
    _current_step.reset(token)


class TestProgressAccountingEndToEnd:
    """A full step-2 run reports its sample total up front and then tracks the shared
    counter the workers advance. Exercises the pre-flight count, the counter handoff, and
    the pull-through in one shot.
    """

    def test_two_event_run_reports_total_and_tracks_the_counter(self, tmp_path, monkeypatch, bound_step):
        from guppy.orchestration import read_raw_data as read_raw_data_module

        # Build a session folder with a storesList.csv referencing two fake events.
        session_folder = tmp_path / "session"
        run_folder = session_folder / "session_output_1"
        run_folder.mkdir(parents=True)
        (run_folder / "storesList.csv").write_text("event_a,event_b\nsignal_a,signal_b\n")

        # Two distinct instances so the run produces two tasks, letting the second read
        # observe what the first already committed. count_samples reports per-event totals;
        # read+save are no-ops on disk.
        observed_during_read = []

        class _ObservingExtractor:
            def __init__(self, event, sample_count):
                self.event = event
                self.sample_count = sample_count

            def count_samples(self, *, event):
                return self.sample_count

            def read(self, *, events, outputPath):
                observed_during_read.append(bound_step.value)
                time.sleep(0.05)
                return [{"store_id": event, "timestamps": [0.0]} for event in events]

            def save(self, *, output_dicts, outputPath):
                return None

        def fake_build_event_to_extractor(*, folder_path, store_array, inputParameters):
            return {
                "event_a": _ObservingExtractor("event_a", 100),
                "event_b": _ObservingExtractor("event_b", 200),
            }

        monkeypatch.setattr(read_raw_data_module, "_build_event_to_extractor", fake_build_event_to_extractor)

        # numberOfCores=1 routes orchestrate_read_raw_data through its serial path,
        # which installs _SAMPLES_DONE in the parent process and calls the
        # read-and-save unit function directly — no pool, no separate patch needed.
        input_parameters = {
            **DEFAULT_ANALYSIS_PARAMETERS,
            "session_folders": [str(session_folder)],
            "numberOfCores": 1,
            "noChannels": 2,
            "selected_runs": {str(session_folder): ["1"]},
        }
        orchestrate_read_raw_data(input_parameters)

        assert bound_step.total == 300
        assert bound_step.value == 300
        # Progress is visible while the run is still going, not only at the end: the first
        # read sees nothing committed, the second sees event_a's 100 samples.
        assert observed_during_read == [0, 100]

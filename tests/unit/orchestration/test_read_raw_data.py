"""Contract tests for orchestrate_read_raw_data error enrichment."""

import multiprocessing as mp
import shutil
import threading
import time

import pytest
from conftest import STUBBED_TESTING_DATA

from guppy.orchestration.read_raw_data import (
    _progress_poller,
    orchestrate_read_raw_data,
)

# orchestrate_read_raw_data writes the parameter snapshot (save_parameters) into each
# selected output dir before reading, so it needs the full analysis-parameter set.
DEFAULT_ANALYSIS_PARAMETERS = {
    "combine_data": False,
    "isosbestic_control": True,
    "control_fit_method": "IRWLS",
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
    "plot_zScore_dff": "None",
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


class TestProgressPoller:
    """``_progress_poller`` flushes the shared samples counter to ``PB_STEPS_FILE``."""

    def test_writes_value_times_ten_on_change(self, tmp_path):
        progress_file = tmp_path / "pb.txt"
        samples_done = mp.Value("q", 0)
        stop_event = threading.Event()
        thread = threading.Thread(
            target=_progress_poller,
            args=(samples_done, stop_event),
            kwargs={"file_path": str(progress_file)},
        )
        thread.start()
        try:
            # Drive three distinct values past the poller; sleep long enough between
            # each so the 200ms poll loop sees them as separate ticks.
            for value in (5, 12, 30):
                with samples_done.get_lock():
                    samples_done.value = value
                time.sleep(0.3)
        finally:
            stop_event.set()
            thread.join(timeout=2.0)

        written_values = [int(line.strip()) for line in progress_file.read_text().splitlines() if line.strip()]
        assert written_values == [50, 120, 300]

    def test_skips_redundant_writes_when_value_unchanged(self, tmp_path):
        progress_file = tmp_path / "pb.txt"
        samples_done = mp.Value("q", 7)
        stop_event = threading.Event()
        thread = threading.Thread(
            target=_progress_poller,
            args=(samples_done, stop_event),
            kwargs={"file_path": str(progress_file)},
        )
        thread.start()
        time.sleep(0.7)  # ~3 poll cycles with no value change
        stop_event.set()
        thread.join(timeout=2.0)

        written_values = [int(line.strip()) for line in progress_file.read_text().splitlines() if line.strip()]
        # The first poll writes the value once; subsequent polls with the same value
        # must not write again.
        assert written_values == [70]


class TestProgressFileAccountingEndToEnd:
    """A full step-2 run with a fake sleeping extractor produces a monotonic, fully
    reconciled progress file. Exercises the pre-flight count, pool worker handoff,
    poller writes, and final reconciliation in one shot.
    """

    def test_two_event_run_produces_monotonic_progress_and_reconciles_to_total(self, tmp_path, monkeypatch):
        from guppy.orchestration import read_raw_data as read_raw_data_module

        progress_file = tmp_path / "pb_steps.txt"
        monkeypatch.setattr(read_raw_data_module, "PB_STEPS_FILE", str(progress_file))

        # Build a session folder with a storesList.csv referencing two fake events.
        session_folder = tmp_path / "session"
        run_folder = session_folder / "session_output_1"
        run_folder.mkdir(parents=True)
        (run_folder / "storesList.csv").write_text("event_a,event_b\nsignal_a,signal_b\n")

        # Fake extractor that sleeps inside read so the poller has time to tick.
        # count_samples reports per-event totals; read+save are no-ops on disk.
        class _SleepingExtractor:
            samples_per_event = {"event_a": 100, "event_b": 200}

            def count_samples(self, *, event):
                return _SleepingExtractor.samples_per_event[event]

            def read(self, *, events, outputPath):
                time.sleep(0.4)
                return [{"store_id": event, "timestamps": [0.0]} for event in events]

            def save(self, *, output_dicts, outputPath):
                return None

        fake_extractor = _SleepingExtractor()

        def fake_build_event_to_extractor(*, folder_path, store_array, inputParameters):
            return {"event_a": fake_extractor, "event_b": fake_extractor}

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

        written_values = [int(line.strip()) for line in progress_file.read_text().splitlines() if line.strip()]
        total_samples = 100 + 200
        expected_max = total_samples * 10

        # First line is the bar's max; the rest are monotonic progress ticks ending at max.
        assert written_values[0] == expected_max
        progress_ticks = written_values[1:]
        assert progress_ticks == sorted(progress_ticks)
        assert progress_ticks[-1] == expected_max
        # Pre-flight wrote a 0 immediately after the max, so we get at least two ticks.
        assert len(progress_ticks) >= 2

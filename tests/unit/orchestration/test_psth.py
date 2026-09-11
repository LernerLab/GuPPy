import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from guppy.orchestration.psth import (
    _validate_events_overlap_signal,
    _validate_psth_window_parameters,
    execute_compute_cross_correlation,
    execute_compute_psth,
    execute_compute_psth_peak_and_area,
    run_psth_step,
)
from guppy.utils._hdf5_io import write_hdf5
from guppy.utils.progress import StepProgress, _current_step


@pytest.fixture
def psth_output_dir(tmp_path):
    """Create a minimal output directory with a z_score_DMS.hdf5 for glob discovery."""
    run_folder = tmp_path / "session1" / "session1_output_1"
    run_folder.mkdir(parents=True)
    with h5py.File(str(run_folder / "z_score_DMS.hdf5"), "w") as hdf5_file:
        hdf5_file.create_dataset("data", data=np.zeros(100))
    return run_folder


# ---------------------------------------------------------------------------
# execute_compute_psth — early-exit for control/signal events
# ---------------------------------------------------------------------------


def test_execute_compute_psth_returns_zero_for_control_event(psth_output_dir, base_input_parameters):
    result = execute_compute_psth(str(psth_output_dir), "control_DMS", base_input_parameters)
    assert result == 0


def test_execute_compute_psth_returns_zero_for_signal_event(psth_output_dir, base_input_parameters):
    result = execute_compute_psth(str(psth_output_dir), "signal_DMS", base_input_parameters)
    assert result == 0


def test_execute_compute_psth_returns_zero_for_mixed_case_control(psth_output_dir, base_input_parameters):
    result = execute_compute_psth(str(psth_output_dir), "Control_DMS", base_input_parameters)
    assert result == 0


# ---------------------------------------------------------------------------
# execute_compute_psth_peak_and_area — early-exit for control/signal events
# ---------------------------------------------------------------------------


def test_execute_compute_psth_peak_and_area_returns_zero_for_control_event(psth_output_dir, base_input_parameters):
    result = execute_compute_psth_peak_and_area(str(psth_output_dir), "control_DMS", base_input_parameters)
    assert result == 0


def test_execute_compute_psth_peak_and_area_returns_zero_for_signal_event(psth_output_dir, base_input_parameters):
    result = execute_compute_psth_peak_and_area(str(psth_output_dir), "signal_DMS", base_input_parameters)
    assert result == 0


# ---------------------------------------------------------------------------
# execute_compute_psth_peak_and_area — AUC units
# ---------------------------------------------------------------------------


@pytest.fixture
def peak_and_area_output_dir(psth_output_dir):
    """Add the timeCorrection and PSTH files execute_compute_psth_peak_and_area reads.

    The PSTH is a single trial rising to 2.0 and back to 0.0 on a 0.5 s time axis,
    so the trapezoid over the (0.0, 1.5) window has hand-checkable areas: 3.0 with
    one-sample spacing and 1.5 in z-score*seconds.
    """
    with h5py.File(str(psth_output_dir / "timeCorrection_DMS.hdf5"), "w") as hdf5_file:
        hdf5_file.create_dataset("sampling_rate", data=np.array([2.0]))
    psth = pd.DataFrame(
        {
            "12.5": [0.0, 2.0, 2.0, 0.0],
            "mean": [0.0, 2.0, 2.0, 0.0],
            "err": [0.0, 0.0, 0.0, 0.0],
            "timestamps": [0.0, 0.5, 1.0, 1.5],
        }
    )
    psth.to_hdf(str(psth_output_dir / "lever_press_DMS_z_score_DMS.h5"), key="df", mode="w")
    return psth_output_dir


@pytest.mark.parametrize("auc_units, expected_area", [("samples", 3.0), ("seconds", 1.5)])
def test_execute_compute_psth_peak_and_area_honors_auc_units(
    peak_and_area_output_dir, base_input_parameters, auc_units, expected_area
):
    base_input_parameters["peak_startPoint"] = [0.0]
    base_input_parameters["peak_endPoint"] = [1.5]
    base_input_parameters["auc_units"] = auc_units

    execute_compute_psth_peak_and_area(str(peak_and_area_output_dir), "lever_press", base_input_parameters)

    written = pd.read_csv(peak_and_area_output_dir / "peak_AUC_lever_press_DMS_z_score_DMS.csv", index_col=0)
    np.testing.assert_allclose(written["area_1"].to_numpy(), np.full(2, expected_area), atol=1e-12)


# ---------------------------------------------------------------------------
# execute_compute_cross_correlation
# ---------------------------------------------------------------------------


def test_execute_compute_cross_correlation_no_op_when_compute_corr_false(
    psth_output_dir, base_input_parameters, monkeypatch
):
    get_corr_calls = []
    monkeypatch.setattr(
        "guppy.orchestration.psth.getCorrCombinations",
        lambda filepath, inputParameters: get_corr_calls.append(filepath) or ([], []),
    )

    base_input_parameters["computeCorr"] = False
    execute_compute_cross_correlation(str(psth_output_dir), "lever_press", base_input_parameters)

    assert len(get_corr_calls) == 0


def test_execute_compute_cross_correlation_no_op_for_transient_events(
    psth_output_dir, base_input_parameters, monkeypatch
):
    """Each recording site's transients are its own event train, so there are no shared trials."""
    get_corr_calls = []
    monkeypatch.setattr(
        "guppy.orchestration.psth.getCorrCombinations",
        lambda filepath, inputParameters: get_corr_calls.append(filepath) or ([], []),
    )

    base_input_parameters["computeCorr"] = True
    base_input_parameters["useTransientsAsEvents"] = True
    execute_compute_cross_correlation(str(psth_output_dir), "transients_z_score", base_input_parameters)

    assert get_corr_calls == []


def _record_artifact_provenance(run_folder, *, remove_artifacts, artifacts_removal_method):
    """Write the artifact provenance the Remove Artifacts step would have left behind."""
    with (Path(str(run_folder)) / "GuPPyParamtersUsed.json").open("w") as parameters_file:
        json.dump(
            {"removeArtifacts": remove_artifacts, "artifactsRemovalMethod": artifacts_removal_method},
            parameters_file,
        )


def test_execute_compute_cross_correlation_raises_when_run_was_concatenated(psth_output_dir, base_input_parameters):
    """The guard reads what was actually applied to this run, not a live form value."""
    base_input_parameters["computeCorr"] = True
    _record_artifact_provenance(psth_output_dir, remove_artifacts=True, artifacts_removal_method="concatenate")

    with pytest.raises(ValueError, match=r"cannot run on concatenated data"):
        execute_compute_cross_correlation(str(psth_output_dir), "lever_press", base_input_parameters)


def test_execute_compute_cross_correlation_allows_run_removed_with_nan(
    psth_output_dir, base_input_parameters, monkeypatch
):
    """A run removed with 'replace with NaN' passes the guard and proceeds to the correlation itself."""
    base_input_parameters["computeCorr"] = True
    _record_artifact_provenance(psth_output_dir, remove_artifacts=True, artifacts_removal_method="replace with NaN")
    get_corr_calls = []
    monkeypatch.setattr(
        "guppy.orchestration.psth.getCorrCombinations",
        lambda filepath, params: get_corr_calls.append(filepath) or (["dms", "vms"], ["z_score"]),
    )
    monkeypatch.setattr("guppy.orchestration.psth.read_Df", lambda *args, **kwargs: None)

    execute_compute_cross_correlation(str(psth_output_dir), "control_DMS", base_input_parameters)

    assert get_corr_calls == [str(psth_output_dir)]


def test_execute_compute_cross_correlation_returns_early_for_control_event(
    psth_output_dir, base_input_parameters, monkeypatch
):
    read_df_calls = []
    # corr_info with 2 entries so the loop would run if not for the early return
    monkeypatch.setattr(
        "guppy.orchestration.psth.getCorrCombinations",
        lambda filepath, inputParameters: (["region_1", "region_2"], ["z_score"]),
    )
    monkeypatch.setattr(
        "guppy.orchestration.psth.read_Df",
        lambda *args, **kwargs: read_df_calls.append(args),
    )

    base_input_parameters["computeCorr"] = True
    execute_compute_cross_correlation(str(psth_output_dir), "control_DMS", base_input_parameters)

    assert len(read_df_calls) == 0


def test_execute_compute_cross_correlation_raises_for_single_recording_site(
    psth_output_dir, base_input_parameters, monkeypatch
):
    """When computeCorr=True but only one signal recording site is present, a ValueError is raised."""
    monkeypatch.setattr(
        "guppy.orchestration.psth.getCorrCombinations",
        lambda filepath, inputParameters: (["dms"], ["z_score"]),
    )

    base_input_parameters["computeCorr"] = True

    with pytest.raises(ValueError, match="only one was found: 'dms'"):
        execute_compute_cross_correlation(str(psth_output_dir), "lever_press", base_input_parameters)


def test_execute_compute_cross_correlation_raises_for_no_recording_sites(
    psth_output_dir, base_input_parameters, monkeypatch
):
    """When computeCorr=True but no signal recording sites are found, a ValueError is raised."""
    monkeypatch.setattr(
        "guppy.orchestration.psth.getCorrCombinations",
        lambda filepath, inputParameters: ([], ["z_score"]),
    )

    base_input_parameters["computeCorr"] = True

    with pytest.raises(ValueError, match="no signal recording sites were found"):
        execute_compute_cross_correlation(str(psth_output_dir), "lever_press", base_input_parameters)


# ---------------------------------------------------------------------------
# _validate_fiber_recording_sites_consistent_for_group
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _validate_psth_window_parameters — upfront step-4 input validation
# ---------------------------------------------------------------------------


@pytest.fixture
def psth_window_inputs():
    """Minimal inputParameters dict for _validate_psth_window_parameters tests.

    Uses production-shaped values: nSecPrev is the negative-side bound and
    baselineCorrectionStart is negative-relative-to-event.
    """
    return {
        "peak_startPoint": [0.0, np.nan],
        "peak_endPoint": [3.0, np.nan],
        "nSecPrev": -10,
        "nSecPost": 20,
        "baselineCorrectionStart": -5,
        "baselineCorrectionEnd": 0,
    }


def test_validate_psth_window_parameters_passes_with_valid_inputs(psth_window_inputs):
    # Should not raise.
    _validate_psth_window_parameters(psth_window_inputs)


def test_validate_psth_window_parameters_skips_when_baseline_correction_off(psth_window_inputs):
    # (0, 0) is the documented sentinel for "skip baseline correction".
    psth_window_inputs["baselineCorrectionStart"] = 0
    psth_window_inputs["baselineCorrectionEnd"] = 0
    _validate_psth_window_parameters(psth_window_inputs)


def test_validate_psth_window_parameters_raises_for_inverted_peak_window(psth_window_inputs):
    psth_window_inputs["peak_startPoint"] = [3.0]
    psth_window_inputs["peak_endPoint"] = [1.0]
    with pytest.raises(ValueError, match=r"Peak End Time is less than or equal to Peak Start Time"):
        _validate_psth_window_parameters(psth_window_inputs)


def test_validate_psth_window_parameters_raises_for_inverted_baseline_window(psth_window_inputs):
    psth_window_inputs["baselineCorrectionStart"] = 0
    psth_window_inputs["baselineCorrectionEnd"] = -5
    with pytest.raises(
        ValueError, match=r"baselineCorrectionStart=0 must be strictly less than baselineCorrectionEnd=-5"
    ):
        _validate_psth_window_parameters(psth_window_inputs)


def test_validate_psth_window_parameters_raises_when_baseline_outside_psth_window(psth_window_inputs):
    # baselineCorrectionEnd=25 exceeds nSecPost=20.
    psth_window_inputs["baselineCorrectionStart"] = -5
    psth_window_inputs["baselineCorrectionEnd"] = 25
    with pytest.raises(ValueError, match=r"PSTH window is \[-10, 20\]s"):
        _validate_psth_window_parameters(psth_window_inputs)


def test_validate_psth_window_parameters_raises_for_unequal_peak_array_lengths(psth_window_inputs):
    psth_window_inputs["peak_startPoint"] = [0.0, 1.0]
    psth_window_inputs["peak_endPoint"] = [2.0]
    with pytest.raises(ValueError, match=r"unequal \(start: 2, end: 1\)"):
        _validate_psth_window_parameters(psth_window_inputs)


# ---------------------------------------------------------------------------
# run_psth_step — chains transient analysis into PSTH in-process
# ---------------------------------------------------------------------------


class TestRunPsthStep:
    @pytest.fixture
    def step_parameters(self, tmp_path, base_input_parameters):
        """Parameters naming one real output directory, with a valid PSTH window."""
        run_folder = tmp_path / "session1" / "session1_output_1"
        run_folder.mkdir(parents=True)
        (run_folder / "storesList.csv").write_text("Dv1A,Dv2A\ncontrol_DMS,signal_DMS\n")
        base_input_parameters["session_folders"] = [str(tmp_path / "session1")]
        base_input_parameters["selected_runs"] = {str(tmp_path / "session1"): ["1"]}
        base_input_parameters["nSecPrev"] = -5.0
        base_input_parameters["peak_startPoint"] = [0.0]
        base_input_parameters["peak_endPoint"] = [5.0]
        return base_input_parameters

    @pytest.fixture
    def recorded_calls(self, monkeypatch):
        """Record the order in which main() invokes the two step-4 workers."""
        calls = []

        def fake_psth_for_each_store(inputParameters):
            calls.append(("psthForEachStore", inputParameters))
            return inputParameters

        def fake_execute_find_freq_and_amp(inputParameters):
            calls.append(("executeFindFreqAndAmp", inputParameters))

        monkeypatch.setattr("guppy.orchestration.psth.psthForEachStore", fake_psth_for_each_store)
        monkeypatch.setattr("guppy.orchestration.psth.executeFindFreqAndAmp", fake_execute_find_freq_and_amp)
        return calls

    def test_runs_transients_then_psth_on_the_same_parameters(self, recorded_calls, step_parameters):
        run_psth_step(step_parameters)

        assert [name for name, _ in recorded_calls] == ["executeFindFreqAndAmp", "psthForEachStore"]
        assert recorded_calls[0][1] is step_parameters
        assert recorded_calls[1][1] is step_parameters

    def test_declares_two_progress_units_per_output_directory(self, recorded_calls, step_parameters):
        """Both halves of step 4 share one bar, so the denominator covers both."""
        step = StepProgress()
        token = _current_step.set(step)

        try:
            run_psth_step(step_parameters)
        finally:
            _current_step.reset(token)

        assert step.total == 2

    def test_invalid_psth_window_is_rejected_before_any_worker_runs(self, recorded_calls, step_parameters):
        step_parameters["baselineCorrectionStart"] = -20.0

        with pytest.raises(ValueError, match="baselineCorrectionStart"):
            run_psth_step(step_parameters)

        assert recorded_calls == []

    def test_transients_failure_is_reported_through_the_progress_channel(
        self, recorded_calls, step_parameters, monkeypatch
    ):
        """A failure in the chained transients step must reach the progress error channel,
        which is how the GUI surfaces it — previously it was lost in a grandchild process."""
        step = StepProgress()
        token = _current_step.set(step)

        def failing_transients(inputParameters):
            raise ValueError("transientsThresh=0 must be positive")

        monkeypatch.setattr("guppy.orchestration.psth.executeFindFreqAndAmp", failing_transients)

        try:
            with pytest.raises(ValueError, match="transientsThresh=0 must be positive"):
                run_psth_step(step_parameters)
        finally:
            _current_step.reset(token)

        assert step.error_message == "transientsThresh=0 must be positive"


class TestValidateEventsOverlapSignal:
    """Step 4 refuses an event store that shares no timeline with the signal."""

    @pytest.fixture
    def step_parameters(self, tmp_path, base_input_parameters):
        """Parameters naming one run folder holding a preprocessed site and one event store."""
        session_folder = tmp_path / "session1"
        run_folder = session_folder / "session1_output_1"
        run_folder.mkdir(parents=True)
        (run_folder / "storesList.csv").write_text("Dv1A,Dv2A,LNRW\ncontrol_dms,signal_dms,port_entries\n")
        # The worker resolves recording sites by globbing the preprocessed metric files.
        write_hdf5(np.zeros(3), "z_score_dms", str(run_folder), "data")
        write_hdf5(np.linspace(0.0, 100.0, 11), "timeCorrection_dms", str(run_folder), "timestampNew")
        base_input_parameters["session_folders"] = [str(session_folder)]
        base_input_parameters["selected_runs"] = {str(session_folder): ["1"]}
        base_input_parameters["nSecPrev"] = -10.0
        base_input_parameters["nSecPost"] = 20.0
        base_input_parameters["peak_startPoint"] = [0.0]
        base_input_parameters["peak_endPoint"] = [5.0]
        base_input_parameters["_run_folder"] = str(run_folder)
        return base_input_parameters

    def test_events_on_another_clock_are_rejected(self, step_parameters):
        # Signal spans [0, 100]s; these events sit ~50000s away, as an unconverted clock would.
        write_hdf5(np.array([49956.0, 50531.0]), "port_entries_dms", step_parameters["_run_folder"], "ts")

        with pytest.raises(ValueError, match=r"no trial overlaps the 'dms' signal"):
            _validate_events_overlap_signal(step_parameters)

    def test_events_inside_the_signal_are_accepted(self, step_parameters):
        write_hdf5(np.array([20.0, 60.0]), "port_entries_dms", step_parameters["_run_folder"], "ts")

        _validate_events_overlap_signal(step_parameters)

    def test_event_window_partly_overlapping_the_signal_is_accepted(self, step_parameters):
        # 105s is past the 100s signal end, but the window's -10s side reaches back to 95s and
        # still catches signal, which is exactly the case rowFormation NaN-pads.
        write_hdf5(np.array([105.0]), "port_entries_dms", step_parameters["_run_folder"], "ts")

        _validate_events_overlap_signal(step_parameters)

    def test_empty_event_store_is_left_to_the_downstream_warning(self, step_parameters):
        write_hdf5(np.array([]), "port_entries_dms", step_parameters["_run_folder"], "ts")

        _validate_events_overlap_signal(step_parameters)

    def test_transient_event_labels_are_skipped(self, step_parameters):
        # Transients are written by executeFindFreqAndAmp, which runs after this check, so their
        # event files do not exist yet on a first run.
        write_hdf5(np.array([20.0]), "port_entries_dms", step_parameters["_run_folder"], "ts")
        step_parameters["useTransientsAsEvents"] = True
        step_parameters["selectForTransientsComputation"] = "z_score"

        _validate_events_overlap_signal(step_parameters)

    def test_off_clock_events_are_rejected_before_any_worker_runs(self, step_parameters, recorded_step4_calls):
        write_hdf5(np.array([49956.0]), "port_entries_dms", step_parameters["_run_folder"], "ts")

        with pytest.raises(ValueError, match=r"no trial overlaps the 'dms' signal"):
            run_psth_step(step_parameters)

        assert recorded_step4_calls == []

    @pytest.fixture
    def recorded_step4_calls(self, monkeypatch):
        """Record any step-4 worker invocation, to prove validation precedes all of them."""
        calls = []
        monkeypatch.setattr(
            "guppy.orchestration.psth.psthForEachStore", lambda inputParameters: calls.append("psthForEachStore")
        )
        monkeypatch.setattr(
            "guppy.orchestration.psth.executeFindFreqAndAmp",
            lambda inputParameters: calls.append("executeFindFreqAndAmp"),
        )
        return calls

    @pytest.mark.parametrize("select_for_compute_psth", ["z_score", "dff", "Both"])
    def test_every_metric_selection_resolves_the_same_recording_site(self, step_parameters, select_for_compute_psth):
        """Recording sites are read off whichever preprocessed metric step 4 will compute over."""
        write_hdf5(np.zeros(3), "dff_dms", step_parameters["_run_folder"], "data")
        write_hdf5(np.array([49956.0]), "port_entries_dms", step_parameters["_run_folder"], "ts")
        step_parameters["selectForComputePsth"] = select_for_compute_psth

        with pytest.raises(ValueError, match=r"no trial overlaps the 'dms' signal"):
            _validate_events_overlap_signal(step_parameters)

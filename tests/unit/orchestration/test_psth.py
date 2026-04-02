import h5py
import numpy as np
import pytest

from guppy.orchestration.psth import (
    execute_compute_cross_correlation,
    execute_compute_psth,
    execute_compute_psth_peak_and_area,
)


@pytest.fixture
def psth_output_dir(tmp_path):
    """Create a minimal output directory with a z_score_DMS.hdf5 for glob discovery."""
    output_dir = tmp_path / "session1" / "session1_output_1"
    output_dir.mkdir(parents=True)
    with h5py.File(str(output_dir / "z_score_DMS.hdf5"), "w") as hdf5_file:
        hdf5_file.create_dataset("data", data=np.zeros(100))
    return output_dir


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


def test_execute_compute_cross_correlation_raises_when_concatenate_and_remove_artifacts(
    psth_output_dir, base_input_parameters
):
    base_input_parameters["computeCorr"] = True
    base_input_parameters["removeArtifacts"] = True
    base_input_parameters["artifactsRemovalMethod"] = "concatenate"

    with pytest.raises(Exception):
        execute_compute_cross_correlation(str(psth_output_dir), "lever_press", base_input_parameters)


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
    base_input_parameters["removeArtifacts"] = False
    execute_compute_cross_correlation(str(psth_output_dir), "control_DMS", base_input_parameters)

    assert len(read_df_calls) == 0

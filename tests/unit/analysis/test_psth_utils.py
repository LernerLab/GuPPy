import numpy as np
import pandas as pd

from guppy.analysis.psth_utils import (
    create_Df_for_cross_correlation,
    create_Df_for_psth,
    getCorrCombinations,
)

# ── create_Df_for_psth ────────────────────────────────────────────────────────


def test_create_df_for_psth_creates_hdf5_file_with_mean_column(tmp_path):
    # 2 trial rows + 1 timestamps row; 3 timepoints
    # After transpose: shape (3, 3); mean across trials [0,1] at each timepoint
    # timepoint 0: mean([1.0, 2.0]) = 1.5; timepoint 1: mean([3.0, 4.0]) = 3.5
    psth = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0], [0.0, 1.0, 2.0]])
    columns = ["sess1", "sess2", "timestamps"]

    create_Df_for_psth(str(tmp_path), "event_lever", "z_score_dms", psth, columns=columns)

    assert (tmp_path / "event_lever_z_score_dms.h5").exists()
    df = pd.read_hdf(tmp_path / "event_lever_z_score_dms.h5", key="df")
    assert "mean" in df.columns
    assert "err" in df.columns
    np.testing.assert_allclose(df["mean"].iloc[0], 1.5, atol=1e-4)
    np.testing.assert_allclose(df["mean"].iloc[1], 3.5, atol=1e-4)


def test_create_df_for_psth_timestamps_column_excluded_from_mean(tmp_path):
    # Only "sess1" is a trial column; timestamps should not contribute to mean
    psth = np.array([[10.0, 20.0, 30.0], [0.0, 1.0, 2.0]])
    columns = ["sess1", "timestamps"]

    create_Df_for_psth(str(tmp_path), "event_lever", "z_score_dms", psth, columns=columns)

    df = pd.read_hdf(tmp_path / "event_lever_z_score_dms.h5", key="df")
    # mean of a single trial equals the trial itself
    np.testing.assert_allclose(df["mean"].iloc[0], 10.0, atol=1e-4)


# ── create_Df_for_cross_correlation ───────────────────────────────────────────


def test_create_df_for_cross_correlation_creates_hdf5_file_with_mean_column(tmp_path):
    # Same structure as PSTH; 2 sessions + timestamps
    psth = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6], [0.0, 1.0, 2.0]])
    columns = ["sess1", "sess2", "timestamps"]

    create_Df_for_cross_correlation(str(tmp_path), "corr_lever", "z_score_dms_nac", psth, columns=columns)

    assert (tmp_path / "corr_lever_z_score_dms_nac.h5").exists()
    df = pd.read_hdf(tmp_path / "corr_lever_z_score_dms_nac.h5", key="df")
    assert "mean" in df.columns
    # mean at timepoint 0: mean([0.1, 0.2]) = 0.15
    np.testing.assert_allclose(df["mean"].iloc[0], 0.15, atol=1e-4)


# ── getCorrCombinations ───────────────────────────────────────────────────────


def test_get_corr_combinations_two_signals_returns_pair(tmp_path):
    (tmp_path / "z_score_dms.hdf5").touch()
    (tmp_path / "z_score_nac.hdf5").touch()
    input_parameters = {"selectForComputePsth": "z_score"}

    corr_info, signal_type = getCorrCombinations(str(tmp_path), input_parameters)

    # np.unique sorts: ["dms", "nac"]
    assert len(corr_info) == 2
    assert set(corr_info) == {"dms", "nac"}
    assert "z_score" in signal_type


def test_get_corr_combinations_three_signals_returns_circular_triplet(tmp_path):
    (tmp_path / "z_score_dms.hdf5").touch()
    (tmp_path / "z_score_nac.hdf5").touch()
    (tmp_path / "z_score_vms.hdf5").touch()
    input_parameters = {"selectForComputePsth": "z_score"}

    corr_info, _ = getCorrCombinations(str(tmp_path), input_parameters)

    # 3 unique names → corr_info = names + [names[0]]; length 4 and first == last
    assert len(corr_info) == 4
    assert corr_info[0] == corr_info[-1]


def test_get_corr_combinations_one_signal_returns_empty(tmp_path):
    (tmp_path / "z_score_dms.hdf5").touch()
    input_parameters = {"selectForComputePsth": "z_score"}

    corr_info, _ = getCorrCombinations(str(tmp_path), input_parameters)

    assert corr_info == []

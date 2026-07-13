import numpy as np
import pandas as pd

from guppy.analysis.psth_utils import (
    create_Df_for_cross_correlation,
    create_Df_for_psth,
    getCorrCombinations,
    match_trials_by_timestamp,
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


def test_get_corr_combinations_one_signal_returns_single_name(tmp_path):
    (tmp_path / "z_score_dms.hdf5").touch()
    input_parameters = {"selectForComputePsth": "z_score"}

    corr_info, _ = getCorrCombinations(str(tmp_path), input_parameters)

    assert corr_info == ["dms"]


# ── match_trials_by_timestamp ─────────────────────────────────────────────────


def test_match_identical_labels_pairs_all_trials():
    labels = [10.0, 20.0, 30.0]
    indices_a, indices_b, matched = match_trials_by_timestamp(labels, list(labels))
    np.testing.assert_array_equal(indices_a, [0, 1, 2])
    np.testing.assert_array_equal(indices_b, [0, 1, 2])
    assert matched == [10.0, 20.0, 30.0]


def test_match_subsample_jitter_still_pairs():
    # Same events, corrected slightly differently per region (sub-sample jitter).
    labels_a = [10.0, 20.0, 30.0]
    labels_b = [10.002, 20.001, 29.998]
    indices_a, indices_b, matched = match_trials_by_timestamp(labels_a, labels_b)
    np.testing.assert_array_equal(indices_a, [0, 1, 2])
    np.testing.assert_array_equal(indices_b, [0, 1, 2])


def test_match_drops_trial_present_in_only_one_region():
    # Region B is missing the 20.0 event; only 10.0 and 30.0 are shared.
    labels_a = [10.0, 20.0, 30.0]
    labels_b = [10.0, 30.0]
    indices_a, indices_b, matched = match_trials_by_timestamp(labels_a, labels_b)
    np.testing.assert_array_equal(indices_a, [0, 2])
    np.testing.assert_array_equal(indices_b, [0, 1])
    assert matched == [10.0, 30.0]


def test_match_parses_string_labels_from_hdf5():
    # Columns read back from HDF5 arrive as strings.
    indices_a, indices_b, matched = match_trials_by_timestamp(["10.0", "20.0"], ["10.0", "20.0"])
    np.testing.assert_array_equal(indices_a, [0, 1])
    np.testing.assert_array_equal(indices_b, [0, 1])


def test_match_no_shared_trials_returns_empty():
    # Distinct events farther apart than the tolerance are not paired.
    indices_a, indices_b, matched = match_trials_by_timestamp([10.0, 20.0], [100.0, 200.0])
    assert matched == []
    assert indices_a.size == 0
    assert indices_b.size == 0


def test_match_bin_columns_matched_by_exact_label():
    # Numeric trials pair by timestamp; bin-aggregate columns pair by exact label.
    labels_a = [10.0, 20.0, "bin_(0-5)", "bin_(5-10)"]
    labels_b = [10.0, 20.0, "bin_(0-5)"]
    indices_a, indices_b, matched = match_trials_by_timestamp(labels_a, labels_b)
    np.testing.assert_array_equal(indices_a, [0, 1, 2])
    np.testing.assert_array_equal(indices_b, [0, 1, 2])
    assert matched == [10.0, 20.0, "bin_(0-5)"]

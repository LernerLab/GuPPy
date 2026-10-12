import json

import h5py
import numpy as np
import pandas as pd
import pytest

from guppy.analysis.psth_utils import write_psth_table
from guppy.analysis.standard_io import write_peak_and_area_to_csv
from guppy.testing.consistency import (
    _collect_relative_paths,
    _compare_json_values,
    _compare_table,
    _normalize_psth_index,
    _normalize_psth_label,
    compare_output_folders,
)


class TestNormalizationHelpers:
    def test_normalize_psth_label_handles_bare_prefixed_and_non_numeric_labels(self):
        assert _normalize_psth_label("138.238440990448") == "138.238441"
        assert _normalize_psth_label("sample_data_csv_1_409.86189556121826") == "sample_data_csv_1_409.8618956"
        assert _normalize_psth_label("not_a_timestamp") == "not_a_timestamp"

    def test_normalize_psth_index_normalizes_each_label(self):
        index = pd.Index(["1.2000000000000002", "session_2.3000000000000003"])

        result = _normalize_psth_index(index)

        assert list(result) == ["1.2", "session_2.3"]

    def test_normalize_psth_label_applies_bare_and_prefixed_offsets(self):
        # Bare event-ts label shifts by bare_offset.
        assert _normalize_psth_label("138.238440990448", bare_offset=1.0) == "139.238441"
        # peak_AUC prefixed label shifts its float suffix by prefixed_offset.
        assert (
            _normalize_psth_label("sample_data_csv_1_138.238440990448", prefixed_offset=1.0)
            == "sample_data_csv_1_139.238441"
        )

    def test_normalize_psth_label_leaves_session_suffix_unchanged_when_prefixed_offset_zero(self):
        # Session-run suffixes (e.g. ..._output_1) match the prefixed-float regex but must NOT
        # be shifted: only peak_AUC files pass a nonzero prefixed_offset.
        assert (
            _normalize_psth_label("Photo_048_392-200728-121222_output_1", bare_offset=1.0, prefixed_offset=0.0)
            == "Photo_048_392-200728-121222_output_1"
        )


class TestPathCollection:
    def test_collect_relative_paths_skips_saved_plots_directory(self, tmp_path):
        expected_file = tmp_path / "keep.json"
        expected_file.write_text("{}")

        skipped_directory = tmp_path / "saved_plots"
        skipped_directory.mkdir()
        (skipped_directory / "plot.png").write_text("fake image")

        result = _collect_relative_paths(str(tmp_path))

        assert sorted(result) == ["keep.json"]


class TestJsonValueComparison:
    def test_compare_json_values_treats_none_and_nan_as_equal(self):
        mismatches: list[str] = []

        _compare_json_values(
            {"a": None, "b": [np.nan, 2.0]},
            {"a": np.nan, "b": [None, 2.0]},
            "params.json",
            "",
            mismatches,
        )

        assert mismatches == []

    def test_compare_json_values_reports_missing_key_and_scalar_mismatch(self):
        mismatches: list[str] = []

        _compare_json_values(
            {"a": 1, "b": 3},
            {"a": 2, "missing": 4},
            "params.json",
            "",
            mismatches,
        )

        assert "params.json @ 'a': expected 2, got 1" in mismatches
        assert "params.json: missing key 'missing' in actual" in mismatches


class TestCompareOutputFolders:
    def test_compare_output_folders_ignores_extra_files_in_actual(self, tmp_path):
        expected_directory = tmp_path / "expected"
        actual_directory = tmp_path / "actual"
        expected_directory.mkdir()
        actual_directory.mkdir()

        expected_data = pd.DataFrame({"value": [1.0, 2.0]})
        actual_data = pd.DataFrame({"value": [1.0, 2.0]})
        expected_data.to_csv(expected_directory / "data.csv")
        actual_data.to_csv(actual_directory / "data.csv")
        (actual_directory / "extra.csv").write_text("index,value\n0,99\n")

        compare_output_folders(actual_dir=str(actual_directory), expected_dir=str(expected_directory))

    def test_compare_output_folders_accumulates_missing_and_content_mismatches(self, tmp_path):
        expected_directory = tmp_path / "expected"
        actual_directory = tmp_path / "actual"
        expected_directory.mkdir()
        actual_directory.mkdir()

        pd.DataFrame({"value": [1.0]}).to_csv(expected_directory / "missing.csv")

        pd.DataFrame({"value": [1.0]}).to_csv(expected_directory / "different.csv")
        pd.DataFrame({"value": [2.0]}).to_csv(actual_directory / "different.csv")

        with pytest.raises(AssertionError, match="Output folder comparison failed") as raised_error:
            compare_output_folders(
                actual_dir=str(actual_directory), expected_dir=str(expected_directory), rtol=1e-12, atol=0.0
            )

        message = str(raised_error.value)
        assert "MISSING in actual: missing.csv" in message
        assert "different.csv: CSV content differs" in message

    def test_compare_output_folders_event_ts_offset_reconciles_ts_dataset(self, tmp_path):
        # Event-timestamp arrays (key "ts") moved to the recording-start basis:
        # actual == reference + timeForLightsTurnOn.
        expected_directory = tmp_path / "expected"
        actual_directory = tmp_path / "actual"
        expected_directory.mkdir()
        actual_directory.mkdir()

        with h5py.File(expected_directory / "ttl_region.hdf5", "w") as f:
            f.create_dataset("ts", data=np.array([138.0, 189.0, 269.0]))
        with h5py.File(actual_directory / "ttl_region.hdf5", "w") as f:
            f.create_dataset("ts", data=np.array([139.0, 190.0, 270.0]))

        # With the matching offset the shifted reference equals actual.
        compare_output_folders(
            actual_dir=str(actual_directory), expected_dir=str(expected_directory), event_ts_offset=1.0
        )
        # Without the offset the basis difference is (correctly) flagged.
        with pytest.raises(AssertionError, match="'ts' numeric data differs"):
            compare_output_folders(actual_dir=str(actual_directory), expected_dir=str(expected_directory))

    def test_compare_output_folders_compares_h5_reference_against_csv_output(self, tmp_path):
        # The reference PSTH is a PyTables .h5 table; the current output is the CSV of the same stem.
        expected_directory = tmp_path / "expected"
        actual_directory = tmp_path / "actual"
        expected_directory.mkdir()
        actual_directory.mkdir()

        pd.DataFrame(
            {"138.238440990448": [0.5, 1.5], "timestamps": [-1.0, 1.0], "mean": [0.5, 1.5], "err": [0.0, 0.0]}
        ).to_hdf(expected_directory / "ttl_region_z_score_region.h5", key="df", mode="w")
        psth = pd.DataFrame(
            {"139.238440990448": [0.5, 1.5], "timestamps": [-1.0, 1.0], "mean": [0.5, 1.5], "err": [0.0, 0.0]}
        )
        write_psth_table(filepath=actual_directory, stem="ttl_region_z_score_region", psth=psth)

        compare_output_folders(
            actual_dir=str(actual_directory), expected_dir=str(expected_directory), event_ts_offset=1.0
        )
        # Without the offset the reference trial label is (correctly) flagged.
        with pytest.raises(AssertionError, match="ttl_region_z_score_region.h5: table content differs"):
            compare_output_folders(actual_dir=str(actual_directory), expected_dir=str(expected_directory))

    def test_compare_output_folders_event_ts_offset_does_not_shift_group_session_columns(self, tmp_path):
        # Group-average trial columns are session-run names (..._output_1) that match the
        # prefixed-float regex. They must compare equal even when event_ts_offset is set,
        # because only bare-float PSTH column labels are event times.
        expected_directory = tmp_path / "expected"
        actual_directory = tmp_path / "actual"
        expected_directory.mkdir()
        actual_directory.mkdir()

        psth = pd.DataFrame(
            {
                "Photo_048_output_1": [0.5, 1.5],
                "Photo_63_output_1": [1.0, 2.0],
                "timestamps": [-1.0, 1.0],
                "mean": [0.75, 1.75],
                "err": [0.25, 0.25],
            }
        )
        psth.to_hdf(expected_directory / "rewarded_z_score_region.h5", key="df", mode="w")
        write_psth_table(filepath=actual_directory, stem="rewarded_z_score_region", psth=psth)

        compare_output_folders(
            actual_dir=str(actual_directory), expected_dir=str(expected_directory), event_ts_offset=1.0
        )

    def test_compare_output_folders_continuous_ts_offset_reconciles_timestamp_datasets(self, tmp_path):
        # An extractor that moved onto the acquisition clock shifts the continuous
        # timestamps ("timestamps" in per-store files, "timestampNew" in timeCorrection)
        # by the recording start: actual == reference + recordingStart.
        expected_directory = tmp_path / "expected"
        actual_directory = tmp_path / "actual"
        expected_directory.mkdir()
        actual_directory.mkdir()

        with h5py.File(expected_directory / "file0_chev1.hdf5", "w") as f:
            f.create_dataset("timestamps", data=np.array([0.0, 0.05, 0.1]))
        with h5py.File(actual_directory / "file0_chev1.hdf5", "w") as f:
            f.create_dataset("timestamps", data=np.array([100.0, 100.05, 100.1]))

        with h5py.File(expected_directory / "timeCorrection_region.hdf5", "w") as f:
            f.create_dataset("timestampNew", data=np.array([1.0, 1.05]))
        with h5py.File(actual_directory / "timeCorrection_region.hdf5", "w") as f:
            f.create_dataset("timestampNew", data=np.array([101.0, 101.05]))

        compare_output_folders(
            actual_dir=str(actual_directory), expected_dir=str(expected_directory), continuous_ts_offset=100.0
        )
        with pytest.raises(AssertionError, match="numeric data differs"):
            compare_output_folders(actual_dir=str(actual_directory), expected_dir=str(expected_directory))

    def test_compare_output_folders_continuous_ts_offset_does_not_shift_signal_data(self, tmp_path):
        # A clock shift leaves signal values untouched, so "data" must still compare exactly
        # even when continuous_ts_offset is set.
        expected_directory = tmp_path / "expected"
        actual_directory = tmp_path / "actual"
        expected_directory.mkdir()
        actual_directory.mkdir()

        with h5py.File(expected_directory / "z_score_region.hdf5", "w") as f:
            f.create_dataset("data", data=np.array([0.5, 1.5]))
        with h5py.File(actual_directory / "z_score_region.hdf5", "w") as f:
            f.create_dataset("data", data=np.array([0.5, 1.5]))

        compare_output_folders(
            actual_dir=str(actual_directory), expected_dir=str(expected_directory), continuous_ts_offset=100.0
        )

    def test_compare_output_folders_continuous_ts_offset_shifts_transients_csv_column(self, tmp_path):
        # transientsOccurrences CSVs carry a "timestamps" column on the continuous clock.
        expected_directory = tmp_path / "expected"
        actual_directory = tmp_path / "actual"
        expected_directory.mkdir()
        actual_directory.mkdir()

        pd.DataFrame({"timestamps": [23.48, 41.02], "amplitude": [1.5, 2.5]}).to_csv(
            expected_directory / "transientsOccurrences_z_score_region.csv"
        )
        pd.DataFrame({"timestamps": [123.48, 141.02], "amplitude": [1.5, 2.5]}).to_csv(
            actual_directory / "transientsOccurrences_z_score_region.csv"
        )

        compare_output_folders(
            actual_dir=str(actual_directory), expected_dir=str(expected_directory), continuous_ts_offset=100.0
        )
        with pytest.raises(AssertionError, match="CSV content differs"):
            compare_output_folders(actual_dir=str(actual_directory), expected_dir=str(expected_directory))

    def test_compare_output_folders_reports_hdf5_string_mismatch_for_non_psth_file(self, tmp_path):
        expected_directory = tmp_path / "expected"
        actual_directory = tmp_path / "actual"
        expected_directory.mkdir()
        actual_directory.mkdir()

        expected_path = expected_directory / "plain_data.hdf5"
        actual_path = actual_directory / "plain_data.hdf5"

        with h5py.File(expected_path, "w") as expected_file:
            expected_file.create_dataset("labels", data=np.array([b"A", b"B"]))

        with h5py.File(actual_path, "w") as actual_file:
            actual_file.create_dataset("labels", data=np.array([b"A", b"C"]))

        with pytest.raises(AssertionError, match="Output folder comparison failed") as raised_error:
            compare_output_folders(actual_dir=str(actual_directory), expected_dir=str(expected_directory))

        assert "plain_data.hdf5: 'labels' string data differs" in str(raised_error.value)

    def test_compare_output_folders_name_map_compares_renamed_file_by_data(self, tmp_path):
        # A reference file mapped to a differently-named actual file passes when the data matches.
        expected_directory = tmp_path / "expected"
        actual_directory = tmp_path / "actual"
        expected_directory.mkdir()
        actual_directory.mkdir()

        pd.DataFrame({"value": [1.0, 2.0]}).to_csv(expected_directory / "PAB_0_region.csv", index=False)
        pd.DataFrame({"value": [1.0, 2.0]}).to_csv(actual_directory / "reward_region.csv", index=False)

        compare_output_folders(
            actual_dir=str(actual_directory),
            expected_dir=str(expected_directory),
            name_map={"PAB_0_region.csv": "reward_region.csv"},
        )

    def test_compare_output_folders_name_map_renamed_file_reports_data_mismatch(self, tmp_path):
        expected_directory = tmp_path / "expected"
        actual_directory = tmp_path / "actual"
        expected_directory.mkdir()
        actual_directory.mkdir()

        pd.DataFrame({"value": [1.0]}).to_csv(expected_directory / "PAB_0_region.csv", index=False)
        pd.DataFrame({"value": [2.0]}).to_csv(actual_directory / "reward_region.csv", index=False)

        with pytest.raises(AssertionError, match="Output folder comparison failed") as raised_error:
            compare_output_folders(
                actual_dir=str(actual_directory),
                expected_dir=str(expected_directory),
                name_map={"PAB_0_region.csv": "reward_region.csv"},
                rtol=1e-12,
                atol=0.0,
            )
        assert "PAB_0_region.csv: CSV content differs" in str(raised_error.value)

    def test_compare_output_folders_name_map_none_allows_intentionally_absent_file(self, tmp_path):
        # A reference file mapped to None is intentionally not produced; its absence is required, not a failure.
        expected_directory = tmp_path / "expected"
        actual_directory = tmp_path / "actual"
        expected_directory.mkdir()
        actual_directory.mkdir()

        pd.DataFrame({"value": [1.0]}).to_csv(expected_directory / "PAB_.csv", index=False)

        compare_output_folders(
            actual_dir=str(actual_directory),
            expected_dir=str(expected_directory),
            name_map={"PAB_.csv": None},
        )

    def test_compare_output_folders_name_map_none_fails_when_file_unexpectedly_present(self, tmp_path):
        expected_directory = tmp_path / "expected"
        actual_directory = tmp_path / "actual"
        expected_directory.mkdir()
        actual_directory.mkdir()

        pd.DataFrame({"value": [1.0]}).to_csv(expected_directory / "PAB_.csv", index=False)
        pd.DataFrame({"value": [1.0]}).to_csv(actual_directory / "PAB_.csv", index=False)

        with pytest.raises(AssertionError, match="Output folder comparison failed") as raised_error:
            compare_output_folders(
                actual_dir=str(actual_directory),
                expected_dir=str(expected_directory),
                name_map={"PAB_.csv": None},
            )
        assert "UNEXPECTEDLY PRESENT (mapped to None): PAB_.csv" in str(raised_error.value)

    def test_compare_output_folders_label_map_reconciles_renamed_csv_index(self, tmp_path):
        # A group labels its members by a path below the output root; the reference was
        # written when the run folder carried its session in its own name.
        expected_directory = tmp_path / "expected"
        actual_directory = tmp_path / "actual"
        expected_directory.mkdir()
        actual_directory.mkdir()

        pd.DataFrame({"amplitude": [1.5]}, index=["session_a_output_1"]).to_csv(
            expected_directory / "freqAndAmp_z_score_region.csv"
        )
        pd.DataFrame({"amplitude": [1.5]}, index=["session_a/output_1"]).to_csv(
            actual_directory / "freqAndAmp_z_score_region.csv"
        )

        compare_output_folders(
            actual_dir=str(actual_directory),
            expected_dir=str(expected_directory),
            label_map={"session_a_output_1": "session_a/output_1"},
        )

    def test_compare_output_folders_label_map_reconciles_renamed_hdf5_labels(self, tmp_path):
        expected_directory = tmp_path / "expected"
        actual_directory = tmp_path / "actual"
        expected_directory.mkdir()
        actual_directory.mkdir()

        with h5py.File(expected_directory / "plain_data.hdf5", "w") as expected_file:
            expected_file.create_dataset("labels", data=np.array([b"session_a_output_1", b"mean"]))
        with h5py.File(actual_directory / "plain_data.hdf5", "w") as actual_file:
            actual_file.create_dataset("labels", data=np.array([b"session_a/output_1", b"mean"]))

        compare_output_folders(
            actual_dir=str(actual_directory),
            expected_dir=str(expected_directory),
            label_map={"session_a_output_1": "session_a/output_1"},
        )

    def test_compare_output_folders_label_map_still_reports_unmapped_label_mismatch(self, tmp_path):
        expected_directory = tmp_path / "expected"
        actual_directory = tmp_path / "actual"
        expected_directory.mkdir()
        actual_directory.mkdir()

        with h5py.File(expected_directory / "plain_data.hdf5", "w") as expected_file:
            expected_file.create_dataset("labels", data=np.array([b"session_a_output_1", b"mean"]))
        with h5py.File(actual_directory / "plain_data.hdf5", "w") as actual_file:
            actual_file.create_dataset("labels", data=np.array([b"session_a/output_1", b"median"]))

        with pytest.raises(AssertionError, match="Output folder comparison failed") as raised_error:
            compare_output_folders(
                actual_dir=str(actual_directory),
                expected_dir=str(expected_directory),
                label_map={"session_a_output_1": "session_a/output_1"},
            )
        assert "plain_data.hdf5: 'labels' string data differs" in str(raised_error.value)


class TestCompareTable:
    def test_psth_reference_matches_csv_output_with_bins_and_shifted_labels(self, tmp_path):
        # Reference columns are in the legacy in-memory order (trials, bins, timestamps, mean,
        # err) and its bare-float trial labels sit 1.0 s before the current ones.
        expected_path = tmp_path / "ttl_region_z_score_region.h5"
        pd.DataFrame(
            {
                "138.238440990448": [0.5, 1.5],
                "189.68911623954773": [1.5, 2.5],
                "bin_(0-1)": [1.0, 2.0],
                "bin_err_(0-1)": [0.5, 0.5],
                "timestamps": [-1.0, 1.0],
                "mean": [1.0, 2.0],
                "err": [0.5, 0.5],
            }
        ).to_hdf(expected_path, key="df", mode="w")
        psth = pd.DataFrame(
            {
                "139.238440990448": [0.5, 1.5],
                "190.68911623954773": [1.5, 2.5],
                "bin_(0-1)": [1.0, 2.0],
                "bin_err_(0-1)": [0.5, 0.5],
                "timestamps": [-1.0, 1.0],
                "mean": [1.0, 2.0],
                "err": [0.5, 0.5],
            }
        )
        write_psth_table(filepath=tmp_path, stem="ttl_region_z_score_region", psth=psth)
        mismatches = []

        _compare_table(
            tmp_path / "ttl_region_z_score_region.csv",
            expected_path,
            "ttl_region_z_score_region.h5",
            mismatches,
            1e-5,
            1e-8,
            event_ts_offset=1.0,
        )

        assert mismatches == []

    @pytest.mark.parametrize("reference_suffix", [".h5", ".csv"])
    def test_peak_auc_reference_matches_trials_and_bins_csvs(self, tmp_path, reference_suffix):
        # The reference is one table indexed "<session>_<trial>", ordered per session as its
        # trials, then its bins, then its mean, with event times 1.0 s before the current ones.
        expected_directory = tmp_path / "expected"
        actual_directory = tmp_path / "actual"
        expected_directory.mkdir()
        actual_directory.mkdir()
        expected = pd.DataFrame(
            {"peak_pos_1": [1.0, 2.0, 1.5, 1.5, 3.0, 2.5, 3.0], "area_1": [0.5, 1.0, 0.75, 0.75, 1.5, 1.25, 1.5]},
            index=[
                "session_a_138.238440990448",
                "session_a_189.68911623954773",
                "session_a_bin_(0-1)",
                "session_a_mean",
                "session_b_12.5",
                "session_b_bin_(0-1)",
                "session_b_mean",
            ],
        )
        expected_path = expected_directory / ("peak_AUC_ttl_region_z_score_region" + reference_suffix)
        if reference_suffix == ".h5":
            expected.to_hdf(expected_path, key="df", mode="w")
        else:
            expected.to_csv(expected_path)
        trials = pd.DataFrame(
            {
                "session": ["session_a", "session_a", "session_a", "session_b", "session_b"],
                "trial": ["139.238440990448", "190.68911623954773", "mean", "13.5", "mean"],
                "peak_pos_1": [1.0, 2.0, 1.5, 3.0, 3.0],
                "area_1": [0.5, 1.0, 0.75, 1.5, 1.5],
            }
        )
        bins = pd.DataFrame(
            {
                "session": ["session_a", "session_b"],
                "bin": ["bin_(0-1)", "bin_(0-1)"],
                "peak_pos_1": [1.5, 2.5],
                "area_1": [0.75, 1.25],
            }
        )
        write_peak_and_area_to_csv(
            filepath=actual_directory, trials=trials, bins=bins, name="ttl_region_z_score_region"
        )
        mismatches = []

        _compare_table(
            actual_directory / "peak_AUC_ttl_region_z_score_region.csv",
            expected_path,
            expected_path.name,
            mismatches,
            1e-5,
            1e-8,
            event_ts_offset=1.0,
        )

        assert mismatches == []

    def test_reports_a_numeric_mismatch(self, tmp_path):
        expected_path = tmp_path / "ttl_region_z_score_region.h5"
        pd.DataFrame({"12.5": [0.5, 1.5], "timestamps": [-1.0, 1.0], "mean": [0.5, 1.5], "err": [0.0, 0.0]}).to_hdf(
            expected_path, key="df", mode="w"
        )
        psth = pd.DataFrame({"12.5": [0.5, 1.5], "timestamps": [-1.0, 1.0], "mean": [0.5, 2.5], "err": [0.0, 0.0]})
        write_psth_table(filepath=tmp_path, stem="ttl_region_z_score_region", psth=psth)
        mismatches = []

        _compare_table(
            tmp_path / "ttl_region_z_score_region.csv",
            expected_path,
            "ttl_region_z_score_region.h5",
            mismatches,
            1e-5,
            1e-8,
        )

        assert len(mismatches) == 1
        assert mismatches[0].startswith("ttl_region_z_score_region.h5: table content differs")
        assert '"mean"' in mismatches[0]


class TestCompareJsonFilePath:
    def test_compare_output_folders_reports_json_value_differences(self, tmp_path):
        expected_directory = tmp_path / "expected"
        actual_directory = tmp_path / "actual"
        expected_directory.mkdir()
        actual_directory.mkdir()

        expected_json_path = expected_directory / "parameters.json"
        actual_json_path = actual_directory / "parameters.json"

        expected_json_path.write_text(json.dumps({"a": 1, "nested": {"b": [1, 2]}}))
        actual_json_path.write_text(json.dumps({"a": 2, "nested": {"b": [1, 3]}}))

        with pytest.raises(AssertionError, match="Output folder comparison failed") as raised_error:
            compare_output_folders(actual_dir=str(actual_directory), expected_dir=str(expected_directory))

        message = str(raised_error.value)
        assert "parameters.json @ 'a': expected 1, got 2" in message
        assert "parameters.json @ 'nested.b[1]': expected 2, got 3" in message

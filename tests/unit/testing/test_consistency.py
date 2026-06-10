import json

import h5py
import numpy as np
import pandas as pd
import pytest

from guppy.testing.consistency import (
    _collect_relative_paths,
    _compare_json_values,
    _normalize_psth_index,
    _normalize_psth_label,
    _normalize_psth_str_array,
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

    def test_normalize_psth_str_array_normalizes_bytes_and_preserves_shape(self):
        values = np.array([[b"1.2000000000000002", b"session_2.3000000000000003"]], dtype=object)

        result = _normalize_psth_str_array(values)

        assert result.shape == (1, 2)
        assert result.tolist() == [["1.2", "session_2.3"]]

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

    def test_compare_output_folders_normalizes_psth_axis_labels_for_z_score_hdf5(self, tmp_path):
        expected_directory = tmp_path / "expected"
        actual_directory = tmp_path / "actual"
        expected_directory.mkdir()
        actual_directory.mkdir()

        expected_path = expected_directory / "z_score_region.h5"
        actual_path = actual_directory / "z_score_region.h5"

        with h5py.File(expected_path, "w") as expected_file:
            expected_file.create_dataset("axis0", data=np.array([b"1.2000000000000002", b"session_2.3000000000000003"]))
            expected_file.create_dataset("values", data=np.array([1.0, 2.0]))

        with h5py.File(actual_path, "w") as actual_file:
            actual_file.create_dataset("axis0", data=np.array([b"1.2", b"session_2.3"]))
            actual_file.create_dataset("values", data=np.array([1.0, 2.0]))

        compare_output_folders(actual_dir=str(actual_directory), expected_dir=str(expected_directory))

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

    def test_compare_output_folders_event_ts_offset_shifts_psth_bare_labels(self, tmp_path):
        # z_score trial columns are bare event-ts floats; the reference is shifted into the
        # current basis by event_ts_offset.
        expected_directory = tmp_path / "expected"
        actual_directory = tmp_path / "actual"
        expected_directory.mkdir()
        actual_directory.mkdir()

        with h5py.File(expected_directory / "z_score_region.h5", "w") as f:
            f.create_dataset("axis0", data=np.array([b"138.238440990448", b"189.68911623954773"]))
        with h5py.File(actual_directory / "z_score_region.h5", "w") as f:
            f.create_dataset("axis0", data=np.array([b"139.238440990448", b"190.68911623954773"]))

        compare_output_folders(
            actual_dir=str(actual_directory), expected_dir=str(expected_directory), event_ts_offset=1.0
        )

    def test_compare_output_folders_event_ts_offset_does_not_shift_group_session_columns(self, tmp_path):
        # Group-average trial columns are session-run names (..._output_1) that match the
        # prefixed-float regex. They must compare equal even when event_ts_offset is set,
        # because a z_score (non-peak_AUC) file uses prefixed_offset=0.
        expected_directory = tmp_path / "expected"
        actual_directory = tmp_path / "actual"
        expected_directory.mkdir()
        actual_directory.mkdir()

        cols = np.array([b"Photo_048_output_1", b"Photo_63_output_1"])
        with h5py.File(expected_directory / "rewarded_z_score_region.h5", "w") as f:
            f.create_dataset("axis0", data=cols)
        with h5py.File(actual_directory / "rewarded_z_score_region.h5", "w") as f:
            f.create_dataset("axis0", data=cols)

        compare_output_folders(
            actual_dir=str(actual_directory), expected_dir=str(expected_directory), event_ts_offset=1.0
        )

    def test_compare_output_folders_event_ts_offset_shifts_peak_auc_hdf5_index_not_columns(self, tmp_path):
        # peak_AUC HDF5 stores event-ts labels in the index axis (axis1) and structural
        # column names (peak_pos_1, area_1) in axis0/block0_items. Only axis1 must shift;
        # the "_1" column suffixes match the prefixed-float regex but must NOT shift.
        expected_directory = tmp_path / "expected"
        actual_directory = tmp_path / "actual"
        expected_directory.mkdir()
        actual_directory.mkdir()

        columns = np.array([b"peak_pos_1", b"area_1"])
        with h5py.File(expected_directory / "peak_AUC_ttl_region.h5", "w") as f:
            f.create_dataset("axis0", data=columns)
            f.create_dataset("block0_items", data=columns)
            f.create_dataset("axis1", data=np.array([b"sample_1_138.238440990448"]))
        with h5py.File(actual_directory / "peak_AUC_ttl_region.h5", "w") as f:
            f.create_dataset("axis0", data=columns)
            f.create_dataset("block0_items", data=columns)
            f.create_dataset("axis1", data=np.array([b"sample_1_139.238440990448"]))

        compare_output_folders(
            actual_dir=str(actual_directory), expected_dir=str(expected_directory), event_ts_offset=1.0
        )

    def test_compare_output_folders_event_ts_offset_shifts_peak_auc_csv_index(self, tmp_path):
        # peak_AUC row index is prefixed floats whose suffix IS the event ts → shift the suffix.
        expected_directory = tmp_path / "expected"
        actual_directory = tmp_path / "actual"
        expected_directory.mkdir()
        actual_directory.mkdir()

        pd.DataFrame({"area": [0.5]}, index=["sample_data_csv_1_138.238440990448"]).to_csv(
            expected_directory / "peak_AUC_ttl_region.csv"
        )
        pd.DataFrame({"area": [0.5]}, index=["sample_data_csv_1_139.238440990448"]).to_csv(
            actual_directory / "peak_AUC_ttl_region.csv"
        )

        compare_output_folders(
            actual_dir=str(actual_directory), expected_dir=str(expected_directory), event_ts_offset=1.0
        )

    def test_compare_output_folders_reports_hdf5_string_mismatch_for_non_psth_file(self, tmp_path):
        expected_directory = tmp_path / "expected"
        actual_directory = tmp_path / "actual"
        expected_directory.mkdir()
        actual_directory.mkdir()

        expected_path = expected_directory / "plain_data.h5"
        actual_path = actual_directory / "plain_data.h5"

        with h5py.File(expected_path, "w") as expected_file:
            expected_file.create_dataset("labels", data=np.array([b"A", b"B"]))

        with h5py.File(actual_path, "w") as actual_file:
            actual_file.create_dataset("labels", data=np.array([b"A", b"C"]))

        with pytest.raises(AssertionError, match="Output folder comparison failed") as raised_error:
            compare_output_folders(actual_dir=str(actual_directory), expected_dir=str(expected_directory))

        assert "plain_data.h5: 'labels' string data differs" in str(raised_error.value)

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

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
    compare_npm_session_files,
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


class TestCompareNpmSessionFiles:
    def test_compare_npm_session_files_compares_only_top_level_csv_files(self, tmp_path):
        expected_session_directory = tmp_path / "expected_session"
        actual_session_directory = tmp_path / "actual_session"
        expected_session_directory.mkdir()
        actual_session_directory.mkdir()

        pd.DataFrame({"value": [1.0, 2.0]}).to_csv(expected_session_directory / "event0.csv")
        pd.DataFrame({"value": [1.0, 2.0]}).to_csv(actual_session_directory / "event0.csv")

        nested_expected_directory = expected_session_directory / "nested"
        nested_actual_directory = actual_session_directory / "nested"
        nested_expected_directory.mkdir()
        nested_actual_directory.mkdir()

        pd.DataFrame({"value": [100.0]}).to_csv(nested_expected_directory / "ignored.csv")
        pd.DataFrame({"value": [999.0]}).to_csv(nested_actual_directory / "ignored.csv")

        compare_npm_session_files(
            actual_session_dir=str(actual_session_directory),
            expected_session_dir=str(expected_session_directory),
        )

    def test_compare_npm_session_files_reports_missing_expected_csv(self, tmp_path):
        expected_session_directory = tmp_path / "expected_session"
        actual_session_directory = tmp_path / "actual_session"
        expected_session_directory.mkdir()
        actual_session_directory.mkdir()

        pd.DataFrame({"value": [1.0]}).to_csv(expected_session_directory / "event0.csv")

        with pytest.raises(AssertionError, match="NPM session file comparison failed") as raised_error:
            compare_npm_session_files(
                actual_session_dir=str(actual_session_directory),
                expected_session_dir=str(expected_session_directory),
            )

        assert "MISSING in actual session dir: event0.csv" in str(raised_error.value)


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

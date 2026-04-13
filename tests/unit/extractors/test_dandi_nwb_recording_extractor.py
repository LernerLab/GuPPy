"""Tests for DandiNwbRecordingExtractor and its helper functions."""

from guppy.extractors.dandi_nwb_recording_extractor import (
    DANDI_URI_PREFIX,
    is_dandi_uri,
    parse_dandi_uri,
)


class TestIsDandiUri:
    def test_valid_uri(self):
        assert is_dandi_uri("dandi://000971/sub-01/file.nwb") is True

    def test_local_path(self):
        assert is_dandi_uri("/home/user/data/session1") is False

    def test_empty_string(self):
        assert is_dandi_uri("") is False

    def test_none(self):
        assert is_dandi_uri(None) is False

    def test_prefix_only(self):
        assert is_dandi_uri("dandi://") is True

    def test_non_string(self):
        assert is_dandi_uri(12345) is False


class TestParseDandiUri:
    def test_simple_uri(self):
        dandiset_id, asset_path = parse_dandi_uri("dandi://000971/sub-01/file.nwb")
        assert dandiset_id == "000971"
        assert asset_path == "sub-01/file.nwb"

    def test_nested_asset_path(self):
        dandiset_id, asset_path = parse_dandi_uri(
            "dandi://000971/sub-112-283/sub-112-283_ses-FP-PS-2019-06-20T09-32-04_behavior.nwb"
        )
        assert dandiset_id == "000971"
        assert asset_path == "sub-112-283/sub-112-283_ses-FP-PS-2019-06-20T09-32-04_behavior.nwb"

    def test_prefix_constant(self):
        assert DANDI_URI_PREFIX == "dandi://"

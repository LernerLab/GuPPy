"""Unit tests for the pure-Python validation helpers in npm_gui_prompts."""

import pytest

from guppy.frontend.npm_gui_prompts import _validate_timestamp_configuration


class TestValidateTimestampConfiguration:
    def test_returns_none_when_both_fields_populated(self):
        result = _validate_timestamp_configuration(timestamp_column_name="Timestamp", time_unit="seconds")
        assert result is None

    def test_raises_when_timestamp_column_blank(self):
        with pytest.raises(ValueError, match="'Select which timestamps to use'"):
            _validate_timestamp_configuration(timestamp_column_name="", time_unit="seconds")

    def test_raises_when_time_unit_blank(self):
        with pytest.raises(ValueError, match="'Select timestamps unit'"):
            _validate_timestamp_configuration(timestamp_column_name="Timestamp", time_unit="")

    def test_raises_when_both_blank_and_lists_both_fields(self):
        with pytest.raises(ValueError) as exception_info:
            _validate_timestamp_configuration(timestamp_column_name="", time_unit="")
        message = str(exception_info.value)
        assert "'Select which timestamps to use'" in message
        assert "'Select timestamps unit'" in message

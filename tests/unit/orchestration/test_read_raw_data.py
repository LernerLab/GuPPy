"""Contract tests for orchestrate_read_raw_data error enrichment."""

import shutil

import pytest
from conftest import STUBBED_TESTING_DATA

from guppy.orchestration.read_raw_data import orchestrate_read_raw_data


class TestOrchestrateReadRawDataErrorEnrichment:
    """Missing-event error must list the events the extractor did discover (issue #270)."""

    @pytest.fixture
    def session_with_bogus_event(self, tmp_path):
        """Copy a real stubbed Doric session and add a bogus event to storesList.csv."""
        source_folder = STUBBED_TESTING_DATA / "doric" / "sample_doric_1"
        session_folder = tmp_path / "sample_doric_1"
        shutil.copytree(source_folder, session_folder)

        output_folder = session_folder / "sample_doric_1_output_1"
        output_folder.mkdir()
        stores_list_path = output_folder / "storesList.csv"
        stores_list_path.write_text("NotARealEvent\nsignal_DMS\n")

        return str(session_folder)

    def test_missing_event_error_lists_available_events(self, session_with_bogus_event):
        input_parameters = {
            "folderNames": [session_with_bogus_event],
            "numberOfCores": 1,
            "noChannels": 2,
            "selectedOutputs": {session_with_bogus_event: ["1"]},
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

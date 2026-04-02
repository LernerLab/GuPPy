import shutil

import numpy as np
from conftest import STUBBED_TESTING_DATA

from guppy.extractors import (
    CsvRecordingExtractor,
    DoricRecordingExtractor,
    NpmRecordingExtractor,
    TdtRecordingExtractor,
)
from guppy.orchestration.read_raw_data import _build_event_to_extractor

CSV_SESSION = STUBBED_TESTING_DATA / "csv" / "sample_data_csv_1"
DORIC_SESSION = STUBBED_TESTING_DATA / "doric" / "sample_doric_1"
TDT_SESSION = STUBBED_TESTING_DATA / "tdt" / "Photo_63_207-181030-103332"
NPM_SESSION = STUBBED_TESTING_DATA / "npm" / "sampleData_NPM_1"

CSV_STORENAMES_MAP = {
    "Sample_Control_Channel": "control_region",
    "Sample_Signal_Channel": "signal_region",
    "Sample_TTL": "ttl",
}
DORIC_STORENAMES_MAP = {
    "AIn-1 - Raw": "control_region",
    "AIn-2 - Raw": "signal_region",
    "DI--O-1": "ttl",
}
TDT_STORENAMES_MAP = {
    "Dv1A": "control_dms",
    "Dv2A": "signal_dms",
    "PrtN": "port_entries_dms",
}
NPM_STORENAMES_MAP = {
    "file0_chev1": "signal_region",
    "file0_chod1": "control_region",
}


def _make_stores_list(storenames_map):
    return np.array([list(storenames_map.keys()), list(storenames_map.values())])


def test_doric_session_routes_all_events_to_doric_extractor(tmp_path):
    session_copy = tmp_path / DORIC_SESSION.name
    shutil.copytree(DORIC_SESSION, session_copy)
    stores_list = _make_stores_list(DORIC_STORENAMES_MAP)
    input_parameters = {"noChannels": 2}
    result = _build_event_to_extractor(
        folder_path=str(session_copy),
        storesList=stores_list,
        inputParameters=input_parameters,
    )
    assert result
    for extractor in result.values():
        assert isinstance(extractor, DoricRecordingExtractor)


def test_csv_session_routes_all_events_to_csv_extractor(tmp_path):
    session_copy = tmp_path / CSV_SESSION.name
    shutil.copytree(CSV_SESSION, session_copy)
    stores_list = _make_stores_list(CSV_STORENAMES_MAP)
    input_parameters = {"noChannels": 2}
    result = _build_event_to_extractor(
        folder_path=str(session_copy),
        storesList=stores_list,
        inputParameters=input_parameters,
    )
    assert result
    for extractor in result.values():
        assert isinstance(extractor, CsvRecordingExtractor)


def test_tdt_session_routes_all_events_to_tdt_extractor(tmp_path):
    session_copy = tmp_path / TDT_SESSION.name
    shutil.copytree(TDT_SESSION, session_copy)
    stores_list = _make_stores_list(TDT_STORENAMES_MAP)
    input_parameters = {"noChannels": 2}
    result = _build_event_to_extractor(
        folder_path=str(session_copy),
        storesList=stores_list,
        inputParameters=input_parameters,
    )
    assert result
    for extractor in result.values():
        assert isinstance(extractor, TdtRecordingExtractor)


def test_npm_session_routes_all_events_to_npm_extractor(tmp_path):
    session_copy = tmp_path / NPM_SESSION.name
    shutil.copytree(NPM_SESSION, session_copy)
    stores_list = _make_stores_list(NPM_STORENAMES_MAP)
    input_parameters = {
        "noChannels": 2,
        "npm_split_events": [False, True],
        "npm_timestamp_column_names": None,
        "npm_time_units": None,
    }
    result = _build_event_to_extractor(
        folder_path=str(session_copy),
        storesList=stores_list,
        inputParameters=input_parameters,
    )
    assert result
    for extractor in result.values():
        assert isinstance(extractor, NpmRecordingExtractor)


def test_mixed_tdt_csv_session_partitions_events_correctly(tmp_path):
    session_copy = tmp_path / TDT_SESSION.name
    shutil.copytree(TDT_SESSION, session_copy)
    # Five epoch timestamps inside the TDT recording window (~1540913634–1540913791 s)
    csv_ttl_timestamps = np.array([1540913664.0, 1540913694.0, 1540913724.0, 1540913754.0, 1540913784.0])
    np.savetxt(
        session_copy / "csv_port_entries.csv",
        csv_ttl_timestamps,
        header="timestamps",
        comments="",
        fmt="%.6f",
    )
    mixed_storenames_map = {
        "Dv1A": "control_dms",
        "Dv2A": "signal_dms",
        "csv_port_entries": "port_entries_dms",
    }
    stores_list = _make_stores_list(mixed_storenames_map)
    input_parameters = {"noChannels": 2}
    result = _build_event_to_extractor(
        folder_path=str(session_copy),
        storesList=stores_list,
        inputParameters=input_parameters,
    )
    assert isinstance(result["Dv1A"], TdtRecordingExtractor)
    assert isinstance(result["Dv2A"], TdtRecordingExtractor)
    assert isinstance(result["csv_port_entries"], CsvRecordingExtractor)

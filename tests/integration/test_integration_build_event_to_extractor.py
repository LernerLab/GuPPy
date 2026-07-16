import shutil

import numpy as np

from guppy.extractors import (
    CsvRecordingExtractor,
    DoricRecordingExtractor,
    NpmRecordingExtractor,
    NwbRecordingExtractor,
    TdtRecordingExtractor,
)
from guppy.orchestration.read_raw_data import _build_event_to_extractor
from guppy_test_data import STUBBED_TESTING_DATA

CSV_SESSION = STUBBED_TESTING_DATA / "csv" / "sample_data_csv_1"
DORIC_SESSION = STUBBED_TESTING_DATA / "doric" / "sample_doric_1"
TDT_SESSION = STUBBED_TESTING_DATA / "tdt" / "Photo_63_207-181030-103332"
NPM_SESSION = STUBBED_TESTING_DATA / "npm" / "sampleData_NPM_1"
NWB_SESSION = STUBBED_TESTING_DATA / "nwb" / "mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2"

CSV_STORE_ID_TO_STORE_LABEL = {
    "Sample_Control_Channel": "control_region",
    "Sample_Signal_Channel": "signal_region",
    "Sample_TTL": "ttl",
}
DORIC_STORE_ID_TO_STORE_LABEL = {
    "AIn-1 - Raw": "control_region",
    "AIn-2 - Raw": "signal_region",
    "DI--O-1": "ttl",
}
TDT_STORE_ID_TO_STORE_LABEL = {
    "Dv1A": "control_dms",
    "Dv2A": "signal_dms",
    "PrtN": "port_entries_dms",
}
NPM_STORE_ID_TO_STORE_LABEL = {
    "file0_chev1": "signal_region",
    "file0_chod1": "control_region",
}
NWB_STORE_ID_TO_STORE_LABEL = {
    "fiber_photometry_response_series_0": "control_region",
    "fiber_photometry_response_series_1": "signal_region",
    "events": "ttl",
}


def _make_stores_list(store_id_to_store_label):
    return np.array([list(store_id_to_store_label.keys()), list(store_id_to_store_label.values())])


def test_doric_session_routes_all_events_to_doric_extractor(tmp_path):
    session_copy = tmp_path / DORIC_SESSION.name
    shutil.copytree(DORIC_SESSION, session_copy)
    stores_list = _make_stores_list(DORIC_STORE_ID_TO_STORE_LABEL)
    input_parameters = {"noChannels": 2}
    result = _build_event_to_extractor(
        folder_path=str(session_copy),
        store_array=stores_list,
        inputParameters=input_parameters,
    )
    assert result
    for extractor in result.values():
        assert isinstance(extractor, DoricRecordingExtractor)


def test_csv_session_routes_all_events_to_csv_extractor(tmp_path):
    session_copy = tmp_path / CSV_SESSION.name
    shutil.copytree(CSV_SESSION, session_copy)
    stores_list = _make_stores_list(CSV_STORE_ID_TO_STORE_LABEL)
    input_parameters = {"noChannels": 2}
    result = _build_event_to_extractor(
        folder_path=str(session_copy),
        store_array=stores_list,
        inputParameters=input_parameters,
    )
    assert result
    for extractor in result.values():
        assert isinstance(extractor, CsvRecordingExtractor)


def test_tdt_session_routes_all_events_to_tdt_extractor(tmp_path):
    session_copy = tmp_path / TDT_SESSION.name
    shutil.copytree(TDT_SESSION, session_copy)
    stores_list = _make_stores_list(TDT_STORE_ID_TO_STORE_LABEL)
    input_parameters = {"noChannels": 2}
    result = _build_event_to_extractor(
        folder_path=str(session_copy),
        store_array=stores_list,
        inputParameters=input_parameters,
    )
    assert result
    for extractor in result.values():
        assert isinstance(extractor, TdtRecordingExtractor)


def test_npm_session_routes_all_events_to_npm_extractor(tmp_path):
    session_copy = tmp_path / NPM_SESSION.name
    shutil.copytree(NPM_SESSION, session_copy)
    stores_list = _make_stores_list(NPM_STORE_ID_TO_STORE_LABEL)
    input_parameters = {
        "noChannels": 2,
        "npm_split_events": [False, True],
        "npm_timestamp_column_names": None,
        "npm_time_units": None,
    }
    result = _build_event_to_extractor(
        folder_path=str(session_copy),
        store_array=stores_list,
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
    mixed_store_id_to_store_label = {
        "Dv1A": "control_dms",
        "Dv2A": "signal_dms",
        "csv_port_entries": "port_entries_dms",
    }
    stores_list = _make_stores_list(mixed_store_id_to_store_label)
    input_parameters = {"noChannels": 2}
    result = _build_event_to_extractor(
        folder_path=str(session_copy),
        store_array=stores_list,
        inputParameters=input_parameters,
    )
    assert isinstance(result["Dv1A"], TdtRecordingExtractor)
    assert isinstance(result["Dv2A"], TdtRecordingExtractor)
    assert isinstance(result["csv_port_entries"], CsvRecordingExtractor)


def test_nwb_session_routes_all_events_to_nwb_extractor(tmp_path):
    session_copy = tmp_path / NWB_SESSION.name
    shutil.copytree(NWB_SESSION, session_copy)
    stores_list = _make_stores_list(NWB_STORE_ID_TO_STORE_LABEL)
    input_parameters = {"noChannels": 2}
    result = _build_event_to_extractor(
        folder_path=str(session_copy),
        store_array=stores_list,
        inputParameters=input_parameters,
    )
    assert result
    for extractor in result.values():
        assert isinstance(extractor, NwbRecordingExtractor)

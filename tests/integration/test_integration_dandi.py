"""Integration tests for the DANDI streaming path through step2 and step3.

These exercise the ``inputParameters["mode"] == "dandi"`` branches in
``guppy.orchestration.storenames`` and ``guppy.orchestration.read_raw_data``
without hitting the network: ``_stream_nwb`` is monkeypatched to open a local
mock NWB file from ``stubbed_testing_data/nwb/``.

DANDI is a narrower case than the standard per-modality integration tests —
it only runs step2 and step3 and doesn't fit the shared per-modality fixture
chain in conftest.py, so it lives in its own self-contained file.
"""

import csv
import os

import h5py
import pytest
from conftest import STUBBED_TESTING_DATA, _locate_output_directory
from pynwb import NWBHDF5IO

from guppy.extractors import dandi_nwb_recording_extractor as dandi_module
from guppy.extractors.base_recording_extractor import read_and_save_event
from guppy.orchestration import read_raw_data as read_raw_data_module
from guppy.testing.api import step2, step3

MOCK_NWB_FILE = (
    STUBBED_TESTING_DATA
    / "nwb"
    / "mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2"
    / "mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2.nwb"
)

SENTINEL_DANDI_URI = "dandi://mock/events.nwb"

STORENAMES_MAP = {
    "fiber_photometry_response_series_0": "control_region",
    "fiber_photometry_response_series_1": "signal_region",
    "events": "ttl",
}


@pytest.fixture
def patched_stream_nwb(monkeypatch):
    """Replace _stream_nwb with a function that opens the local mock NWB file."""

    def local_stream_nwb(*, dandiset_id, asset_path):
        io = NWBHDF5IO(str(MOCK_NWB_FILE), "r", load_namespaces=True)
        nwbfile = io.read()
        return nwbfile, io

    monkeypatch.setattr(dandi_module, "_stream_nwb", local_stream_nwb)

    # step3 spawns a multiprocessing pool that doesn't inherit the monkeypatch.
    # Replace the imported helper with a serial loop so the patched _stream_nwb
    # stays in effect.
    def serial_read_and_save_all_events(event_to_extractor, outputPath, numProcesses):
        for event, extractor in event_to_extractor.items():
            read_and_save_event(extractor, str(event), outputPath)

    monkeypatch.setattr(read_raw_data_module, "read_and_save_all_events", serial_read_and_save_all_events)


@pytest.fixture
def dandi_pipeline_state(tmp_path):
    """Create an empty session folder under tmp_path and return pipeline state."""
    base_directory = tmp_path / "dandi_base"
    base_directory.mkdir()
    session_copy = base_directory / "dandi_session"
    session_copy.mkdir()
    return {
        "base_directory": str(base_directory),
        "session_copy": str(session_copy),
    }


@pytest.fixture
def step2_dandi_output(dandi_pipeline_state, patched_stream_nwb):
    step2(
        base_dir=dandi_pipeline_state["base_directory"],
        selected_folders=[dandi_pipeline_state["session_copy"]],
        storenames_map=STORENAMES_MAP,
        dandi_uri_map={dandi_pipeline_state["session_copy"]: SENTINEL_DANDI_URI},
    )
    dandi_pipeline_state["output_directory"] = _locate_output_directory(
        session_copy=dandi_pipeline_state["session_copy"]
    )
    return dandi_pipeline_state


@pytest.fixture
def step3_dandi_output(step2_dandi_output, patched_stream_nwb):
    step3(
        base_dir=step2_dandi_output["base_directory"],
        selected_folders=[step2_dandi_output["session_copy"]],
        dandi_uri_map={step2_dandi_output["session_copy"]: SENTINEL_DANDI_URI},
    )
    step2_dandi_output["output_directory"] = _locate_output_directory(session_copy=step2_dandi_output["session_copy"])
    return step2_dandi_output


class TestDandiIntegration:
    """Exercises the DANDI orchestration branches for step2 and step3."""

    def test_step2_writes_stores_list(self, step2_dandi_output):
        stores_file_path = os.path.join(step2_dandi_output["output_directory"], "storesList.csv")
        assert os.path.exists(stores_file_path)

        with open(stores_file_path, newline="") as stores_file:
            stores_rows = list(csv.reader(stores_file))

        assert len(stores_rows) == 2
        assert stores_rows[0] == list(STORENAMES_MAP.keys())
        assert stores_rows[1] == list(STORENAMES_MAP.values())

    def test_step3_writes_hdf5_per_event(self, step3_dandi_output):
        output_directory = step3_dandi_output["output_directory"]
        for raw_storename in STORENAMES_MAP.keys():
            storename_file_path = os.path.join(output_directory, f"{raw_storename}.hdf5")
            assert os.path.exists(
                storename_file_path
            ), f"Missing HDF5 for storename {raw_storename!r} at {storename_file_path}"

            with h5py.File(storename_file_path, "r") as storename_file:
                assert "timestamps" in storename_file
                assert storename_file["timestamps"].shape[0] > 0


class TestDandiIntegrationMultiAsset:
    """Confirm dandi_uri_map routes per-session URIs correctly when multiple assets are selected."""

    def test_step2_writes_stores_list_for_each_session(self, tmp_path, patched_stream_nwb):
        base_directory = tmp_path / "dandi_base_multi"
        base_directory.mkdir()
        session_a = base_directory / "session_a"
        session_b = base_directory / "session_b"
        session_a.mkdir()
        session_b.mkdir()

        dandi_uri_map = {
            str(session_a): "dandi://mock/a.nwb",
            str(session_b): "dandi://mock/b.nwb",
        }

        step2(
            base_dir=str(base_directory),
            selected_folders=[str(session_a), str(session_b)],
            storenames_map=STORENAMES_MAP,
            dandi_uri_map=dandi_uri_map,
        )

        for session_directory in (session_a, session_b):
            output_directory = _locate_output_directory(session_copy=str(session_directory))
            stores_file_path = os.path.join(output_directory, "storesList.csv")
            assert os.path.exists(stores_file_path), f"Missing storesList.csv under {session_directory}"

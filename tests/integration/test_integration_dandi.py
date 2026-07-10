"""Integration tests for the DANDI streaming path through step1 and step2.

These exercise the ``inputParameters["mode"] == "dandi"`` branches in
``guppy.orchestration.store_labeling`` and ``guppy.orchestration.read_raw_data``
without hitting the network: ``_stream_nwb`` is monkeypatched to open a local
mock NWB file from ``stubbed_testing_data/nwb/``.

DANDI is a narrower case than the standard per-modality integration tests —
it only runs step1 and step2 and doesn't fit the shared per-modality fixture
chain in conftest.py, so it lives in its own self-contained file.
"""

import csv
import os

import h5py
import pytest
from conftest import STUBBED_TESTING_DATA, _locate_output_directory
from pynwb import NWBHDF5IO

from guppy.extractors import dandi_nwb_recording_extractor as dandi_module
from guppy.testing.api import step1, step2

MOCK_NWB_FILE = (
    STUBBED_TESTING_DATA
    / "nwb"
    / "mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2"
    / "mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2.nwb"
)

SENTINEL_DANDI_URI = "dandi://mock/events.nwb"

STORE_ID_TO_STORE_LABEL = {
    "fiber_photometry_response_series_0": "control_region",
    "fiber_photometry_response_series_1": "signal_region",
    "events": "ttl",
}


@pytest.fixture
def patched_stream_nwb(monkeypatch):
    """Replace _stream_nwb with a function that opens the local mock NWB file.

    With step2 running serially in the parent process (numberOfCores=1, the api.step2
    default), this single patch is all that's needed — the orchestrator's serial
    path runs in-process and naturally sees this monkeypatch.
    """

    import io as io_module

    from guppy.extractors.dandi_nwb_recording_extractor import _CountingRemfile

    def local_stream_nwb(*, dandiset_id, asset_path):
        io = NWBHDF5IO(str(MOCK_NWB_FILE), "r", load_namespaces=True)
        nwbfile = io.read()
        return nwbfile, io, _CountingRemfile(io_module.BytesIO())

    monkeypatch.setattr(dandi_module, "_stream_nwb", local_stream_nwb)


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
def step1_dandi_output(dandi_pipeline_state, patched_stream_nwb):
    step1(
        base_dir=dandi_pipeline_state["base_directory"],
        selected_folders=[dandi_pipeline_state["session_copy"]],
        store_id_to_store_label=STORE_ID_TO_STORE_LABEL,
        dandi_uri_map={dandi_pipeline_state["session_copy"]: SENTINEL_DANDI_URI},
    )
    dandi_pipeline_state["output_directory"] = _locate_output_directory(
        session_copy=dandi_pipeline_state["session_copy"]
    )
    return dandi_pipeline_state


@pytest.fixture
def step2_dandi_output(step1_dandi_output, patched_stream_nwb):
    step2(
        base_dir=step1_dandi_output["base_directory"],
        selected_folders=[step1_dandi_output["session_copy"]],
        dandi_uri_map={step1_dandi_output["session_copy"]: SENTINEL_DANDI_URI},
        selected_runs={step1_dandi_output["session_copy"]: ["1"]},
    )
    step1_dandi_output["output_directory"] = _locate_output_directory(session_copy=step1_dandi_output["session_copy"])
    return step1_dandi_output


class TestDandiIntegration:
    """Exercises the DANDI orchestration branches for step1 and step2."""

    def test_step1_writes_stores_list(self, step1_dandi_output):
        stores_file_path = os.path.join(step1_dandi_output["output_directory"], "storesList.csv")
        assert os.path.exists(stores_file_path)

        with open(stores_file_path, newline="") as stores_file:
            stores_rows = list(csv.reader(stores_file))

        assert len(stores_rows) == 2
        assert stores_rows[0] == list(STORE_ID_TO_STORE_LABEL.keys())
        assert stores_rows[1] == list(STORE_ID_TO_STORE_LABEL.values())

    def test_step2_writes_hdf5_per_event(self, step2_dandi_output):
        output_directory = step2_dandi_output["output_directory"]
        for store_id in STORE_ID_TO_STORE_LABEL.keys():
            store_id_file_path = os.path.join(output_directory, f"{store_id}.hdf5")
            assert os.path.exists(store_id_file_path), f"Missing HDF5 for store_id {store_id!r} at {store_id_file_path}"

            with h5py.File(store_id_file_path, "r") as store_id_file:
                assert "timestamps" in store_id_file
                assert store_id_file["timestamps"].shape[0] > 0


class TestDandiIntegrationMultiAsset:
    """Confirm dandi_uri_map routes per-session URIs correctly when multiple assets are selected."""

    def test_step1_writes_stores_list_for_each_session(self, tmp_path, patched_stream_nwb):
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

        step1(
            base_dir=str(base_directory),
            selected_folders=[str(session_a), str(session_b)],
            store_id_to_store_label=STORE_ID_TO_STORE_LABEL,
            dandi_uri_map=dandi_uri_map,
        )

        for session_directory in (session_a, session_b):
            output_directory = _locate_output_directory(session_copy=str(session_directory))
            stores_file_path = os.path.join(output_directory, "storesList.csv")
            assert os.path.exists(stores_file_path), f"Missing storesList.csv under {session_directory}"

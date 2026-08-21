"""Tests for guppy.utils.nwb_io."""

import json
import subprocess
import sys

import pytest

from guppy.utils.nwb_io import _CORE_TYPE_MAP
from guppy_test_data import STUBBED_TESTING_DATA

MOCK_NWB_NDX_FIBER_PHOTOMETRY_V0_1_FILE = (
    STUBBED_TESTING_DATA
    / "nwb"
    / "mock_nwbfile_ndx_fiber_photometry_v0_1_ndx_events_v0_2"
    / "mock_nwbfile_ndx_fiber_photometry_v0_1_ndx_events_v0_2.nwb"
)

# Registering ndx-fiber-photometry 0.2.x before reading a file written with 0.1.0 is what makes the
# process-wide type map shadow the file's own spec, and importing ndx-guppy is how an export does
# it. The probe therefore has to own its interpreter, as
# tests/unit/extractors/test_nwb_recording_extractor.py did before open_nwbfile_io existed. One
# reader per interpreter too: two reads of one path share hdmf's builder cache, so a failed read
# would poison the read under test.
#
# The two imports below are in the order an export puts them in, and that order is the point: the
# step module is what the GUI and the headless API load, and it reaches ndx-guppy only later, from
# inside add_to_nwbfile.
POISONED_READ_PROBE_SCRIPT = """
import json, sys, warnings
warnings.simplefilter("ignore")

import guppy.orchestration.export_nwb  # noqa: F401  -- the real entry point; imports nwb_io
import ndx_guppy  # noqa: F401  -- registers ndx-fiber-photometry 0.2.x, as add_to_nwbfile does

from pynwb import NWBHDF5IO
from guppy.utils.nwb_io import open_nwbfile_io

file_path, reader = sys.argv[1:3]
result = {"ndx_fiber_photometry_imported": "ndx_fiber_photometry" in sys.modules}
open_io = (lambda: NWBHDF5IO(file_path, "r")) if reader == "plain" else (lambda: open_nwbfile_io(path=file_path))

try:
    with open_io() as io:
        nwbfile = io.read()
        namespace = io.manager.type_map.namespace_catalog.get_namespace("ndx-fiber-photometry")
        result["namespace_version"] = namespace.version
        result["device_count"] = len(nwbfile.devices)
        result["response_series_shape"] = list(nwbfile.acquisition["fiber_photometry_response_series"].data.shape)
    result["read"] = "ok"
except Exception as exception:
    result["read"] = type(exception).__name__

print(json.dumps(result))
"""


def _run_probe(reader: str) -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            POISONED_READ_PROBE_SCRIPT,
            str(MOCK_NWB_NDX_FIBER_PHOTOMETRY_V0_1_FILE),
            reader,
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_the_snapshot_predates_every_extension():
    # The whole strategy rests on this copy being taken while the global type map still holds the
    # core namespaces alone. An extension imported ahead of guppy.utils.nwb_io would land in it.
    assert set(_CORE_TYPE_MAP.namespace_catalog.namespaces) == {"core", "hdmf-common", "hdmf-experimental"}


class TestOpenNwbfileIoUnderAConflictingExtension:
    """``open_nwbfile_io`` reads a v0.1.0 file in a process where 0.2.x is already registered."""

    @pytest.fixture(scope="class")
    def plain_probe(self) -> dict:
        return _run_probe("plain")

    @pytest.fixture(scope="class")
    def probe(self) -> dict:
        return _run_probe("open_nwbfile_io")

    def test_the_conflicting_extension_is_registered(self, probe):
        # The premise: without it, the plain read below would succeed and prove nothing.
        assert probe["ndx_fiber_photometry_imported"] is True

    def test_a_plain_read_fails(self, plain_probe):
        assert plain_probe["read"] == "ConstructError"

    def test_the_files_own_namespace_version_is_used(self, probe):
        assert probe["read"] == "ok"
        assert probe["namespace_version"] == "0.1.0"

    def test_the_file_reads_whole(self, probe):
        assert probe["device_count"] == 7
        assert probe["response_series_shape"] == [3000, 2]

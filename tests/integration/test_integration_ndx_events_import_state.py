"""Verify NWB event extraction is unaffected by whether ``ndx_events`` has been imported.

``LabeledEvents`` is exposed by two different classes depending on the process-wide pynwb type
map: importing ``ndx_events`` registers the hand-written class (which stores its labels on
``.labels``), while leaving it unimported makes pynwb generate a class from the file's cached
spec (which stores them on ``.data__labels``). A process has a single type map, so the two states
cannot both be produced in one interpreter — each is exercised in its own subprocess.
"""

import json
import subprocess
import sys

import numpy as np
import pytest

from guppy_test_data import STUBBED_TESTING_DATA

NWB_SESSION = STUBBED_TESTING_DATA / "nwb" / "mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2"

EXPECTED_EVENTS = [
    "AnnotatedEventsTable_Punishment",
    "AnnotatedEventsTable_Reward",
    "events",
    "fiber_photometry_response_series_0",
    "fiber_photometry_response_series_1",
    "labeled_events_label_1",
    "labeled_events_label_2",
    "labeled_events_label_3",
]
EXPECTED_LABEL_2_TIMESTAMPS = np.array([41.0, 44.0, 47.0, 50.0, 53.0])

# Runs in a fresh interpreter so the type map is decided solely by ``import_ndx_events``.
# Discovery covers _discover_ndx_events_v02; the read covers _build_event_index_v02.
PROBE_SCRIPT = """
import json, sys
if sys.argv[1] == "import_ndx_events":
    import ndx_events  # noqa: F401
from guppy.extractors.nwb_recording_extractor import NwbRecordingExtractor

folder_path = sys.argv[2]
events, flags = NwbRecordingExtractor.discover_events_and_flags(folder_path)
extractor = NwbRecordingExtractor(folder_path=folder_path)
output_dicts = extractor.read(events=["labeled_events_label_2"], outputPath=folder_path)
print(json.dumps({
    "events": sorted(events),
    "flags": flags,
    "label_2_timestamps": output_dicts[0]["timestamps"].tolist(),
}))
"""


@pytest.mark.parametrize("import_mode", ["import_ndx_events", "no_ndx_events"])
def test_labeled_events_extracted_identically_in_both_import_states(import_mode):
    completed = subprocess.run(
        [sys.executable, "-c", PROBE_SCRIPT, import_mode, str(NWB_SESSION)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)

    assert result["events"] == EXPECTED_EVENTS
    assert result["flags"] == []
    np.testing.assert_allclose(result["label_2_timestamps"], EXPECTED_LABEL_2_TIMESTAMPS)

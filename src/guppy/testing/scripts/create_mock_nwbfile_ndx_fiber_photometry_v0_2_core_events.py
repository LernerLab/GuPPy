"""Generate a mock NWB file for testing the NWB recording extractor with core pynwb 4.0 events.

IMPORTANT: This script must be run with pynwb>=4 and ndx-fiber-photometry==0.2.3 installed.

The events types (``EventsTable``, ``TimestampVectorData``) are part of the core NWB schema as of
NWB Schema 2.10.0 / pynwb 4.0, so no events extension is required.

To run, create the isolated conda environment defined alongside this script:
    conda env create -f src/guppy/testing/scripts/environment_ndx_fiber_photometry_v0_2_core_events.yaml
    conda activate guppy_ndx_fiber_photometry_v0_2_core_events

Then run from the project root:
    python src/guppy/testing/scripts/create_mock_nwbfile_ndx_fiber_photometry_v0_2_core_events.py

The output is written to
stubbed_testing_data/nwb/mock_nwbfile_ndx_fiber_photometry_v0_2_core_events/mock_nwbfile_ndx_fiber_photometry_v0_2_core_events.nwb,
relative to the repository root. The directory is created if it does not exist.

The file contains:
- FiberPhotometryResponseSeries with 3000 samples at 30 Hz across 2 channels (control, signal)
- Core EventsTable named "simple_events" (timestamps 45–54 s, no annotation column)
  → discovers as one event "simple_events"
- Core EventsTable named "annotated_events" with a text "annotation" column containing
  "Reward" (timestamps 41–45 s) and "Punishment" (timestamps 55–59 s)
  → discovers as two events "annotated_events_Reward" and "annotated_events_Punishment"
- Core EventsTable named "strobe_events" with a text "strobe" code column (codes 16, 2064,
  0 at timestamps 60–64 s), mirroring NeuroConv's TDTEventsInterface output
  → discovers as "strobe_events_0", "strobe_events_16", "strobe_events_2064"
"""

import datetime
from pathlib import Path

import numpy as np
from _mock_nwb_common import add_ndx_fiber_photometry_metadata
from ndx_fiber_photometry import FiberPhotometryResponseSeries
from pynwb import NWBHDF5IO, NWBFile
from pynwb.event import EventsTable

# Output path relative to this script's location (repo_root/stubbed_testing_data/nwb/...)
_REPO_ROOT = Path(__file__).resolve().parents[4]
_OUTPUT_PATH = (
    _REPO_ROOT
    / "stubbed_testing_data"
    / "nwb"
    / "mock_nwbfile_ndx_fiber_photometry_v0_2_core_events"
    / "mock_nwbfile_ndx_fiber_photometry_v0_2_core_events.nwb"
)


def main() -> None:
    """Create and write a mock NWB file using ndx-fiber-photometry v0.2 and core pynwb 4.0 events."""
    nwbfile = NWBFile(
        session_description="Mock session for NWB extractor testing (core pynwb 4.0 events).",
        identifier="mock_nwbfile_ndx_fiber_photometry_v0_2_core_events",
        session_start_time=datetime.datetime.now(datetime.timezone.utc),
    )

    # Build ndx-fiber-photometry boilerplate (required for a valid NWB file,
    # but Guppy only reads the FiberPhotometryResponseSeries and events below).
    fiber_photometry_table_region = add_ndx_fiber_photometry_metadata(nwbfile)

    # --- FiberPhotometryResponseSeries: 3000 samples at 30 Hz, 2 channels (control, signal) ---
    nwbfile.add_acquisition(
        FiberPhotometryResponseSeries(
            name="fiber_photometry_response_series",
            description="Mock fluorescence traces: 3000 samples at 30 Hz, 2 channels (control, signal).",
            data=np.random.randn(3000, 2),
            unit="n.a.",
            rate=30.0,
            fiber_photometry_table_region=fiber_photometry_table_region,
        )
    )

    # --- Simple EventsTable: timestamps 45–54 s, no annotation column ---
    # Discovers as a single event named "simple_events".
    simple_events = EventsTable(name="simple_events", description="Mock events with no annotation column.")
    for timestamp in np.arange(45, 55, dtype=np.float64):
        simple_events.add_row(timestamp=timestamp)
    nwbfile.add_events_table(simple_events)

    # --- Annotated EventsTable: timestamps 41–59 s, text "annotation" column ---
    # An "annotation" column with values "Reward" and "Punishment".
    # Discovers as "annotated_events_Reward" and "annotated_events_Punishment".
    annotated_events = EventsTable(
        name="annotated_events",
        description="Mock events with a text annotation column.",
    )
    annotated_events.add_column(name="annotation", description="Text label for each event.")
    for timestamp in np.arange(41, 46, dtype=np.float64):
        annotated_events.add_row(timestamp=timestamp, annotation="Reward")
    for timestamp in np.arange(55, 60, dtype=np.float64):
        annotated_events.add_row(timestamp=timestamp, annotation="Punishment")
    nwbfile.add_events_table(annotated_events)

    # --- Strobe EventsTable: timestamps 60–64 s, text "strobe" code column ---
    # Mirrors what NeuroConv's TDTEventsInterface writes for a coded (strobe) epoc:
    # a single value column named "strobe" (not "annotation"). Codes are 16, 2064, 0.
    # Discovers as "strobe_events_0", "strobe_events_16", "strobe_events_2064".
    strobe_events = EventsTable(
        name="strobe_events",
        description="Mock events with a strobe code column, as written by TDTEventsInterface.",
    )
    strobe_events.add_column(name="strobe", description="Strobe code for each event.")
    for timestamp, strobe_code in zip(np.arange(60, 65, dtype=np.float64), ["16", "2064", "0", "16", "2064"]):
        strobe_events.add_row(timestamp=timestamp, strobe=strobe_code)
    nwbfile.add_events_table(strobe_events)

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NWBHDF5IO(_OUTPUT_PATH, "w") as io:
        io.write(nwbfile)
    print(f"Mock NWB file written to {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()

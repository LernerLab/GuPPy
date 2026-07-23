"""Generate a mock NWB file for testing the NWB recording extractor.

IMPORTANT: This script must be run with ndx-fiber-photometry==0.2.3 and ndx-events==0.2.2 installed.

To run, create the isolated conda environment defined alongside this script:
    conda env create -f src/guppy/testing/scripts/environment_ndx_fiber_photometry_v0_2_ndx_events_v0_2.yaml
    conda activate guppy_ndx_fiber_photometry_v0_2_ndx_events_v0_2

Then run from the project root:
    python src/guppy/testing/scripts/create_mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2.py

The output is written to stubbed_testing_data/nwb/mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2/mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2.nwb,
relative to the repository root. The directory is created if it does not exist.

The file contains:
- FiberPhotometryResponseSeries with 3000 samples at 30 Hz across 2 channels (control, signal)
- Events (timestamps 45–54 s)
- LabeledEvents with 3 labels (timestamps 40–54 s)
- AnnotatedEventsTable with Reward and Punishment event types
"""

import datetime
from pathlib import Path

import numpy as np
from _mock_nwb_common import add_ndx_fiber_photometry_metadata
from ndx_events import AnnotatedEventsTable, Events, LabeledEvents
from ndx_fiber_photometry import FiberPhotometryResponseSeries
from pynwb import NWBHDF5IO, NWBFile

# Output path relative to this script's location (repo_root/stubbed_testing_data/nwb/...)
_REPO_ROOT = Path(__file__).resolve().parents[4]
_OUTPUT_PATH = (
    _REPO_ROOT
    / "stubbed_testing_data"
    / "nwb"
    / "mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2"
    / "mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2.nwb"
)


def main() -> None:
    """Create and write a mock NWB file using ndx-fiber-photometry v0.2 and ndx-events v0.2."""
    nwbfile = NWBFile(
        session_description="Mock session for NWB extractor testing.",
        identifier="mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2",
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

    # --- Events (timestamps 45–54 s) ---
    nwbfile.add_acquisition(
        Events(
            name="events",
            description="Mock events.",
            timestamps=np.arange(45, 55, dtype=np.float64),
        )
    )

    # --- LabeledEvents with 3 labels (timestamps 40–54 s) ---
    nwbfile.add_acquisition(
        LabeledEvents(
            name="labeled_events",
            description="Mock labeled events.",
            timestamps=np.arange(40, 55, dtype=np.float64),
            data=np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.uint8),
            labels=["label_1", "label_2", "label_3"],
        )
    )

    # --- AnnotatedEventsTable with Reward and Punishment event types ---
    annotated_events = AnnotatedEventsTable(
        name="AnnotatedEventsTable",
        description="Annotated events from the mock experiment.",
        resolution=1e-5,
    )
    annotated_events.add_event_type(
        label="Reward",
        event_description="Times when the subject received juice reward.",
        event_times=[41.0, 42.0, 43.0, 44.0, 45.0],
    )
    annotated_events.add_event_type(
        label="Punishment",
        event_description="Times when the subject received a mild shock.",
        event_times=[55.0, 56.0, 57.0, 58.0, 59.0],
    )
    nwbfile.add_acquisition(annotated_events)

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NWBHDF5IO(_OUTPUT_PATH, "w") as io:
        io.write(nwbfile)
    print(f"Mock NWB file written to {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()

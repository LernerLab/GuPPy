"""Generate a mock NWB file for testing the NWB recording extractor against ndx-fiber-photometry v0.1.0.

IMPORTANT: This script must be run with ndx-fiber-photometry==0.1.0 installed.
The current (post-v0.1.0) version moved device classes to ndx-ophys-devices, making
this script incompatible with the newer package.

To run, create an environment with the old version:
    pip install ndx-fiber-photometry==0.1.0

Then run from the project root:
    python src/guppy/testing/scripts/create_mock_nwbfile_v0_1_0.py

The output is written to stubbed_testing_data/nwb/mock_nwbfile_v0_1_0/mock_nwbfile_v0_1_0.nwb,
relative to the repository root. The directory is created if it does not exist.

The file contains:
- FiberPhotometryResponseSeries with 3000 samples at 30 Hz across 2 channels (control, signal)
- Events (timestamps 45–54 s)
- LabeledEvents with 3 labels (timestamps 40–54 s)
- AnnotatedEventsTable with Reward and Punishment event types

All ndx-fiber-photometry metadata uses the v0.1.0 API, which predates the ndx-ophys-devices
split. Devices are created directly from ndx_fiber_photometry (no separate *Model classes),
and FiberPhotometry only accepts fiber_photometry_table (no virus/injection/indicator containers).
"""

import datetime
from pathlib import Path

import numpy as np
from ndx_events import AnnotatedEventsTable, Events, LabeledEvents
from ndx_fiber_photometry import (
    BandOpticalFilter,
    DichroicMirror,
    ExcitationSource,
    FiberPhotometry,
    FiberPhotometryResponseSeries,
    FiberPhotometryTable,
    Indicator,
    OpticalFiber,
    Photodetector,
)
from pynwb import NWBHDF5IO, NWBFile

# Output path relative to this script's location (repo_root/stubbed_testing_data/nwb/...)
_REPO_ROOT = Path(__file__).resolve().parents[4]
_OUTPUT_PATH = _REPO_ROOT / "stubbed_testing_data" / "nwb" / "mock_nwbfile_v0_1_0" / "mock_nwbfile_v0_1_0.nwb"


def _add_ndx_fiber_photometry_metadata(nwbfile):
    """Add all ndx-fiber-photometry v0.1.0 hardware metadata to *nwbfile*.

    Uses the v0.1.0 API where devices are instantiated directly from ndx_fiber_photometry
    (no separate ndx-ophys-devices model classes) and FiberPhotometry only holds a
    FiberPhotometryTable.

    Parameters
    ----------
    nwbfile : NWBFile
        The file to populate in-place.

    Returns
    -------
    fiber_photometry_table_region
        Table region referencing both channels; pass this to
        FiberPhotometryResponseSeries.
    """
    # --- Indicator (shared by both channels) ---
    indicator = Indicator(
        name="indicator",
        description="Photometry indicator.",
        label="GCamp6f",
        injection_location="VTA",
        injection_coordinates_in_mm=(3.0, 2.0, 1.0),
    )

    # --- Optical fiber (shared by both channels) ---
    optical_fiber = OpticalFiber(
        name="optical_fiber",
        model="OF-123",
        numerical_aperture=0.2,
        core_diameter_in_um=400.0,
    )

    # --- Excitation sources (one per channel: isosbestic control at 405 nm, signal at 470 nm) ---
    excitation_source_control = ExcitationSource(
        name="excitation_source_control",
        description="405 nm isosbestic excitation source.",
        model="laser model",
        illumination_type="laser",
        excitation_wavelength_in_nm=405.0,
    )
    excitation_source_signal = ExcitationSource(
        name="excitation_source_signal",
        description="470 nm signal excitation source.",
        model="laser model",
        illumination_type="laser",
        excitation_wavelength_in_nm=470.0,
    )

    # --- Photodetector (shared by both channels) ---
    photodetector = Photodetector(
        name="photodetector",
        description="Photodetector for green emission.",
        detector_type="PMT",
        detected_wavelength_in_nm=525.0,
        gain=100.0,
    )

    # --- Dichroic mirror (shared by both channels) ---
    dichroic_mirror = DichroicMirror(
        name="dichroic_mirror",
        description="Dichroic mirror for fiber photometry.",
        model="DM-123",
        cut_on_wavelength_in_nm=470.0,
        cut_off_wavelength_in_nm=500.0,
        reflection_band_in_nm=(490.0, 520.0),
        transmission_band_in_nm=(460.0, 480.0),
        angle_of_incidence_in_degrees=45.0,
    )

    # --- Band optical filter (shared by both channels) ---
    optical_filter = BandOpticalFilter(
        name="optical_filter",
        description="Band optical filter for fiber photometry.",
        model="BOF-123",
        center_wavelength_in_nm=505.0,
        bandwidth_in_nm=30.0,
        filter_type="Bandpass",
    )

    # --- Fiber photometry table: row 0 = control channel, row 1 = signal channel ---
    fiber_photometry_table = FiberPhotometryTable(
        name="fiber_photometry_table",
        description="Metadata table mapping each photometry channel to its hardware.",
    )
    fiber_photometry_table.add_row(
        location="VTA",
        coordinates=(3.0, 2.0, 1.0),
        indicator=indicator,
        optical_fiber=optical_fiber,
        excitation_source=excitation_source_control,
        photodetector=photodetector,
        dichroic_mirror=dichroic_mirror,
        emission_filter=optical_filter,
    )  # control channel (isosbestic)
    fiber_photometry_table.add_row(
        location="VTA",
        coordinates=(3.0, 2.0, 1.0),
        indicator=indicator,
        optical_fiber=optical_fiber,
        excitation_source=excitation_source_signal,
        photodetector=photodetector,
        dichroic_mirror=dichroic_mirror,
        emission_filter=optical_filter,
    )  # signal channel

    fiber_photometry_table_region = fiber_photometry_table.create_fiber_photometry_table_region(
        region=[0, 1], description="control and signal channels"
    )

    # --- Register devices and lab metadata ---
    nwbfile.add_device(indicator)
    nwbfile.add_device(optical_fiber)
    nwbfile.add_device(excitation_source_control)
    nwbfile.add_device(excitation_source_signal)
    nwbfile.add_device(photodetector)
    nwbfile.add_device(dichroic_mirror)
    nwbfile.add_device(optical_filter)

    # v0.1.0: FiberPhotometry only takes fiber_photometry_table (no virus/injection containers)
    nwbfile.add_lab_meta_data(
        FiberPhotometry(
            name="fiber_photometry",
            fiber_photometry_table=fiber_photometry_table,
        )
    )

    return fiber_photometry_table_region


def main():
    nwbfile = NWBFile(
        session_description="Mock session for NWB extractor testing (ndx-fiber-photometry v0.1.0).",
        identifier="mock_nwbfile_v0_1_0",
        session_start_time=datetime.datetime.now(datetime.timezone.utc),
    )

    # Build ndx-fiber-photometry v0.1.0 boilerplate (required for a valid NWB file,
    # but Guppy only reads the FiberPhotometryResponseSeries and events below).
    fiber_photometry_table_region = _add_ndx_fiber_photometry_metadata(nwbfile)

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

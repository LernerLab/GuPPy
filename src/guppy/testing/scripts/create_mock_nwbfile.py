"""Generate a mock NWB file for testing the NWB recording extractor.

Run from the project root:
    python src/guppy/testing/scripts/create_mock_nwbfile.py

The output is written to stubbed_testing_data/nwb/mock_nwbfile/mock_nwbfile.nwb,
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
from ndx_events import AnnotatedEventsTable, Events, LabeledEvents
from ndx_fiber_photometry import (
    FiberPhotometry,
    FiberPhotometryIndicators,
    FiberPhotometryResponseSeries,
    FiberPhotometryTable,
    FiberPhotometryViruses,
    FiberPhotometryVirusInjections,
)
from ndx_ophys_devices import (
    BandOpticalFilter,
    BandOpticalFilterModel,
    DichroicMirror,
    DichroicMirrorModel,
    ExcitationSource,
    ExcitationSourceModel,
    FiberInsertion,
    Indicator,
    OpticalFiber,
    OpticalFiberModel,
    Photodetector,
    PhotodetectorModel,
    ViralVector,
    ViralVectorInjection,
)
from pynwb import NWBHDF5IO, NWBFile

# Output path relative to this script's location (repo_root/stubbed_testing_data/nwb/...)
_REPO_ROOT = Path(__file__).resolve().parents[4]
_OUTPUT_PATH = _REPO_ROOT / "stubbed_testing_data" / "nwb" / "mock_nwbfile" / "mock_nwbfile.nwb"


def _add_ndx_fiber_photometry_metadata(nwbfile):
    """Add all ndx-fiber-photometry hardware metadata to *nwbfile*.

    This boilerplate is required to produce a valid NWBFile, but Guppy only
    reads the FiberPhotometryResponseSeries produced by the caller — not any
    of the hardware objects registered here.

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
    # --- Viral vector and injection (shared by both channels) ---
    viral_vector = ViralVector(
        name="viral_vector",
        description="AAV viral vector for the photometry indicator.",
        construct_name="AAV-CaMKII-GCaMP6f",
        manufacturer="Vector Manufacturer",
        titer_in_vg_per_ml=1.0e12,
    )
    viruses = FiberPhotometryViruses(viral_vectors=[viral_vector])

    viral_vector_injection = ViralVectorInjection(
        name="viral_vector_injection",
        description="Viral vector injection for fiber photometry.",
        location="Ventral Tegmental Area (VTA)",
        hemisphere="right",
        reference="Bregma at the cortical surface",
        ap_in_mm=3.0,
        ml_in_mm=2.0,
        dv_in_mm=1.0,
        pitch_in_deg=0.0,
        yaw_in_deg=0.0,
        roll_in_deg=0.0,
        stereotactic_rotation_in_deg=0.0,
        stereotactic_tilt_in_deg=0.0,
        volume_in_uL=0.45,
        injection_date="1970-01-01T00:00:00+00:00",
        viral_vector=viral_vector,
    )
    virus_injections = FiberPhotometryVirusInjections(viral_vector_injections=[viral_vector_injection])

    # --- Indicator (shared by both channels) ---
    indicator = Indicator(
        name="indicator",
        description="Photometry indicator.",
        label="GCamp6f",
        viral_vector_injection=viral_vector_injection,
    )
    indicators = FiberPhotometryIndicators(indicators=[indicator])

    # --- Optical fiber (shared by both channels) ---
    optical_fiber_model = OpticalFiberModel(
        name="optical_fiber_model",
        manufacturer="Fiber Manufacturer",
        model_number="OF-123",
        description="Optical fiber model for fiber photometry.",
        numerical_aperture=0.2,
        core_diameter_in_um=400.0,
        active_length_in_mm=2.0,
        ferrule_name="cFCF - ∅2.5mm Ceramic Ferrule",
        ferrule_model="SM-SC-CF-10-FM",
        ferrule_diameter_in_mm=2.5,
    )
    optical_fiber = OpticalFiber(
        name="optical_fiber",
        description="Optical fiber for fiber photometry.",
        serial_number="OF-SN-123456",
        model=optical_fiber_model,
        fiber_insertion=FiberInsertion(
            name="fiber_insertion",
            depth_in_mm=3.5,
            insertion_position_ap_in_mm=3.0,
            insertion_position_ml_in_mm=2.0,
            insertion_position_dv_in_mm=1.0,
            position_reference="bregma",
            hemisphere="right",
            insertion_angle_pitch_in_deg=10.0,
        ),
    )

    # --- Excitation source (shared by both channels) ---
    excitation_source_model = ExcitationSourceModel(
        name="excitation_source_model",
        manufacturer="Laser Manufacturer",
        model_number="ES-123",
        description="Excitation source model for fiber photometry.",
        source_type="laser",
        excitation_mode="one-photon",
        wavelength_range_in_nm=[400.0, 800.0],
    )
    excitation_source = ExcitationSource(
        name="excitation_source",
        description="Excitation source for fiber photometry.",
        serial_number="ES-SN-123456",
        model=excitation_source_model,
        power_in_W=0.7,
        intensity_in_W_per_m2=0.005,
        exposure_time_in_s=2.51e-13,
    )

    # --- Photodetector (shared by both channels) ---
    photodetector_model = PhotodetectorModel(
        name="photodetector_model",
        manufacturer="Detector Manufacturer",
        model_number="PD-123",
        description="Photodetector model for fiber photometry.",
        detector_type="PMT",
        wavelength_range_in_nm=[400.0, 800.0],
        gain=100.0,
        gain_unit="A/W",
    )
    photodetector = Photodetector(
        name="photodetector",
        description="Photodetector for fiber photometry.",
        serial_number="PD-SN-123456",
        model=photodetector_model,
    )

    # --- Dichroic mirror (shared by both channels) ---
    dichroic_mirror = DichroicMirror(
        name="dichroic_mirror",
        description="Dichroic mirror for fiber photometry.",
        serial_number="DM-SN-123456",
        model=DichroicMirrorModel(
            name="dichroic_mirror_model",
            manufacturer="Mirror Manufacturer",
            model_number="DM-123",
            description="Dichroic mirror model for fiber photometry.",
            cut_on_wavelength_in_nm=470.0,
            cut_off_wavelength_in_nm=500.0,
            reflection_band_in_nm=[490.0, 520.0],
            transmission_band_in_nm=[460.0, 480.0],
            angle_of_incidence_in_degrees=45.0,
        ),
    )

    # --- Optical filter (shared by both channels) ---
    optical_filter = BandOpticalFilter(
        name="optical_filter",
        description="Band optical filter for fiber photometry.",
        serial_number="BOF-SN-123456",
        model=BandOpticalFilterModel(
            name="optical_filter_model",
            manufacturer="Filter Manufacturer",
            model_number="BOF-123",
            description="Band optical filter model for fiber photometry.",
            filter_type="Bandpass",
            center_wavelength_in_nm=505.0,
            bandwidth_in_nm=30.0,  # 505 ± 15 nm
        ),
    )

    # --- Fiber photometry table: row 0 = control channel, row 1 = signal channel ---
    fiber_photometry_table = FiberPhotometryTable(
        name="fiber_photometry_table",
        description="Metadata table mapping each photometry channel to its hardware.",
    )
    shared_hardware = dict(
        location="VTA",
        indicator=indicator,
        optical_fiber=optical_fiber,
        excitation_source=excitation_source,
        photodetector=photodetector,
        dichroic_mirror=dichroic_mirror,
        emission_filter=optical_filter,
    )
    fiber_photometry_table.add_row(
        excitation_wavelength_in_nm=405.0,
        emission_wavelength_in_nm=525.0,
        **shared_hardware,
    )  # control channel (isosbestic)
    fiber_photometry_table.add_row(
        excitation_wavelength_in_nm=470.0,
        emission_wavelength_in_nm=525.0,
        **shared_hardware,
    )  # signal channel

    fiber_photometry_table_region = fiber_photometry_table.create_fiber_photometry_table_region(
        region=[0, 1], description="control and signal channels"
    )

    # --- Register all devices and lab metadata ---
    nwbfile.add_device_model(optical_fiber_model)
    nwbfile.add_device(optical_fiber)
    nwbfile.add_device_model(excitation_source_model)
    nwbfile.add_device(excitation_source)
    nwbfile.add_device_model(photodetector_model)
    nwbfile.add_device(photodetector)
    nwbfile.add_device_model(dichroic_mirror.model)
    nwbfile.add_device(dichroic_mirror)
    nwbfile.add_device_model(optical_filter.model)
    nwbfile.add_device(optical_filter)

    nwbfile.add_lab_meta_data(
        FiberPhotometry(
            name="fiber_photometry",
            fiber_photometry_table=fiber_photometry_table,
            fiber_photometry_viruses=viruses,
            fiber_photometry_virus_injections=virus_injections,
            fiber_photometry_indicators=indicators,
        )
    )

    return fiber_photometry_table_region


def main():
    nwbfile = NWBFile(
        session_description="Mock session for NWB extractor testing.",
        identifier="mock_nwbfile",
        session_start_time=datetime.datetime.now(datetime.timezone.utc),
    )

    # Build ndx-fiber-photometry boilerplate (required for a valid NWB file,
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

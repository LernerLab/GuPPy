"""Shared ndx-fiber-photometry v0.2 hardware-metadata boilerplate for the mock-NWB generators.

Imported as a sibling module by the ``create_mock_nwbfile_ndx_fiber_photometry_v0_2_*`` scripts
(the script's own directory is on ``sys.path`` when run as ``python .../create_mock_...py``), so it
must only depend on the same isolated-environment packages those scripts use — ndx-fiber-photometry
0.2.x, ndx-ophys-devices, and pynwb — never on Guppy itself.
"""

from hdmf.common import DynamicTableRegion
from ndx_fiber_photometry import (
    FiberPhotometry,
    FiberPhotometryIndicators,
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
from pynwb import NWBFile


def add_ndx_fiber_photometry_metadata(nwbfile: NWBFile) -> DynamicTableRegion:
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

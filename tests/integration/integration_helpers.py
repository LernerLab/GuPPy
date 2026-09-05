"""Integration-suite-local shared constants and helpers.

Split out of ``conftest.py`` so the handful of integration test modules that need these symbols
at import time (``test_integration_step2.py``, ``test_integration_dandi.py``) can import them
directly via a package-relative import, instead of relying on the ambiguous bare ``conftest`` name.
"""

from datetime import datetime
from pathlib import Path

from guppy.utils.acquisition_format import acquisition_supplies_session_start_time
from guppy.utils.nwb_metadata import (
    build_metadata_dict,
    derive_channels,
    dump_yaml,
    load_yaml,
    parse_metadata_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_METADATA = PROJECT_ROOT / "data" / "fiber_photometry_metadata_example.yaml"

# Supplied by the form for the formats whose raw files record no recording start.
SUPPLIED_SESSION_START_TIME = datetime.fromisoformat("2020-05-04T09:00:00+00:00")

# Per-role annotations for the derived channels, using the example library's device names.
COMMON_CHANNEL_ANNOTATIONS = {
    "emission_wavelength_in_nm": 525.0,
    "optical_fiber": "optical_fiber",
    "photodetector": "photodetector",
    "indicator": "dms_green_fluorophore",
    "dichroic_mirror": "dichroic_mirror",
    "emission_filter": "emission_filter",
}
ROLE_CHANNEL_ANNOTATIONS = {
    "control": {  # isosbestic
        "excitation_wavelength_in_nm": 405.0,
        "excitation_source": "excitation_source_isosbestic_control",
        "excitation_filter": "isosbestic_excitation_filter",
    },
    "signal": {  # calcium
        "excitation_wavelength_in_nm": 465.0,
        "excitation_source": "excitation_source_calcium_signal",
        "excitation_filter": "excitation_filter",
    },
}

REPRESENTATIVE_SESSIONS = {
    "csv": {
        "session_subdir": "csv/sample_data_csv_1",
        "store_id_to_store_label": {
            "Sample_Control_Channel": "control_region",
            "Sample_Signal_Channel": "signal_region",
            "Sample_TTL": "ttl",
        },
        "npm_timestamp_column_name": None,
        "npm_time_unit": None,
        "npm_split_events": None,
    },
    "tdt": {
        "session_subdir": "tdt/Photo_63_207-181030-103332",
        "store_id_to_store_label": {
            "Dv1A": "control_dms",
            "Dv2A": "signal_dms",
            "PrtN": "port_entries_dms",
        },
        "npm_timestamp_column_name": None,
        "npm_time_unit": None,
        "npm_split_events": None,
    },
    "npm": {
        "session_subdir": "npm/sampleData_NPM_5",
        "store_id_to_store_label": {
            "file0_chev1": "control_region1",
            "file0_chod1": "signal_region1",
            "event0": "ttl_region1",
        },
        "npm_timestamp_column_name": None,
        # Header-less session: its clock is in milliseconds, which only the user can state.
        "npm_time_unit": "milliseconds",
        "npm_split_events": None,
    },
    "doric": {
        "session_subdir": "doric/sample_doric_1",
        "store_id_to_store_label": {
            "AIn-1 - Raw": "control_region",
            "AIn-2 - Raw": "signal_region",
            "DI--O-1": "ttl",
        },
        "npm_timestamp_column_name": None,
        "npm_time_unit": None,
        "npm_split_events": None,
    },
    "nwb": {
        "session_subdir": "nwb/mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2",
        "store_id_to_store_label": {
            "fiber_photometry_response_series_0": "control_region",
            "fiber_photometry_response_series_1": "signal_region",
            "events": "ttl",
        },
        "npm_timestamp_column_name": None,
        "npm_time_unit": None,
        "npm_split_events": None,
    },
}


def _locate_output_directory(*, session_copy: str) -> str:
    session_name = Path(session_copy).name
    output_directories = sorted(list(Path(session_copy).glob(f"{session_name}_output_*")))
    assert output_directories, f"No output directories found in {session_copy}"

    for output_directory in output_directories:
        if (Path(output_directory) / "storesList.csv").exists():
            return output_directory

    raise AssertionError(f"No storesList.csv found in any output directory under {session_copy}")


def write_metadata_yaml(*, session_folder_path: str, output_directory: str, acquisition_format: str, path: Path) -> str:
    """Write a session metadata YAML the way the form does: device library + per-channel annotations."""
    channels = derive_channels(output_dir=output_directory)
    # Reuse the example's hardware/biology library (recombined into merged device entries).
    devices, _channel_rows, _scalars = parse_metadata_dict(metadata=load_yaml(EXAMPLE_METADATA), channels=channels)
    channel_rows = [{**COMMON_CHANNEL_ANNOTATIONS, **ROLE_CHANNEL_ANNOTATIONS[channel.role]} for channel in channels]
    scalars = {
        "session_description": "RI30 photometry session",
        "identifier": f"{acquisition_format}_run1",
        "lab": "Lerner Lab",
        "institution": "Northwestern University",
        "subject_id": "63_207",
        "sex": "M",
        "species": "Mus musculus",
    }
    # The form only asks for a start time where the raw files carry none.
    if not acquisition_supplies_session_start_time(
        session_folder_path=session_folder_path, acquisition_format=acquisition_format
    ):
        scalars["session_start_time"] = SUPPLIED_SESSION_START_TIME.isoformat()

    dump_yaml(
        metadata=build_metadata_dict(devices=devices, channel_rows=channel_rows, scalars=scalars, channels=channels),
        path=path,
    )
    return str(path)

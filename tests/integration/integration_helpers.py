"""Integration-suite-local shared constants and helpers.

Split out of ``conftest.py`` so the handful of integration test modules that need these symbols
at import time (``test_integration_step2.py``, ``test_integration_dandi.py``) can import them
directly via a package-relative import, instead of relying on the ambiguous bare ``conftest`` name.
"""

import glob
import os

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
        "npm_split_events": [True, True],
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
        "npm_split_events": [True, True],
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
        "npm_split_events": [True, True],
    },
    "pyphotometry": {
        # Strobed two-input recording: the analog inputs are one tick of the 260 Hz sampling timer
        # apart, which is what the pipeline has to carry through unchanged. This session exists
        # only to feed the pipeline: it is three minutes long so the default -10 to +20 s PSTH
        # window fits around its events, which none of the mode-coverage fixtures is long enough
        # to do.
        #
        # Its mode reads a detector per excitation source, so the two stores are two fibers rather
        # than a signal-plus-isosbestic pair on one. Labeling them as a pair is a convenience for
        # the fixture: that mode has no isosbestic channel, and the pipeline needs a signal and a
        # control to run.
        "session_subdir": "pyphotometry/full_pipeline_session",
        "store_id_to_store_label": {
            "detector_2_excitation_2": "control_region",
            "detector_1_excitation_1": "signal_region",
            "digital_2": "ttl",
        },
        "npm_timestamp_column_name": None,
        "npm_time_unit": None,
        "npm_split_events": [True, True],
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
        "npm_split_events": [True, True],
    },
}


def _locate_output_directory(*, session_copy: str) -> str:
    session_name = os.path.basename(session_copy)
    output_directories = sorted(glob.glob(os.path.join(session_copy, f"{session_name}_output_*")))
    assert output_directories, f"No output directories found in {session_copy}"

    for output_directory in output_directories:
        if os.path.exists(os.path.join(output_directory, "storesList.csv")):
            return output_directory

    raise AssertionError(f"No storesList.csv found in any output directory under {session_copy}")

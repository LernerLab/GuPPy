"""Summarize the contents of the stubbed testing data directory.

Run from the project root:
    python scripts/summarize_stubbed_testing_data.py

For each stubbed session, reports:
  - Total folder size (bytes)
  - Stub duration (seconds)
  - TTL channel name
  - Number of TTL events
"""

import shutil
import tempfile
from pathlib import Path

from guppy.extractors.csv_recording_extractor import CsvRecordingExtractor
from guppy.extractors.doric_recording_extractor import DoricRecordingExtractor
from guppy.extractors.npm_recording_extractor import NpmRecordingExtractor
from guppy.extractors.tdt_recording_extractor import TdtRecordingExtractor

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
STUBBED_TESTING_DATA = PROJECT_ROOT / "stubbed_testing_data"


def _sessions():
    tdt = STUBBED_TESTING_DATA / "tdt"
    doric = STUBBED_TESTING_DATA / "doric"
    npm = STUBBED_TESTING_DATA / "npm"
    csv = STUBBED_TESTING_DATA / "csv"

    return [
        # --- TDT ---
        {
            "modality": "TDT",
            "name": "Photo_63_207-181030-103332",
            "folder_path": tdt / "Photo_63_207-181030-103332",
            "extractor_class": TdtRecordingExtractor,
            "constructor_kwargs": {},
            "control_event": "Dv1A",
            "ttl_event": "PrtN",
            "discover_kwargs": {},
        },
        {
            "modality": "TDT",
            "name": "Photometry-161823",
            "folder_path": tdt / "Photometry-161823",
            "extractor_class": TdtRecordingExtractor,
            "constructor_kwargs": {},
            "control_event": "405R",
            "ttl_event": "PAB/",
            "discover_kwargs": {},
        },
        {
            "modality": "TDT",
            "name": "Photo_048_392-200728-121222",
            "folder_path": tdt / "Photo_048_392-200728-121222",
            "extractor_class": TdtRecordingExtractor,
            "constructor_kwargs": {},
            "control_event": "Dv1A",
            "ttl_event": "PrtN",
            "discover_kwargs": {},
        },
        # --- Doric ---
        {
            "modality": "Doric",
            "name": "sample_doric_1",
            "folder_path": doric / "sample_doric_1",
            "extractor_class": DoricRecordingExtractor,
            "constructor_kwargs": {
                "event_name_to_event_type": {
                    "AIn-1 - Raw": "control",
                    "AIn-2 - Raw": "signal",
                    "DI--O-1": "ttl",
                }
            },
            "control_event": "AIn-1 - Raw",
            "ttl_event": "DI--O-1",
            "discover_kwargs": {},
        },
        {
            "modality": "Doric",
            "name": "sample_doric_2",
            "folder_path": doric / "sample_doric_2",
            "extractor_class": DoricRecordingExtractor,
            "constructor_kwargs": {
                "event_name_to_event_type": {
                    "AIn-1 - Dem (ref)": "control",
                    "AIn-1 - Dem (da)": "signal",
                    "DI/O-1": "ttl",
                }
            },
            "control_event": "AIn-1 - Dem (ref)",
            "ttl_event": "DI/O-1",
            "discover_kwargs": {},
        },
        {
            "modality": "Doric",
            "name": "sample_doric_3",
            "folder_path": doric / "sample_doric_3",
            "extractor_class": DoricRecordingExtractor,
            "constructor_kwargs": {
                "event_name_to_event_type": {
                    "CAM1_EXC1/ROI01": "control",
                    "CAM1_EXC2/ROI01": "signal",
                    "DigitalIO/CAM1": "ttl",
                }
            },
            "control_event": "CAM1_EXC1/ROI01",
            "ttl_event": "DigitalIO/CAM1",
            "discover_kwargs": {},
        },
        {
            "modality": "Doric",
            "name": "sample_doric_4",
            "folder_path": doric / "sample_doric_4",
            "extractor_class": DoricRecordingExtractor,
            "constructor_kwargs": {
                "event_name_to_event_type": {
                    "Series0001/AIN01xAOUT01-LockIn": "control",
                    "Series0001/AIN01xAOUT02-LockIn": "signal",
                }
            },
            "control_event": "Series0001/AIN01xAOUT01-LockIn",
            "ttl_event": None,
            "discover_kwargs": {},
        },
        {
            "modality": "Doric",
            "name": "sample_doric_5",
            "folder_path": doric / "sample_doric_5",
            "extractor_class": DoricRecordingExtractor,
            "constructor_kwargs": {
                "event_name_to_event_type": {
                    "Series0001/AIN01xAOUT01-LockIn": "control",
                    "Series0001/AIN01xAOUT02-LockIn": "signal",
                }
            },
            "control_event": "Series0001/AIN01xAOUT01-LockIn",
            "ttl_event": None,
            "discover_kwargs": {},
        },
        # --- CSV ---
        {
            "modality": "CSV",
            "name": "sample_data_csv_1",
            "folder_path": csv / "sample_data_csv_1",
            "extractor_class": CsvRecordingExtractor,
            "constructor_kwargs": {},
            "control_event": "Sample_Control_Channel",
            "ttl_event": "Sample_TTL",
            "discover_kwargs": {},
        },
        # --- NPM ---
        {
            "modality": "NPM",
            "name": "sampleData_NPM_1",
            "folder_path": npm / "sampleData_NPM_1",
            "extractor_class": NpmRecordingExtractor,
            "constructor_kwargs": {},
            "control_event": "file0_chod1",
            "ttl_event": "event0",
            "discover_kwargs": {"num_ch": 2, "inputParameters": {}},
        },
        {
            "modality": "NPM",
            "name": "sampleData_NPM_2",
            "folder_path": npm / "sampleData_NPM_2",
            "extractor_class": NpmRecordingExtractor,
            "constructor_kwargs": {},
            "control_event": "file0_chev1",
            "ttl_event": None,
            "discover_kwargs": {"num_ch": 2, "inputParameters": {}},
        },
        {
            "modality": "NPM",
            "name": "sampleData_NPM_3",
            "folder_path": npm / "sampleData_NPM_3",
            "extractor_class": NpmRecordingExtractor,
            "constructor_kwargs": {},
            "control_event": "file0_chod1",
            "ttl_event": "event0",
            "discover_kwargs": {"num_ch": 2, "inputParameters": {}},
        },
        {
            "modality": "NPM",
            "name": "sampleData_NPM_4",
            "folder_path": npm / "sampleData_NPM_4",
            "extractor_class": NpmRecordingExtractor,
            "constructor_kwargs": {},
            "control_event": "file0_chod1",
            "ttl_event": "event0",
            "discover_kwargs": {"num_ch": 2, "inputParameters": {}},
        },
        {
            "modality": "NPM",
            "name": "sampleData_NPM_5",
            "folder_path": npm / "sampleData_NPM_5",
            "extractor_class": NpmRecordingExtractor,
            "constructor_kwargs": {},
            "control_event": "file0_chod1",
            "ttl_event": "event0",
            "discover_kwargs": {"num_ch": 2, "inputParameters": {}},
        },
    ]


def _folder_size_in_bytes(folder_path):
    return sum(path.stat().st_size for path in Path(folder_path).rglob("*") if path.is_file())


def _format_size(size_in_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.1f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.1f} TB"


def _summarize_session(session, working_folder_path):
    original_folder_path = session["folder_path"]
    extractor_class = session["extractor_class"]
    constructor_kwargs = session["constructor_kwargs"]
    control_event = session["control_event"]
    ttl_event = session["ttl_event"]
    discover_kwargs = session["discover_kwargs"]

    # Measure size from the original (uncontaminated) data
    folder_size = _folder_size_in_bytes(original_folder_path)

    if discover_kwargs:
        extractor_class.discover_events_and_flags(working_folder_path, **discover_kwargs)

    extractor = extractor_class(folder_path=str(working_folder_path), **constructor_kwargs)

    control_result = extractor.read(events=[control_event], outputPath="")
    timestamps = control_result[0]["timestamps"]
    duration = float(timestamps[-1] - timestamps[0])

    if ttl_event is not None:
        if hasattr(extractor, "_readtev"):
            # TDT: use _readtev directly to avoid storesList.csv dependency for multi-behavior epocs
            S = extractor._readtev(ttl_event)
            number_of_ttls = len(S["timestamps"])
        else:
            ttl_result = extractor.read(events=[ttl_event], outputPath="")
            number_of_ttls = len(ttl_result[0]["timestamps"])
        ttl_channel = ttl_event
    else:
        number_of_ttls = 0
        ttl_channel = "N/A"

    return {
        "modality": session["modality"],
        "name": session["name"],
        "folder_size": folder_size,
        "duration": duration,
        "ttl_channel": ttl_channel,
        "number_of_ttls": number_of_ttls,
    }


def main():
    rows = []
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_directory_path = Path(temporary_directory)
        for session in _sessions():
            print(f"  Reading {session['modality']:5s} {session['name']} ...", end=" ", flush=True)
            working_folder_path = temporary_directory_path / session["name"]
            shutil.copytree(session["folder_path"], working_folder_path)
            row = _summarize_session(session, working_folder_path)
            rows.append(row)
            print("done")

    print()

    col_modality = "Modality"
    col_name = "Session"
    col_size = "Size"
    col_duration = "Duration (s)"
    col_ttl_channel = "TTL Channel"
    col_number_of_ttls = "# TTLs"

    size_strings = [_format_size(row["folder_size"]) for row in rows]

    modality_width = max(len(col_modality), max(len(row["modality"]) for row in rows))
    name_width = max(len(col_name), max(len(row["name"]) for row in rows))
    size_width = max(len(col_size), max(len(s) for s in size_strings))
    duration_width = max(len(col_duration), max(len(f"{row['duration']:.1f}") for row in rows))
    ttl_channel_width = max(len(col_ttl_channel), max(len(row["ttl_channel"]) for row in rows))
    number_of_ttls_width = max(len(col_number_of_ttls), max(len(str(row["number_of_ttls"])) for row in rows))

    header = (
        col_modality.ljust(modality_width)
        + "  "
        + col_name.ljust(name_width)
        + "  "
        + col_size.rjust(size_width)
        + "  "
        + col_duration.rjust(duration_width)
        + "  "
        + col_ttl_channel.ljust(ttl_channel_width)
        + "  "
        + col_number_of_ttls.rjust(number_of_ttls_width)
    )
    separator = "-" * len(header)
    print(header)
    print(separator)

    for row, size_string in zip(rows, size_strings):
        line = (
            row["modality"].ljust(modality_width)
            + "  "
            + row["name"].ljust(name_width)
            + "  "
            + size_string.rjust(size_width)
            + "  "
            + f"{row['duration']:.1f}".rjust(duration_width)
            + "  "
            + row["ttl_channel"].ljust(ttl_channel_width)
            + "  "
            + str(row["number_of_ttls"]).rjust(number_of_ttls_width)
        )
        print(line)


if __name__ == "__main__":
    main()

"""
Prototype script for streaming NWB fiber photometry data from the DANDI Archive
through the GuPPy analysis pipeline.

Usage
-----
Discover available events in a DANDI NWB file::

    python scripts/dandi_streaming_prototype.py --discover

Run the full pipeline (steps 1-5) on the streamed data::

    python scripts/dandi_streaming_prototype.py --output-dir /path/to/output

Override the default dandiset::

    python scripts/dandi_streaming_prototype.py --dandiset-id 000971 \\
        --asset-path "sub-112-283/sub-112-283_ses-FP-PS-2019-06-20T09-32-04_behavior.nwb" \\
        --output-dir /path/to/output

Requirements
------------
Install optional DANDI dependencies::

    pip install -e ".[dandi]"
"""

import argparse
import json
import logging
import os

from guppy.extractors.dandi_nwb_recording_extractor import (
    DandiNwbRecordingExtractor,
    parse_dandi_uri,
)
from guppy.testing.api import step1, step2, step3, step4, step5

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DANDISET_ID = "000971"
DEFAULT_ASSET_PATH = "sub-112-283/sub-112-283_ses-FP-PS-2019-06-20T09-32-04_behavior.nwb"


def build_dandi_uri(*, dandiset_id, asset_path):
    """Build a DANDI URI from a dandiset ID and asset path."""
    return f"dandi://{dandiset_id}/{asset_path}"


def discover_events(*, dandi_uri):
    """Discover and print available events from a DANDI-hosted NWB file."""
    logger.info(f"Discovering events from {dandi_uri} ...")
    events, flags = DandiNwbRecordingExtractor.discover_events_and_flags(folder_path=dandi_uri)
    print("\nDiscovered events:")
    for event in events:
        print(f"  - {event}")
    if flags:
        print(f"\nFlags: {flags}")
    return events


def run_pipeline(*, dandi_uri, output_directory, storenames_map):
    """
    Run the full GuPPy pipeline (steps 1-5) on streamed DANDI data.

    Parameters
    ----------
    dandi_uri : str
        DANDI URI to the NWB file.
    output_directory : str
        Local directory where outputs will be written.
    storenames_map : dict
        Mapping from raw event names to semantic labels.
    """
    output_directory = os.path.abspath(output_directory)

    # Derive a session name from the DANDI asset path
    _, asset_path = parse_dandi_uri(dandi_uri)
    session_name = os.path.splitext(os.path.basename(asset_path))[0]

    # Create local directory structure
    base_directory = output_directory
    session_directory = os.path.join(base_directory, session_name)
    os.makedirs(session_directory, exist_ok=True)

    logger.info(f"Session directory: {session_directory}")
    logger.info(f"DANDI URI: {dandi_uri}")
    logger.info(f"Storenames map: {json.dumps(storenames_map, indent=2)}")

    # Step 1: Save input parameters
    logger.info("Running Step 1 (Save Input Parameters)...")
    step1(base_dir=base_directory, selected_folders=[session_directory])

    # Step 2: Save storenames (discovers events via streaming)
    logger.info("Running Step 2 (Save Storenames)...")
    step2(
        base_dir=base_directory,
        selected_folders=[session_directory],
        storenames_map=storenames_map,
        dandi_uri=dandi_uri,
    )

    # Step 3: Read raw data (streams from DANDI, saves HDF5 locally)
    logger.info("Running Step 3 (Read Raw Data)...")
    step3(
        base_dir=base_directory,
        selected_folders=[session_directory],
        number_of_cores=1,
        dandi_uri=dandi_uri,
    )

    # Step 4: Preprocess and remove artifacts
    logger.info("Running Step 4 (Preprocess)...")
    step4(
        base_dir=base_directory,
        selected_folders=[session_directory],
    )

    # Step 5: PSTH computation
    logger.info("Running Step 5 (PSTH)...")
    step5(
        base_dir=base_directory,
        selected_folders=[session_directory],
        number_of_cores=1,
    )

    logger.info(f"Pipeline complete. Outputs written to: {session_directory}")


def main():
    parser = argparse.ArgumentParser(description="Stream NWB fiber photometry data from DANDI through GuPPy.")
    parser.add_argument("--dandiset-id", default=DEFAULT_DANDISET_ID, help="Dandiset ID (default: %(default)s)")
    parser.add_argument(
        "--asset-path", default=DEFAULT_ASSET_PATH, help="Asset path within dandiset (default: %(default)s)"
    )
    parser.add_argument("--output-dir", help="Local directory for pipeline outputs")
    parser.add_argument("--discover", action="store_true", help="Only discover events, don't run pipeline")
    parser.add_argument(
        "--storenames-map",
        type=json.loads,
        help='JSON string mapping raw event names to semantic labels, e.g. \'{"event_0": "control_DMS", "event_1": "signal_DMS"}\'',
    )
    arguments = parser.parse_args()

    dandi_uri = build_dandi_uri(dandiset_id=arguments.dandiset_id, asset_path=arguments.asset_path)

    if arguments.discover:
        discover_events(dandi_uri=dandi_uri)
        return

    if arguments.output_dir is None:
        parser.error("--output-dir is required when running the pipeline")

    if arguments.storenames_map is None:
        # Default storenames map for dandiset 000971
        # Users should first run --discover to see available events, then provide their own map
        parser.error(
            "--storenames-map is required. Run with --discover first to see available events, "
            'then provide a JSON mapping, e.g. --storenames-map \'{"event_0": "control_DMS", "event_1": "signal_DMS"}\''
        )

    run_pipeline(
        dandi_uri=dandi_uri,
        output_directory=arguments.output_dir,
        storenames_map=arguments.storenames_map,
    )


if __name__ == "__main__":
    main()

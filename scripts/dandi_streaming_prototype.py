"""
Prototype script for streaming NWB fiber photometry data from the DANDI Archive
through the GuPPy analysis pipeline.

Usage
-----
1. Edit the configuration variables below to match your dandiset and desired output.
2. Set ``DISCOVER_ONLY = True`` to list available events, then set it back to ``False``
   and fill in ``STORENAMES_MAP`` before running the full pipeline.
3. Run::

       python scripts/dandi_streaming_prototype.py
"""

import json
import logging
import os
import shutil

from guppy.extractors.dandi_nwb_recording_extractor import (
    DandiNwbRecordingExtractor,
    parse_dandi_uri,
)
from guppy.testing.api import step1, step2, step3, step4, step5

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — edit these variables before running
# ---------------------------------------------------------------------------

DANDISET_ID = "000971"
ASSET_PATH = "sub-112-283/sub-112-283_ses-FP-PS-2019-06-20T09-32-04_behavior.nwb"

# Local directory where pipeline outputs will be written
OUTPUT_DIRECTORY = "/Users/pauladkisson/Documents/CatalystNeuro/Guppy/dandi_streaming"

# Set to True to only discover and print available events (no pipeline run)
DISCOVER_ONLY = False

# Mapping from raw event names to semantic labels.
# Run with DISCOVER_ONLY = True first to see available event names,
# then fill in this mapping before running the full pipeline.
STORENAMES_MAP = {
    "fiber_photometry_response_series_0": "signal_DMS",
    "fiber_photometry_response_series_1": "control_DMS",
    "fiber_photometry_response_series_2": "signal_DLS",
    "fiber_photometry_response_series_3": "control_DLS",
    "right_nose_poke_times": "right_nose_poke",
    "left_reward_times": "left_reward",
    "left_nose_poke_times": "left_nose_poke",
}

# ---------------------------------------------------------------------------


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

    # Start from a clean output directory for each run.
    base_directory = output_directory
    if os.path.exists(base_directory):
        shutil.rmtree(base_directory)
    os.makedirs(base_directory)

    # Create local directory structure
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
    dandi_uri_map = {session_directory: dandi_uri}
    step2(
        base_dir=base_directory,
        selected_folders=[session_directory],
        storenames_map=storenames_map,
        dandi_uri_map=dandi_uri_map,
    )

    # Step 3: Read raw data (streams from DANDI, saves HDF5 locally)
    logger.info("Running Step 3 (Read Raw Data)...")
    step3(
        base_dir=base_directory,
        selected_folders=[session_directory],
        number_of_cores=1,
        dandi_uri_map=dandi_uri_map,
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
    dandi_uri = build_dandi_uri(dandiset_id=DANDISET_ID, asset_path=ASSET_PATH)

    if DISCOVER_ONLY:
        discover_events(dandi_uri=dandi_uri)
        return

    run_pipeline(
        dandi_uri=dandi_uri,
        output_directory=OUTPUT_DIRECTORY,
        storenames_map=STORENAMES_MAP,
    )


if __name__ == "__main__":
    main()

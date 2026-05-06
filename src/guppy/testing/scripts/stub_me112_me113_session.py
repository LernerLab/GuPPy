"""One-off script: stub the ME112-ME113-260420-114630 session.

Unlike create_stubbed_testing_data.py, this does NOT read from testing_data/ —
the source session is kept local (not uploaded to the shared Google Drive) and
this script is intended to be run once to produce the committed stubbed
artifact under stubbed_testing_data/tdt/.

The source `Widt` TTL store carries float-valued event codes (0.1, 0.2, 0.4,
0.8, 10.0) cycling every 4 s; 25 s captures the full cycle twice over.

Run from the project root:
    python src/guppy/testing/scripts/stub_me112_me113_session.py
"""

from pathlib import Path

from guppy.extractors.tdt_recording_extractor import TdtRecordingExtractor

SOURCE = Path("/Users/pauladkisson/Documents/CatalystNeuro/Guppy/UserData/ME112-ME113-260420-114630")
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
DESTINATION = PROJECT_ROOT / "stubbed_testing_data" / "tdt" / "ME112-ME113-260420-114630"
DURATION_IN_SECONDS = 25.0


def main():
    """Stub the ME112/ME113 TDT session to the configured destination path."""
    extractor = TdtRecordingExtractor(str(SOURCE))
    print(f"Stubbing {SOURCE} → {DESTINATION} ({DURATION_IN_SECONDS}s) ...", end=" ", flush=True)
    extractor.stub(folder_path=DESTINATION, duration_in_seconds=DURATION_IN_SECONDS)
    print("done")


if __name__ == "__main__":
    main()

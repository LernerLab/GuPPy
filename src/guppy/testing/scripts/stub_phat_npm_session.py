"""One-off script: stub the PhAT Sample2 Neurophotometrics session.

Unlike create_stubbed_testing_data.py, this does NOT read from testing_data/ —
the source is third-party sample data from the Donaldson Lab's PhAT toolkit
(MIT licensed), cloned locally, and this script is intended to be run once to
produce the committed stubbed artifact under stubbed_testing_data/npm/.

    https://github.com/donaldsonlab/PhAT
    FiberPho_Main/sample_data/Sample2_NPM_1fiber.csv

This session is the reproducer for issue #337: its header line
``,Timestamp,msTimestamp,,Region0R,Region1G,,,LedState`` carries four blank
header cells alongside two timestamp columns, one of them named ``Timestamp``.

Truncation is done on the raw text lines rather than through
NpmRecordingExtractor.stub(), which round-trips the file through pandas and so
writes the blank header cells back out as the literal text ``Unnamed: 0`` and
friends. That parses identically, but the committed file would no longer show
the blank headers it exists to exercise. The cutoff rule matches stub(): keep
the rows whose Timestamp is within duration_in_seconds of the first one.

Run from the project root:
    python src/guppy/testing/scripts/stub_phat_npm_session.py
"""

import shutil
from pathlib import Path

SOURCE_FOLDER = Path("/Users/pauladkisson/Documents/CatalystNeuro/Guppy/PhAT")
SOURCE_CSV = SOURCE_FOLDER / "FiberPho_Main" / "sample_data" / "Sample2_NPM_1fiber.csv"
SOURCE_LICENSE = SOURCE_FOLDER / "LICENSE"
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
DESTINATION = PROJECT_ROOT / "stubbed_testing_data" / "npm" / "sampleData_NPM_6"
# The session carries no TTL events, so the duration only has to leave enough samples per
# LED state to measure a sampling rate. 16 s matches sampleData_NPM_2, the other no-TTL NPM stub.
DURATION_IN_SECONDS = 16.0
# Column 1 of the header is "Timestamp", in seconds.
TIMESTAMP_COLUMN_INDEX = 1


def main() -> None:
    """Truncate the PhAT Sample2 photometry file into the committed stub folder."""
    DESTINATION.mkdir(parents=True, exist_ok=True)

    lines = SOURCE_CSV.read_text().splitlines()
    header_line, data_lines = lines[0], lines[1:]
    first_timestamp = float(data_lines[0].split(",")[TIMESTAMP_COLUMN_INDEX])
    cutoff = first_timestamp + DURATION_IN_SECONDS
    kept_lines = [line for line in data_lines if float(line.split(",")[TIMESTAMP_COLUMN_INDEX]) <= cutoff]

    destination_csv = DESTINATION / SOURCE_CSV.name
    destination_csv.write_text("\n".join([header_line, *kept_lines]) + "\n")
    shutil.copyfile(SOURCE_LICENSE, DESTINATION / "LICENSE")

    print(f"Stubbed {SOURCE_CSV} → {destination_csv} ({DURATION_IN_SECONDS}s, {len(kept_lines)} rows)")


if __name__ == "__main__":
    main()

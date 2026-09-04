import shutil
from pathlib import Path

import pandas as pd
import pytest

from guppy.extractors.csv_recording_extractor import CsvRecordingExtractor
from guppy.extractors.detect_acquisition_formats import detect_acquisition_formats
from guppy.testing.api import import_custom_events

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUBBED_CSV_SESSION = PROJECT_ROOT / "stubbed_testing_data" / "csv" / "sample_data_csv_1"


@pytest.fixture
def base_dir_with_session(tmp_path):
    base_dir = tmp_path / "data_root"
    base_dir.mkdir()
    session_copy = base_dir / "sample_data_csv_1"
    shutil.copytree(STUBBED_CSV_SESSION, session_copy)
    return str(base_dir), str(session_copy)


def test_pasted_event_surfaces_as_store(base_dir_with_session):
    base_dir, session = base_dir_with_session

    import_custom_events(
        base_dir=base_dir,
        selected_folders=[session],
        custom_events_map={"sample_data_csv_1": {"movement_onset": [0.5, 1.5, 2.5]}},
    )

    # The CSV is written into the session folder in GuPPy-compatible form.
    csv_path = Path(session) / "movement_onset.csv"
    assert Path(csv_path).exists()
    df = pd.read_csv(csv_path)
    assert list(df.columns) == ["timestamps"]
    assert df["timestamps"].tolist() == [0.5, 1.5, 2.5]

    # And it is detected + surfaced as a store named after the file.
    assert "csv" in detect_acquisition_formats(session)
    events, flags = CsvRecordingExtractor.discover_events_and_flags(session)
    assert "movement_onset" in events
    assert flags[events.index("movement_onset")] == "event_csv"

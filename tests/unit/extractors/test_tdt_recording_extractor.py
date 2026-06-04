"""Contract tests for TdtRecordingExtractor."""

import os
import shutil

import numpy as np
import pytest
from conftest import STUBBED_TESTING_DATA

from guppy.extractors.tdt_recording_extractor import TdtRecordingExtractor

from .recording_extractor_test_mixin import RecordingExtractorTestMixin

# ---------------------------------------------------------------------------
# _ismember
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arr, element, expected",
    [
        (["a", "b", "a", "c"], "a", [1, 0, 1, 0]),
        (["x", "y", "z"], "x", [1, 0, 0]),
        (["x", "y", "z"], "w", [0, 0, 0]),
        ([], "a", []),
    ],
)
def test_ismember_returns_indicator_array(arr, element, expected):
    result = TdtRecordingExtractor._ismember(arr, element)
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# _event_needs_splitting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data, sampling_rate, expected",
    [
        (np.array([1.0, 2.0, 3.0]), 0, False),  # uniform diffs at sampling_rate=0 → no split
        (np.array([1.0, 2.0, 5.0]), 0, True),  # non-uniform diffs at sampling_rate=0 → split
        (np.array([1.0, 2.0, 5.0]), 250.0, False),  # nonzero sampling_rate → never split
        (np.array([1.0]), 0, False),  # single element → no diff → no split
        (np.array([]), 0, False),  # empty → no diff → no split
    ],
)
def test_event_needs_splitting(data, sampling_rate, expected):
    assert TdtRecordingExtractor._event_needs_splitting(data=data, sampling_rate=sampling_rate) == expected


# ---------------------------------------------------------------------------
# _format_split_suffix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (5, "5"),
        (np.int32(3), "3"),
        (5.0, "5"),  # integer-valued float → no "p"
        (0.1, "0p1"),
        (1.6, "1p6"),
        (10.0, "10"),
        (np.float32(0.1), "0p1"),  # float32 precision artifact collapsed by "{:g}"
    ],
)
def test_format_split_suffix(value, expected):
    assert TdtRecordingExtractor._format_split_suffix(value) == expected


# ---------------------------------------------------------------------------
# Error paths in _readtsq / _readtev / read
# ---------------------------------------------------------------------------


def test_readtsq_raises_for_multiple_tsq_files(tmp_path):
    (tmp_path / "session_a.tsq").write_bytes(b"")
    (tmp_path / "session_b.tsq").write_bytes(b"")
    with pytest.raises(ValueError, match=r"Multiple .tsq files"):
        TdtRecordingExtractor._readtsq(str(tmp_path))


def test_readtsq_returns_zeros_when_no_tsq_present(tmp_path):
    """No-tsq path should return (0, 0) without raising — used by mixed-modality detection."""
    header_df, flag = TdtRecordingExtractor._readtsq(str(tmp_path))
    assert header_df == 0
    assert flag == 0


def test_readtev_raises_for_multiple_tev_files(tmp_path):
    """A folder with one valid tank's tsq plus an extra .tev triggers the multi-tev guard."""
    source = os.path.join(STUBBED_TESTING_DATA, "tdt", "Photo_63_207-181030-103332")
    for filename in os.listdir(source):
        shutil.copy(os.path.join(source, filename), tmp_path / filename)
    # Add a second .tev so the read path detects duplicates.
    (tmp_path / "extra.tev").write_bytes(b"")
    extractor = TdtRecordingExtractor(str(tmp_path))
    with pytest.raises(ValueError, match=r"Multiple .tev files"):
        extractor._readtev("Dv1A")


def test_readtev_raises_when_store_name_not_present(tmp_path):
    """A bogus store name surfaces a ValueError listing the available stores."""
    extractor = TdtRecordingExtractor(os.path.join(STUBBED_TESTING_DATA, "tdt", "Photo_63_207-181030-103332"))
    with pytest.raises(ValueError, match=r"'BOGUS' not found.*Available stores"):
        extractor._readtev("BOGUS")


# ---------------------------------------------------------------------------
# Shared fixtures for all TDT test classes
# ---------------------------------------------------------------------------


class TdtRecordingExtractorTestMixin(RecordingExtractorTestMixin):
    """Provides ``expected_*`` fixtures for all TDT extractor test classes.

    TDT's ``read()`` writes intermediate files, so ``outputPath=str(tmp_path)``
    is used throughout. TTL fixtures return ``None`` when ``ttl_event is None``.
    """

    @pytest.fixture
    def expected_control_timestamps(self, tmp_path):
        result = self.extractor_instance.read(events=[self.control_event], outputPath=str(tmp_path))
        return result[0]["timestamps"]

    @pytest.fixture
    def expected_control_data(self, tmp_path):
        result = self.extractor_instance.read(events=[self.control_event], outputPath=str(tmp_path))
        return result[0]["data"]

    @pytest.fixture
    def expected_signal_timestamps(self, tmp_path):
        result = self.extractor_instance.read(events=[self.signal_event], outputPath=str(tmp_path))
        return result[0]["timestamps"]

    @pytest.fixture
    def expected_signal_data(self, tmp_path):
        result = self.extractor_instance.read(events=[self.signal_event], outputPath=str(tmp_path))
        return result[0]["data"]

    @pytest.fixture
    def expected_ttl_timestamps(self, tmp_path):
        if self.ttl_event is None:
            return None
        result = self.extractor_instance.read(events=[self.ttl_event], outputPath=str(tmp_path))
        return result[0]["timestamps"]


# ---------------------------------------------------------------------------
# Contract test classes
# ---------------------------------------------------------------------------


class TestTdtRecordingExtractor(TdtRecordingExtractorTestMixin):
    extractor_class = TdtRecordingExtractor
    folder_path = os.path.join(STUBBED_TESTING_DATA, "tdt", "Photo_63_207-181030-103332")
    extractor_instance = TdtRecordingExtractor(folder_path)
    expected_events = ["Dv1A", "Dv2A", "PrtN"]
    discover_kwargs = {}
    control_event = "Dv1A"
    signal_event = "Dv2A"
    ttl_event = "PrtN"
    stub_ttl_test_duration_in_seconds = 100.0


class TestTdtRecordingExtractorSample2(TdtRecordingExtractorTestMixin):
    extractor_class = TdtRecordingExtractor
    folder_path = os.path.join(STUBBED_TESTING_DATA, "tdt", "Photo_048_392-200728-121222")
    extractor_instance = TdtRecordingExtractor(folder_path)
    expected_events = ["Dv1A", "Dv2A", "PrtN"]
    discover_kwargs = {}
    control_event = "Dv1A"
    signal_event = "Dv2A"
    ttl_event = "PrtN"
    stub_ttl_test_duration_in_seconds = 100.0


class TdtRecordingExtractorSplitTestMixin(TdtRecordingExtractorTestMixin):
    """Generic tests for TDT epoc stores that split into per-marker-value sub-events.

    Subclasses set ``split_parent`` (the parent store name) and ``expected_split_events``
    (the sub-event names discover should enumerate). Splits are now determined at discover
    time, so ``read()`` is a pure lookup that writes nothing.
    """

    def test_discover_enumerates_split_subevents(self):
        events, _ = self.extractor_class.discover_events_and_flags(self.folder_path)
        for split_event in self.expected_split_events:
            assert split_event in events
        # The parent store is replaced by its sub-events and is not itself a discovered event.
        assert self.split_parent not in events

    def test_discover_split_subevents_have_no_duplicates(self):
        events, _ = self.extractor_class.discover_events_and_flags(self.folder_path)
        discovered_splits = [event for event in events if event in self.expected_split_events]
        assert sorted(discovered_splits) == sorted(set(self.expected_split_events))

    def test_read_split_subevent_matches_parent_filter(self):
        split_map = self.extractor_class._compute_split_map(self.extractor_instance._header_df)
        parent_dict = self.extractor_instance._readtev(event=self.split_parent)
        for split_name, value in split_map[self.split_parent]:
            result = self.extractor_instance.read(events=[split_name], outputPath="")
            expected_timestamps = parent_dict["timestamps"][np.where(parent_dict["data"] == value)[0]]
            np.testing.assert_array_equal(result[0]["timestamps"], expected_timestamps)
            assert result[0]["storename"] == split_name

    def test_read_split_subevents_writes_nothing(self, tmp_path):
        before = set(os.listdir(tmp_path))
        self.extractor_instance.read(events=list(self.expected_split_events), outputPath=str(tmp_path))
        assert set(os.listdir(tmp_path)) == before

    def test_count_samples_split_subevent_matches_parent_rows(self):
        split_map = self.extractor_class._compute_split_map(self.extractor_instance._header_df)
        names = np.asarray(self.extractor_instance._header_df["name"], dtype=str)
        parent_strobes = np.asarray(self.extractor_instance._header_df["strobe"])[
            np.where(names == self.split_parent)[0]
        ]
        for split_name, value in split_map[self.split_parent]:
            assert self.extractor_instance.count_samples(event=split_name) == int(
                np.count_nonzero(parent_strobes == value)
            )


class TestTdtRecordingExtractorSplitEvent(TdtRecordingExtractorSplitTestMixin):
    extractor_class = TdtRecordingExtractor
    folder_path = os.path.join(STUBBED_TESTING_DATA, "tdt", "Photometry-161823")
    extractor_instance = TdtRecordingExtractor(folder_path)
    # PAB/ splits into one sub-event per marker value, enumerated at discover time.
    split_parent = "PAB/"
    expected_split_events = ["PAB0", "PAB16", "PAB2064"]
    expected_events = ["405R", "490R", "PAB0", "PAB16", "PAB2064"]
    discover_kwargs = {}
    control_event = "405R"
    signal_event = "490R"
    ttl_event = None


class TestTdtRecordingExtractorSplitFloat(TdtRecordingExtractorSplitTestMixin):
    extractor_class = TdtRecordingExtractor
    folder_path = os.path.join(STUBBED_TESTING_DATA, "tdt", "ME112-ME113-260420-114630")
    extractor_instance = TdtRecordingExtractor(folder_path)
    # Widt carries float-valued codes (0.1, 0.2, 0.4, 0.8, 10.0); splitting must produce one
    # sub-event per unique float (regression for the int()-collapse bug), not collapse them.
    split_parent = "Widt"
    expected_split_events = ["Widt0p1", "Widt0p2", "Widt0p4", "Widt0p8", "Widt10"]
    expected_events = ["415A", "465A", "Widt0p1", "Widt0p2", "Widt0p4", "Widt0p8", "Widt10"]
    discover_kwargs = {}
    control_event = "415A"
    signal_event = "465A"
    ttl_event = None

"""Contract tests for TdtRecordingExtractor."""

import csv
import os

import numpy as np
import pytest

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


from conftest import STUBBED_TESTING_DATA

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


class TestTdtRecordingExtractorSplitEvent(TdtRecordingExtractorTestMixin):
    extractor_class = TdtRecordingExtractor
    folder_path = os.path.join(STUBBED_TESTING_DATA, "tdt", "Photometry-161823")
    extractor_instance = TdtRecordingExtractor(folder_path)
    # PAB/ is a split-event channel. read(events=["PAB/"]) internally calls
    # _split_event_storesList, which requires a storesList.csv in outputPath.
    # The _pab_storesList_setup autouse fixture pre-creates that file in tmp_path
    # before every test so that all inherited mixin tests work without modification.
    expected_events = ["405R", "490R", "PAB/"]
    discover_kwargs = {}
    control_event = "405R"
    signal_event = "490R"
    ttl_event = "PAB/"
    stub_ttl_test_duration_in_seconds = 200.0

    @pytest.fixture(autouse=True)
    def _pab_storesList_setup(self, tmp_path):
        """Write a minimal storesList.csv and return tmp_path for reuse.

        Returning tmp_path lets other fixtures depend on this one to both
        guarantee ordering and receive the prepared output path.
        """
        with open(tmp_path / "storesList.csv", "w", newline="") as stores_file:
            csv.writer(stores_file).writerows([["PAB/"], ["ttl"]])
        return tmp_path

    @pytest.fixture
    def expected_ttl_timestamps(self, _pab_storesList_setup):
        result = self.extractor_instance.read(events=["PAB/"], outputPath=str(_pab_storesList_setup))
        return result[0]["timestamps"]

    def test_split_event_produces_sub_events(self, tmp_path):
        # PAB/ carries non-uniform event codes → _event_needs_splitting returns True.
        # _pab_storesList_setup (autouse) has already written storesList.csv.
        result = self.extractor_instance.read(events=["PAB/"], outputPath=str(tmp_path))

        # The first dict is the original unsplit PAB/ event; subsequent dicts are
        # the per-code sub-events (e.g. PAB0, PAB16, PAB2064).
        assert len(result) > 1
        storenames = [output_dict["storename"] for output_dict in result]
        assert "PAB/" in storenames
        # All sub-event storenames should start with "PAB" (slash stripped).
        sub_event_storenames = [name for name in storenames if name != "PAB/"]
        assert all(name.startswith("PAB") for name in sub_event_storenames)


class TestTdtRecordingExtractorSplitFloat(TdtRecordingExtractorTestMixin):
    extractor_class = TdtRecordingExtractor
    folder_path = os.path.join(STUBBED_TESTING_DATA, "tdt", "ME112-ME113-260420-114630")
    extractor_instance = TdtRecordingExtractor(folder_path)
    # Widt is a split-event channel whose codes are floats (0.1, 0.2, 0.4, 0.8, 10.0)
    # — regression case for the int()-collapse bug. _widt_storesList_setup autouse
    # fixture prepares storesList.csv exactly like the PAB/ case above.
    expected_events = ["415A", "465A", "Widt"]
    discover_kwargs = {}
    control_event = "415A"
    signal_event = "465A"
    ttl_event = "Widt"
    stub_ttl_test_duration_in_seconds = 15.0

    @pytest.fixture(autouse=True)
    def _widt_storesList_setup(self, tmp_path):
        with open(tmp_path / "storesList.csv", "w", newline="") as stores_file:
            csv.writer(stores_file).writerows([["Widt"], ["ttl"]])
        return tmp_path

    @pytest.fixture
    def expected_ttl_timestamps(self, _widt_storesList_setup):
        result = self.extractor_instance.read(events=["Widt"], outputPath=str(_widt_storesList_setup))
        return result[0]["timestamps"]

    def test_split_event_preserves_unique_floats(self, _widt_storesList_setup):
        # Widt carries float-valued codes (0.1, 0.2, 0.4, 0.8, 10.0) — splitting
        # must produce one sub-event per unique float, not collapse via int() to
        # four "Widt0" + one "Widt10".
        result = self.extractor_instance.read(events=["Widt"], outputPath=str(_widt_storesList_setup))

        storenames = [output_dict["storename"] for output_dict in result]
        sub_event_storenames = sorted(name for name in storenames if name != "Widt")
        assert sub_event_storenames == ["Widt0p1", "Widt0p2", "Widt0p4", "Widt0p8", "Widt10"]

    def test_split_event_storesList_has_no_duplicates(self, _widt_storesList_setup):
        self.extractor_instance.read(events=["Widt"], outputPath=str(_widt_storesList_setup))
        stores_list = np.genfromtxt(str(_widt_storesList_setup / "storesList.csv"), dtype="str", delimiter=",").reshape(
            2, -1
        )

        assert sorted(stores_list[0]) == ["Widt0p1", "Widt0p2", "Widt0p4", "Widt0p8", "Widt10"]
        assert sorted(stores_list[1]) == ["Widt_0p1", "Widt_0p2", "Widt_0p4", "Widt_0p8", "Widt_10"]
        assert "Widt" not in stores_list[0]

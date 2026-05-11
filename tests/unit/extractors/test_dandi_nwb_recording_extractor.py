"""Tests for DandiNwbRecordingExtractor and its helper functions.

The extractor streams NWB files from the DANDI Archive in production. For offline
unit testing, ``_stream_nwb`` is monkeypatched to open local mock NWB files from
``stubbed_testing_data/nwb/``. This lets us exercise the full
BaseRecordingExtractor contract (via ``NwbRecordingExtractorTestMixin``) and the
DANDI-specific URI parsing / IO-lifetime wiring without any network access.
"""

import numpy as np
import pytest
from conftest import STUBBED_TESTING_DATA
from pynwb import NWBHDF5IO

from guppy.extractors import dandi_nwb_recording_extractor as dandi_module
from guppy.extractors.dandi_nwb_recording_extractor import (
    DANDI_URI_PREFIX,
    DandiNwbRecordingExtractor,
    is_dandi_uri,
    parse_dandi_uri,
)

from .test_nwb_recording_extractor import NwbRecordingExtractorTestMixin

MOCK_NWB_FOLDER = STUBBED_TESTING_DATA / "nwb" / "mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2"
MOCK_NWB_FILE = MOCK_NWB_FOLDER / "mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2.nwb"


# ---------------------------------------------------------------------------
# Helper: make a _stream_nwb replacement that opens a local mock file
# ---------------------------------------------------------------------------


class _StreamNwbSpy:
    """Replacement for ``_stream_nwb`` that opens a local NWB file and records calls.

    Records each call's kwargs so tests can assert on them, and tracks whether
    ``io.close()`` is invoked by wrapping the returned IO object's ``close``.
    """

    def __init__(self, local_nwb_file):
        self.local_nwb_file = str(local_nwb_file)
        self.calls = []
        self.close_count = 0

    def __call__(self, *, dandiset_id, asset_path):
        self.calls.append({"dandiset_id": dandiset_id, "asset_path": asset_path})
        io = NWBHDF5IO(self.local_nwb_file, "r", load_namespaces=True)
        nwbfile = io.read()

        original_close = io.close
        spy = self

        def tracked_close(*args, **kwargs):
            spy.close_count += 1
            return original_close(*args, **kwargs)

        io.close = tracked_close
        import io as io_module

        from guppy.extractors.dandi_nwb_recording_extractor import _CountingRemfile

        return nwbfile, io, _CountingRemfile(io_module.BytesIO())


# ---------------------------------------------------------------------------
# Pure-helper tests
# ---------------------------------------------------------------------------


class TestIsDandiUri:
    def test_valid_uri(self):
        assert is_dandi_uri("dandi://000971/sub-01/file.nwb") is True

    def test_local_path(self):
        assert is_dandi_uri("/home/user/data/session1") is False

    def test_empty_string(self):
        assert is_dandi_uri("") is False

    def test_none(self):
        assert is_dandi_uri(None) is False

    def test_prefix_only(self):
        assert is_dandi_uri("dandi://") is True

    def test_non_string(self):
        assert is_dandi_uri(12345) is False


class TestParseDandiUri:
    def test_simple_uri(self):
        dandiset_id, asset_path = parse_dandi_uri("dandi://000971/sub-01/file.nwb")
        assert dandiset_id == "000971"
        assert asset_path == "sub-01/file.nwb"

    def test_nested_asset_path(self):
        dandiset_id, asset_path = parse_dandi_uri(
            "dandi://000971/sub-112-283/sub-112-283_ses-FP-PS-2019-06-20T09-32-04_behavior.nwb"
        )
        assert dandiset_id == "000971"
        assert asset_path == "sub-112-283/sub-112-283_ses-FP-PS-2019-06-20T09-32-04_behavior.nwb"

    def test_prefix_constant(self):
        assert DANDI_URI_PREFIX == "dandi://"

    def test_uri_without_asset_path_raises(self):
        # Locks in current behavior: a URI with no asset path component fails loudly.
        with pytest.raises(IndexError):
            parse_dandi_uri("dandi://000971")


# ---------------------------------------------------------------------------
# DANDI-specific wiring tests: URI parsing, _stream_nwb invocation, io.close
# ---------------------------------------------------------------------------


class TestDandiStreamingWiring:
    """Exercises the DANDI-specific plumbing around the shared NWB helpers."""

    DANDI_URI = "dandi://000971/sub-01/file.nwb"

    @pytest.fixture
    def stream_spy(self, monkeypatch):
        spy = _StreamNwbSpy(MOCK_NWB_FILE)
        monkeypatch.setattr(dandi_module, "_stream_nwb", spy)
        return spy

    def test_init_stores_folder_path_without_streaming(self, monkeypatch):
        calls = []

        def fail_if_called(**kwargs):
            calls.append(kwargs)
            raise AssertionError("_stream_nwb should not be called from __init__")

        monkeypatch.setattr(dandi_module, "_stream_nwb", fail_if_called)
        extractor = DandiNwbRecordingExtractor(folder_path=self.DANDI_URI)
        assert extractor.folder_path == self.DANDI_URI
        assert calls == []

    def test_discover_parses_uri_and_passes_to_stream_nwb(self, stream_spy):
        DandiNwbRecordingExtractor.discover_events_and_flags(self.DANDI_URI)
        assert stream_spy.calls == [{"dandiset_id": "000971", "asset_path": "sub-01/file.nwb"}]

    def test_discover_closes_io(self, stream_spy):
        DandiNwbRecordingExtractor.discover_events_and_flags(self.DANDI_URI)
        assert stream_spy.close_count == 1

    def test_discover_returns_flags_empty_list(self, stream_spy):
        _, flags = DandiNwbRecordingExtractor.discover_events_and_flags(self.DANDI_URI)
        assert flags == []

    def test_read_parses_uri_and_passes_to_stream_nwb(self, stream_spy, tmp_path):
        extractor = DandiNwbRecordingExtractor(folder_path=self.DANDI_URI)
        extractor.read(events=["fiber_photometry_response_series_0"], outputPath=str(tmp_path))
        assert stream_spy.calls == [{"dandiset_id": "000971", "asset_path": "sub-01/file.nwb"}]

    def test_read_closes_io(self, stream_spy, tmp_path):
        extractor = DandiNwbRecordingExtractor(folder_path=self.DANDI_URI)
        extractor.read(events=["fiber_photometry_response_series_0"], outputPath=str(tmp_path))
        assert stream_spy.close_count == 1

    def test_stub_raises_not_implemented(self, tmp_path):
        extractor = DandiNwbRecordingExtractor(folder_path=self.DANDI_URI)
        with pytest.raises(NotImplementedError, match="Stub method is not supported"):
            extractor.stub(folder_path=tmp_path / "stubbed")


# ---------------------------------------------------------------------------
# Contract tests via NwbRecordingExtractorTestMixin
#
# Each subclass patches _stream_nwb to open one of the local mock NWB files,
# then runs the full ~14-test NWB contract against DandiNwbRecordingExtractor.
# ---------------------------------------------------------------------------


class DandiNwbContractMixin(NwbRecordingExtractorTestMixin):
    """Shared base for DANDI contract-test classes.

    Subclasses must define ``local_nwb_file`` (the real .nwb path that the
    patched ``_stream_nwb`` should open) and the ``folder_path`` sentinel URI.
    """

    extractor_class = DandiNwbRecordingExtractor
    local_nwb_file = None  # set by subclass

    @pytest.fixture(autouse=True)
    def _patch_stream_nwb(self, monkeypatch):
        spy = _StreamNwbSpy(self.local_nwb_file)
        monkeypatch.setattr(dandi_module, "_stream_nwb", spy)
        yield spy

    @pytest.fixture
    def isolated_folder_path(self):
        # Override base-mixin default which wraps folder_path in Path(),
        # collapsing ``dandi://`` to ``dandi:/``. The DANDI extractor needs
        # the raw URI string.
        return self.folder_path


class TestDandiNwbRecordingExtractorEvents(DandiNwbContractMixin):
    """Contract tests using ndx-events v0.2 ``Events`` as the TTL channel.

    Runs the full ~14-test NWB contract against one mock NWB variant. The other
    NWB-variant combinations (ndx-fiber-photometry v0.1, ndx-events v0.4, other
    TTL types) are already exhaustively covered by ``TestNwbRecordingExtractor*``
    in ``test_nwb_recording_extractor.py`` — the DANDI extractor only adds URI
    parsing and ``_stream_nwb`` wiring on top of the same shared helpers, so one
    variant is sufficient here.
    """

    folder_path = "dandi://mock/events.nwb"
    file_path = str(MOCK_NWB_FILE)
    local_nwb_file = MOCK_NWB_FILE
    extractor_instance = DandiNwbRecordingExtractor(folder_path="dandi://mock/events.nwb")
    ttl_event = "events"

    @pytest.fixture
    def expected_ttl_timestamps(self):
        return np.arange(45, 55, dtype=np.float64)

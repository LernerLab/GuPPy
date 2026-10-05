"""Tests for deciding which DANDI assets and dandisets hold photometry GuPPy can read.

The reads happen over HTTP byte ranges, so they run against a local server that answers ranges
the way the archive does, serving the real mock NWB files in ``stubbed_testing_data/nwb/``. Real
sockets, real ``Range`` headers, real h5py -- only the address is local. The same contract runs
against the archive in ``test_dandi_filter_live.py``.
"""

import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import h5py
import numpy as np
import pytest

from guppy.utils.dandi_filter import (
    DandisetReference,
    PhotometryVerdictCache,
    PrefetchedRemoteFile,
    asset_holds_photometry,
    filter_assets,
    order_for_verification,
    scan_assets_for_photometry,
    scan_order,
    verify_dandisets,
)
from guppy.utils.dandi_search import AssetSummary
from guppy_test_data import STUBBED_TESTING_DATA

from .dandi_filter_test_mixin import DandiFilterTestMixin

NWB_DATA = STUBBED_TESTING_DATA / "nwb"
MOCK_NWB_FILES = {
    name: NWB_DATA / name / f"{name}.nwb"
    for name in (
        "mock_nwbfile_ndx_fiber_photometry_v0_1_ndx_events_v0_2",
        "mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2",
        "mock_nwbfile_ndx_fiber_photometry_v0_2_core_events",
    )
}

# Offsets of the markers planted in the payload the reader fixture serves, chosen to sit in the
# prefetched head, in the gap between the windows, and in the prefetched tail.
PAYLOAD_SIZE = 400_000
HEAD_MARKER_OFFSET = 0
GAP_MARKER_OFFSET = 100_000
TAIL_MARKER_OFFSET = PAYLOAD_SIZE - 11  # the last bytes of the file, inside the tail window


class RangeRequestHandler(BaseHTTPRequestHandler):
    """Serve a directory over HTTP, honoring the byte ranges the prefetching reader asks for."""

    def do_GET(self) -> None:
        payload = (Path(self.server.served_directory) / self.path.lstrip("/")).read_bytes()
        range_header = self.headers.get("Range")
        if range_header is None:
            body, status = payload, 200
        else:
            first, last = range_header.removeprefix("bytes=").split("-")
            body, status = payload[int(first) : int(last) + 1], 206
            self.server.requested_ranges.append((int(first), int(last)))
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        """Keep the test output free of one access log line per range request."""


@pytest.fixture
def byte_server(tmp_path):
    """Serve ``tmp_path`` over HTTP, yielding the base URL its files are readable from."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), RangeRequestHandler)
    server.served_directory = str(tmp_path)
    server.requested_ranges = []
    Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


def served_url(server, name):
    """Return the URL ``name`` is served from."""
    return f"http://127.0.0.1:{server.server_port}/{name}"


class TestPrefetchedRemoteFile:
    @pytest.fixture
    def payload(self, tmp_path):
        """A file larger than both prefetch windows, marked in the head, the gap and the tail."""
        data = bytearray(b"." * PAYLOAD_SIZE)
        data[HEAD_MARKER_OFFSET : HEAD_MARKER_OFFSET + 11] = b"HEAD-MARKER"
        data[GAP_MARKER_OFFSET : GAP_MARKER_OFFSET + 10] = b"GAP-MARKER"
        data[TAIL_MARKER_OFFSET : TAIL_MARKER_OFFSET + 11] = b"TAIL-MARKER"
        (tmp_path / "large.bin").write_bytes(bytes(data))
        return bytes(data)

    @pytest.fixture
    def reader(self, byte_server, payload):
        return PrefetchedRemoteFile(content_url=served_url(byte_server, "large.bin"), size_in_bytes=PAYLOAD_SIZE)

    def test_a_large_file_is_prefetched_as_a_head_and_a_tail(self, byte_server, reader):
        # The two windows are fetched in parallel, so they arrive in either order.
        assert sorted(byte_server.requested_ranges) == [(0, 65535), (137856, 399999)]

    def test_reads_from_the_head_window_cost_no_request(self, byte_server, reader):
        byte_server.requested_ranges.clear()
        assert reader.read(11) == b"HEAD-MARKER"
        assert byte_server.requested_ranges == []
        assert reader.gap_read_count == 0

    def test_reads_from_the_tail_window_cost_no_request(self, byte_server, reader):
        byte_server.requested_ranges.clear()
        reader.seek(TAIL_MARKER_OFFSET)
        assert reader.read(11) == b"TAIL-MARKER"
        assert byte_server.requested_ranges == []
        assert reader.gap_read_count == 0

    def test_a_read_between_the_windows_falls_back_to_its_own_request(self, byte_server, reader):
        byte_server.requested_ranges.clear()
        reader.seek(GAP_MARKER_OFFSET)
        assert reader.read(10) == b"GAP-MARKER"
        assert byte_server.requested_ranges == [(100_000, 100_009)]
        assert reader.gap_read_count == 1

    def test_a_small_file_is_fetched_whole_in_one_request(self, byte_server, tmp_path):
        (tmp_path / "small.bin").write_bytes(b"SMALL-PAYLOAD")
        reader = PrefetchedRemoteFile(content_url=served_url(byte_server, "small.bin"), size_in_bytes=13)
        assert byte_server.requested_ranges == [(0, 12)]
        assert reader.read() == b"SMALL-PAYLOAD"
        assert reader.gap_read_count == 0

    def test_seeking_reports_where_it_landed(self, reader):
        assert reader.seek(50) == 50
        assert reader.tell() == 50
        assert reader.seek(10, io.SEEK_CUR) == 60
        assert reader.seek(-11, io.SEEK_END) == PAYLOAD_SIZE - 11
        assert reader.read(11) == b"TAIL-MARKER"

    def test_a_read_past_the_end_stops_at_the_end(self, reader):
        reader.seek(PAYLOAD_SIZE - 11)
        assert reader.read(500) == b"TAIL-MARKER"
        assert reader.tell() == PAYLOAD_SIZE


class TestAssetHoldsPhotometry:
    def _asset(self, server, name, path):
        return AssetSummary(
            asset_id=name,
            path=name,
            size_in_bytes=path.stat().st_size,
            content_url=served_url(server, name),
        )

    @pytest.mark.parametrize("mock_name", sorted(MOCK_NWB_FILES))
    def test_a_photometry_file_is_recognized(self, byte_server, tmp_path, mock_name):
        served = tmp_path / "photometry.nwb"
        served.write_bytes(MOCK_NWB_FILES[mock_name].read_bytes())
        assert asset_holds_photometry(self._asset(byte_server, "photometry.nwb", served)) is True

    def test_the_container_is_found_under_whatever_name_its_author_gave_it(self, byte_server, tmp_path):
        # Writers name the FiberPhotometry container themselves: neuroconv writes
        # "fiber_photometry", dandiset 001038 writes "FiberPhotometry".
        path = tmp_path / "camel_case.nwb"
        with h5py.File(path, "w") as file:
            container = file.create_group("general/FiberPhotometry")
            container.attrs["neurodata_type"] = "FiberPhotometry"
            series = file.create_group("acquisition/Traces")
            series.attrs["neurodata_type"] = "FiberPhotometryResponseSeries"
        assert asset_holds_photometry(self._asset(byte_server, "camel_case.nwb", path)) is True

    def test_a_response_series_in_a_processing_module_counts(self, byte_server, tmp_path):
        path = tmp_path / "processed.nwb"
        with h5py.File(path, "w") as file:
            container = file.create_group("general/fiber_photometry")
            container.attrs["neurodata_type"] = "FiberPhotometry"
            series = file.create_group("processing/ophys/Traces")
            series.attrs["neurodata_type"] = "FiberPhotometryResponseSeries"
        assert asset_holds_photometry(self._asset(byte_server, "processed.nwb", path)) is True

    def test_the_container_without_a_response_series_is_not_photometry(self, byte_server, tmp_path):
        # Dandiset 000689 writes the extension's metadata table but stores its traces as
        # RoiResponseSeries, which GuPPy's reader does not pick up.
        path = tmp_path / "roi_series.nwb"
        with h5py.File(path, "w") as file:
            container = file.create_group("general/fiber_photometry")
            container.attrs["neurodata_type"] = "FiberPhotometry"
            series = file.create_group("acquisition/RoiResponseSeriesRegion0G")
            series.attrs["neurodata_type"] = "RoiResponseSeries"
        assert asset_holds_photometry(self._asset(byte_server, "roi_series.nwb", path)) is False

    def test_a_group_named_like_the_container_but_untyped_is_not_one(self, byte_server, tmp_path):
        path = tmp_path / "untyped.nwb"
        with h5py.File(path, "w") as file:
            file.create_group("general/fiber_photometry")
        assert asset_holds_photometry(self._asset(byte_server, "untyped.nwb", path)) is False

    def test_an_unreadable_asset_has_no_verdict_rather_than_a_negative_one(self, byte_server):
        # A dropped read reported as False would be indistinguishable from a real answer.
        asset = AssetSummary(
            asset_id="missing",
            path="missing.nwb",
            size_in_bytes=1024,
            content_url=served_url(byte_server, "missing.nwb"),
        )
        assert asset_holds_photometry(asset) is None


class TestScanAssetsForPhotometry:
    def test_scanning_nothing_asks_the_archive_nothing(self, byte_server):
        assert scan_assets_for_photometry([]) == {}
        assert byte_server.requested_ranges == []


# ---------------------------------------------------------------------------------------------
# Verification layer: which dandisets hold photometry
# ---------------------------------------------------------------------------------------------


class TestPhotometryVerdictCache:
    @pytest.fixture
    def assets(self):
        return [
            AssetSummary(asset_id="a", path="one.nwb", size_in_bytes=1, content_url="u"),
            AssetSummary(asset_id="b", path="two.nwb", size_in_bytes=1, content_url="u"),
        ]

    def test_an_absent_file_is_an_empty_cache(self, tmp_path, assets):
        cache = PhotometryVerdictCache(path=tmp_path / "missing.json")
        assert len(cache) == 0
        assert cache.known(assets) == {}
        assert cache.unknown(assets) == assets

    def test_verdicts_survive_a_round_trip_and_are_keyed_by_asset_id(self, tmp_path, assets):
        cache = PhotometryVerdictCache(path=tmp_path / "verdicts.json")
        cache.record({"a": True})
        cache.save()

        reloaded = PhotometryVerdictCache(path=tmp_path / "verdicts.json")
        assert reloaded.known(assets) == {"one.nwb": True}
        assert [asset.asset_id for asset in reloaded.unknown(assets)] == ["b"]

    def test_an_unreadable_cache_is_ignored_rather_than_raised(self, tmp_path, assets, caplog):
        path = tmp_path / "corrupt.json"
        path.write_text("{not json")
        cache = PhotometryVerdictCache(path=path)
        assert len(cache) == 0
        assert "corrupt.json" in caplog.text


class TestOrderForVerification:
    @pytest.fixture
    def references(self):
        return [
            DandisetReference(identifier="000001", version="draft", asset_count=4000),
            DandisetReference(identifier="000002", version="draft", asset_count=10),
            DandisetReference(identifier="000003", version="draft", asset_count=200),
        ]

    def test_smallest_dandisets_are_read_first(self, references):
        ordered = order_for_verification(references)
        assert [reference.identifier for reference in ordered] == [
            "000002",
            "000003",
            "000001",
        ]

    def test_ordering_keeps_every_reference(self, references):
        assert sorted(order_for_verification(references), key=lambda r: r.identifier) == sorted(
            references, key=lambda r: r.identifier
        )

    def test_ordering_nothing_returns_nothing(self):
        assert order_for_verification([]) == []


class TestScanOrder:
    def _assets(self, sizes):
        return [
            AssetSummary(
                asset_id=str(size),
                path=f"{size}.nwb",
                size_in_bytes=size,
                content_url="u",
            )
            for size in sizes
        ]

    def test_the_ends_come_first_then_the_middle(self):
        ordered = scan_order(self._assets([10, 20, 30, 40, 50]))
        # Sorted largest-first that is 50, 40, 30, 20, 10; the walk takes both ends, then
        # bisects what is left.
        assert [asset.size_in_bytes for asset in ordered] == [50, 10, 30, 40, 20]

    def test_every_asset_is_visited_exactly_once(self):
        sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        ordered = scan_order(self._assets(sizes))
        assert sorted(asset.size_in_bytes for asset in ordered) == sizes

    def test_a_dandiset_of_one_asset_is_ordered(self):
        assert [asset.size_in_bytes for asset in scan_order(self._assets([7]))] == [7]

    def test_ordering_nothing_returns_nothing(self):
        assert scan_order([]) == []


class TestVerifyDandisets:
    """The scan itself is exercised above; these cover the per-dandiset decisions around it."""

    @pytest.fixture
    def archive_assets(self, byte_server, tmp_path):
        """Two dandisets: one whose photometry is not its largest asset, one with none."""
        photometry = tmp_path / "photometry.nwb"
        photometry.write_bytes(MOCK_NWB_FILES["mock_nwbfile_ndx_fiber_photometry_v0_2_core_events"].read_bytes())
        # Padded past the photometry file so it sorts first, which is what makes this dandiset
        # prove the scan carries on past the largest asset rather than stopping at it.
        big_behavior = tmp_path / "big_behavior.nwb"
        with h5py.File(big_behavior, "w") as file:
            file.create_group("general/devices")
            file.create_dataset("acquisition/filler", data=np.zeros(photometry.stat().st_size))
        small_behavior = tmp_path / "small_behavior.nwb"
        with h5py.File(small_behavior, "w") as file:
            file.create_group("general/devices")
        assert big_behavior.stat().st_size > photometry.stat().st_size

        def asset(asset_id, path):
            return AssetSummary(
                asset_id=asset_id,
                path=path.name,
                size_in_bytes=path.stat().st_size,
                content_url=served_url(byte_server, path.name),
            )

        return {
            "000001": [asset("big", big_behavior), asset("photometry", photometry)],
            "000002": [asset("small", small_behavior)],
        }

    @pytest.fixture
    def references(self):
        return [
            DandisetReference(identifier="000001", version="draft", asset_count=2),
            DandisetReference(identifier="000002", version="draft", asset_count=1),
        ]

    @pytest.fixture
    def list_assets(self, archive_assets):
        return lambda dandiset_id, version=None: list(archive_assets[dandiset_id])

    def test_a_dandiset_is_confirmed_by_any_asset_not_only_its_largest(self, references, list_assets):
        verdicts = verify_dandisets(references, list_assets_function=list_assets, process_count=2)
        assert verdicts == {"000001": True, "000002": False}

    def test_each_verdict_is_reported_as_it_settles(self, references, list_assets):
        settled = []
        verify_dandisets(
            references,
            list_assets_function=list_assets,
            process_count=2,
            on_verdict=lambda reference, holds: settled.append((reference.identifier, holds)),
        )
        assert settled == [("000001", True), ("000002", False)]

    def test_stopping_leaves_the_unreached_dandisets_out(self, references, list_assets):
        verdicts = verify_dandisets(
            references,
            list_assets_function=list_assets,
            process_count=2,
            should_stop=lambda: True,
        )
        assert verdicts == {}

    def test_a_dandiset_that_cannot_be_listed_is_unresolved(self, references, caplog):
        def refuse(dandiset_id, version=None):
            raise RuntimeError("archive said no")

        verdicts = verify_dandisets(references, list_assets_function=refuse, process_count=2)
        assert verdicts == {"000001": None, "000002": None}
        assert "archive said no" in caplog.text

    def test_a_dandiset_with_an_unreadable_asset_is_unresolved_not_empty(
        self, references, archive_assets, byte_server, tmp_path
    ):
        # 000002's only asset is served from a URL with nothing behind it, so no read of it
        # ever answers and the dandiset cannot honestly be called empty.
        archive_assets["000002"] = [
            AssetSummary(
                asset_id="gone",
                path="gone.nwb",
                size_in_bytes=1024,
                content_url=served_url(byte_server, "gone.nwb"),
            )
        ]
        verdicts = verify_dandisets(
            references,
            list_assets_function=lambda dandiset_id, version=None: list(archive_assets[dandiset_id]),
            process_count=2,
        )
        assert verdicts == {"000001": True, "000002": None}

    def test_an_unresolved_dandiset_is_not_remembered(self, references, archive_assets, byte_server, tmp_path):
        archive_assets["000002"] = [
            AssetSummary(
                asset_id="gone",
                path="gone.nwb",
                size_in_bytes=1024,
                content_url=served_url(byte_server, "gone.nwb"),
            )
        ]
        cache = PhotometryVerdictCache(path=tmp_path / "verdicts.json")
        listing = lambda dandiset_id, version=None: list(archive_assets[dandiset_id])  # noqa: E731
        verify_dandisets(references, list_assets_function=listing, cache=cache, process_count=2)

        reloaded = PhotometryVerdictCache(path=tmp_path / "verdicts.json")
        assert reloaded.dandiset_verdict(references[0]) is True
        # Nothing is remembered about the one that never answered, so it is read again.
        assert reloaded.dandiset_verdict(references[1]) is None

    def test_verifying_nothing_asks_the_archive_nothing(self):
        assert verify_dandisets([]) == {}

    def test_a_cached_positive_settles_a_dandiset_without_scanning(self, references, list_assets, tmp_path):
        cache = PhotometryVerdictCache(path=tmp_path / "verdicts.json")
        cache.record({"big": False, "photometry": True, "small": False})
        scanned = []

        def watched(dandiset_id, version=None):
            scanned.append(dandiset_id)
            return list_assets(dandiset_id, version)

        verdicts = verify_dandisets(references, list_assets_function=watched, cache=cache, process_count=2)
        assert verdicts == {"000001": True, "000002": False}
        # Listing still happens -- a dandiset can gain assets -- but nothing is re-read.
        assert scanned == ["000001", "000002"]

    def test_the_cache_keeps_what_a_run_computed(self, references, list_assets, tmp_path):
        cache = PhotometryVerdictCache(path=tmp_path / "verdicts.json")
        verify_dandisets(references, list_assets_function=list_assets, cache=cache, process_count=2)
        reloaded = PhotometryVerdictCache(path=tmp_path / "verdicts.json")
        assert reloaded.known(
            [
                AssetSummary(
                    asset_id="photometry",
                    path="photometry.nwb",
                    size_in_bytes=1,
                    content_url="u",
                )
            ]
        ) == {"photometry.nwb": True}


class TestFilterAssets:
    @pytest.fixture
    def assets(self):
        return [
            AssetSummary(asset_id="a", path="sub-01/ses-1_behavior.nwb", size_in_bytes=240_000),
            AssetSummary(
                asset_id="b",
                path="sub-01/ses-2_photometry.nwb",
                size_in_bytes=60_000_000,
            ),
            AssetSummary(asset_id="c", path="sub-02/ses-1_behavior.nwb", size_in_bytes=5_000_000),
        ]

    @pytest.fixture
    def verdicts(self):
        return {
            "sub-01/ses-1_behavior.nwb": False,
            "sub-01/ses-2_photometry.nwb": True,
            "sub-02/ses-1_behavior.nwb": False,
        }

    def test_the_filter_switched_off_keeps_everything(self, assets, verdicts):
        assert filter_assets(assets, photometry_by_path=verdicts, photometry_only=False) == assets

    def test_the_filter_keeps_only_the_scanned_photometry_assets(self, assets, verdicts):
        kept = filter_assets(assets, photometry_by_path=verdicts, photometry_only=True)
        assert [asset.asset_id for asset in kept] == ["b"]

    def test_nothing_scanned_yet_keeps_everything(self, assets):
        assert filter_assets(assets, photometry_by_path={}, photometry_only=True) == assets

    def test_an_asset_the_scan_never_reached_is_dropped(self, assets):
        verdicts = {"sub-01/ses-2_photometry.nwb": True}
        kept = filter_assets(assets, photometry_by_path=verdicts, photometry_only=True)
        assert [asset.asset_id for asset in kept] == ["b"]


class TestDandiFilterAgainstTheLocalByteServer(DandiFilterTestMixin):
    """The filter contract, bound to real NWB files served over local byte ranges."""

    expected_photometry_paths = ["sub-01/photometry.nwb"]
    process_count = 2

    @pytest.fixture
    def photometry_asset(self, byte_server, tmp_path):
        served = tmp_path / "photometry.nwb"
        served.write_bytes(MOCK_NWB_FILES["mock_nwbfile_ndx_fiber_photometry_v0_2_core_events"].read_bytes())
        return AssetSummary(
            asset_id="p",
            path="sub-01/photometry.nwb",
            size_in_bytes=served.stat().st_size,
            content_url=served_url(byte_server, "photometry.nwb"),
        )

    @pytest.fixture
    def behavior_only_asset(self, byte_server, tmp_path):
        served = tmp_path / "behavior.nwb"
        with h5py.File(served, "w") as file:
            file.create_group("general/devices")
        return AssetSummary(
            asset_id="b",
            path="sub-01/behavior.nwb",
            size_in_bytes=served.stat().st_size,
            content_url=served_url(byte_server, "behavior.nwb"),
        )

    @pytest.fixture
    def asset_listing(self, photometry_asset, behavior_only_asset):
        return [photometry_asset, behavior_only_asset]

    @pytest.fixture
    def photometry_dandiset(self):
        return DandisetReference(identifier="000001", version="draft", asset_count=2)

    @pytest.fixture
    def list_assets_function(self, asset_listing):
        return lambda dandiset_id, version=None: list(asset_listing)

"""Decide whether a DANDI asset, or a whole dandiset, holds photometry GuPPy can read.

The question is answered off the remote bytes rather than off the archive's metadata, because the
metadata does not carry it (see :mod:`guppy.utils.dandi_search`). Reading is arranged so that the
answer costs a few kilobytes per file: :class:`PrefetchedRemoteFile` fetches a head and a tail
window in parallel and serves h5py's pointer-chasing out of them, and
:func:`asset_holds_photometry` answers in two stages off that one open file.

The same reading answers the question one level up. :func:`verify_dandisets` stops at the first
asset that holds photometry, so confirming a dandiset is usually a handful of files while ruling
one out means reading every asset it has -- an asymmetry the ordering helpers and
:class:`PhotometryVerdictCache` exist to absorb.

The NWB finder helpers here are the cheap half of what :mod:`guppy.utils.dandi_preview` does: they
locate the fiber photometry container and response series, which is all a yes-or-no needs, and the
preview module builds the full picture on top of them.
"""

import io
import json
import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import h5py
import requests
from platformdirs import user_cache_dir
from requests.adapters import HTTPAdapter

from .dandi_search import AssetSummary, DandisetSummary, list_nwb_assets

logger = logging.getLogger(__name__)

# The container ndx-fiber-photometry writes its FiberPhotometryTable into, as a neurodata
# type rather than a name: the name is whichever one the file's author passed, and differs
# between writers. Every FiberPhotometryResponseSeries references a region of that table, so a
# file without the container holds no photometry, which is what makes it the scan's cheap
# first stage: an attribute read per child of /general, all served out of the prefetched tail
# window. The converse does not hold. The container travels with the metadata table, and a
# file can write that table while storing its traces as some other series type, so a file that
# has it is walked for the series itself before it counts as photometry -- reaching the series'
# own object headers, which sit beside their data in the middle of the file.
#
# The cached extension namespaces under /specifications are NOT a usable signal: they record
# which extensions the conversion session had loaded, not which types it wrote, so a
# behavior-only file written by a photometry pipeline still declares ndx-fiber-photometry.
FIBER_PHOTOMETRY_LAB_META_DATA = "FiberPhotometry"

# Bytes prefetched from each end of a remote file before handing it to h5py. Answering the
# question above reads only ~5 KB, but h5py discovers those bytes by pointer-chasing through
# the superblock and object headers -- sixteen round trips, each waiting on the last. HDF5
# places those headers at whatever was the end of the file when they were last written, so a
# head and a tail window fetched in parallel cover them for any file written in a single
# session, which is what a one-shot conversion produces. Widening the windows is not free:
# at these sizes the scan already saturates a typical connection.
PREFETCH_HEAD_IN_BYTES = 64 * 1024
PREFETCH_TAIL_IN_BYTES = 256 * 1024

# Assets scanned concurrently. Processes rather than threads because h5py serializes on a
# global lock, which would otherwise collapse the concurrency to roughly one file at a time.
SCAN_PROCESS_COUNT = 32

# How many dandiset asset listings to fetch ahead of the scan that consumes them.
LISTING_PREFETCH_COUNT = 12

# How many further passes a dandiset's unreadable assets get before it is left unresolved, and
# the delay before the first, doubling after that. The archive resets connections under load, and
# a reset recorded as "no photometry" would be indistinguishable from a real answer. Retrying is
# only ever needed on the way to a negative: one asset holding photometry settles the dandiset
# whatever the others did.
SCAN_RETRY_PASSES = 2
SCAN_RETRY_DELAY_IN_SECONDS = 1.0

# The series type GuPPy's reader picks up traces from. A file can write the metadata table above
# and still store its traces as some other series type, so having the container is not enough.
FIBER_PHOTOMETRY_RESPONSE_SERIES = "FiberPhotometryResponseSeries"

_scan_session: requests.Session | None = None


def _session() -> requests.Session:
    """Return this process's HTTP session, whose connection pool outlives one asset."""
    global _scan_session
    if _scan_session is None:
        _scan_session = requests.Session()
        _scan_session.mount("https://", HTTPAdapter(pool_connections=4, pool_maxsize=4))
    return _scan_session


class PrefetchedRemoteFile(io.RawIOBase):
    """A seekable view of a remote file served from a prefetched head and tail window.

    h5py reads through this object instead of issuing its own range requests, so the reads it
    makes to resolve a path cost no network round trips. A read that falls between the two
    windows -- which happens when bulk data was appended after the metadata was last written --
    is served by a range request of its own, which is correct but back to one trip per read.

    Parameters
    ----------
    content_url : str
        Directly readable URL for the file's bytes.
    size_in_bytes : int
        The file's total size, as the archive's listing reports it.
    """

    def __init__(self, *, content_url: str, size_in_bytes: int) -> None:
        self._content_url = content_url
        self._size = size_in_bytes
        self._position = 0
        self.gap_read_count = 0
        if size_in_bytes <= PREFETCH_HEAD_IN_BYTES + PREFETCH_TAIL_IN_BYTES:
            self._head = self._fetch(0, size_in_bytes)
            self._tail = b""
            self._tail_start = size_in_bytes
        else:
            self._tail_start = size_in_bytes - PREFETCH_TAIL_IN_BYTES
            with ThreadPoolExecutor(max_workers=2) as pool:
                head = pool.submit(self._fetch, 0, PREFETCH_HEAD_IN_BYTES)
                tail = pool.submit(self._fetch, self._tail_start, size_in_bytes)
                self._head, self._tail = head.result(), tail.result()

    def _fetch(self, start: int, end: int) -> bytes:
        """Fetch the half-open byte range ``[start, end)``."""
        response = _session().get(self._content_url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=60)
        response.raise_for_status()
        return response.content

    def readable(self) -> bool:
        """Report that the file can be read."""
        return True

    def seekable(self) -> bool:
        """Report that the file can be seeked."""
        return True

    def tell(self) -> int:
        """Return the current read position."""
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Move the read position and return where it landed."""
        if whence == io.SEEK_SET:
            self._position = offset
        elif whence == io.SEEK_CUR:
            self._position += offset
        else:
            self._position = self._size + offset
        return self._position

    def read(self, size: int = -1) -> bytes:
        """Return ``size`` bytes from the current position, fetching any the windows miss."""
        if size is None or size < 0:
            size = self._size - self._position
        start = self._position
        end = min(start + size, self._size)
        if end <= len(self._head):
            data = self._head[start:end]
        elif start >= self._tail_start:
            data = self._tail[start - self._tail_start : end - self._tail_start]
        else:
            self.gap_read_count += 1
            data = self._fetch(start, end)
        self._position = end
        return data

    def readinto(self, buffer: bytearray) -> int:
        """Fill ``buffer`` from the current position and return how many bytes landed in it."""
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def find_typed_child(group: h5py.Group | None, neurodata_type: str) -> h5py.Group | None:
    """Return the first child of ``group`` carrying ``neurodata_type``, or None."""
    if group is None:
        return None
    for member in group.values():
        if member.attrs.get("neurodata_type") == neurodata_type:
            return member
    return None


def find_fiber_photometry_container(file: h5py.File) -> h5py.Group | None:
    """Return the file's ``FiberPhotometry`` container, whatever its author named it."""
    return find_typed_child(file.get("general"), FIBER_PHOTOMETRY_LAB_META_DATA)


def find_photometry_series(file: h5py.File) -> dict[str, h5py.Group]:
    """Return every ``FiberPhotometryResponseSeries`` group, keyed by object name.

    Searches ``/acquisition`` and the modules under ``/processing`` -- the two places NWB puts
    time series -- rather than walking the whole file, so the scan costs a handful of metadata
    reads even when the file is streamed over the network.
    """
    series: dict[str, h5py.Group] = {}
    for group in series_containers(file):
        for name, member in group.items():
            if member.attrs.get("neurodata_type") == FIBER_PHOTOMETRY_RESPONSE_SERIES:
                series[name] = member
    return series


def series_containers(file: h5py.File) -> list[h5py.Group]:
    """Return the groups that directly contain time series: acquisition and each module."""
    containers = []
    acquisition = file.get("acquisition")
    if acquisition is not None:
        containers.append(acquisition)
    processing = file.get("processing")
    if processing is not None:
        containers.extend(processing.values())
    return containers


def asset_holds_photometry(asset: AssetSummary) -> bool | None:
    """Report whether one remote asset holds a trace GuPPy can read, without downloading it.

    Answers in two stages off a single open file. A file with no ``FiberPhotometry`` container
    under ``/general`` holds no photometry and is answered from the prefetched window alone.
    One that has the container is then walked for the ``FiberPhotometryResponseSeries`` GuPPy
    reads its traces from, at the cost of a few range requests of its own.

    Runs in a worker process, so it takes and returns only picklable values, and it answers
    once rather than retrying: whether a failed read is worth repeating depends on what the
    dandiset's other assets said, which only the caller knows.

    Parameters
    ----------
    asset : AssetSummary
        The asset to inspect, carrying the URL its bytes are read from.

    Returns
    -------
    bool or None
        Whether the file holds a fiber photometry response series, or None when it could not
        be read. A read that fails is not an answer: reporting it as False would let a dropped
        connection quietly turn a photometry dandiset into a behavior-only one.
    """
    try:
        reader = PrefetchedRemoteFile(content_url=asset.content_url, size_in_bytes=asset.size_in_bytes)
        with h5py.File(reader, mode="r") as file:
            if find_fiber_photometry_container(file) is None:
                return False
            return bool(find_photometry_series(file))
    except Exception as error:
        logger.debug("Could not read %s: %s", asset.path, error)
        return None


class PhotometryVerdictCache:
    """Verdicts already known, kept between sessions.

    Two kinds. Asset verdicts are keyed by DANDI asset ID, which addresses an immutable blob,
    so an answer never needs recomputing. Dandiset verdicts are keyed by identifier and stored
    alongside the asset count they were reached at: a dandiset that holds photometry always
    will, while one that does not can acquire it, so a negative is trusted only while the
    dandiset is the size it was when it was read. Either way a repeat costs no requests at
    all, where asset verdicts alone still leave the listing to be fetched.

    Only answers are stored. An asset that could not be read has no verdict to remember, and a
    dandiset holding one is not settled, so neither is written and both are retried next time.

    A cache that cannot be read is an empty one. Nothing here is authoritative -- every entry
    can be recomputed from the archive -- so a corrupt or unwritable file costs time, not
    correctness.

    Parameters
    ----------
    path : Path or None, optional
        File the verdicts are stored in. Defaults to GuPPy's user cache directory.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_verdict_cache_path()
        self._assets: dict[str, bool] = {}
        self._dandisets: dict[str, list] = {}
        if self.path.is_file():
            try:
                stored = json.loads(self.path.read_text())
                self._assets = {key: bool(value) for key, value in (stored.get("assets") or {}).items()}
                self._dandisets = {
                    key: [bool(value[0]), int(value[1])] for key, value in (stored.get("dandisets") or {}).items()
                }
            except (OSError, ValueError, IndexError, TypeError) as error:
                logger.warning(
                    "Ignoring unreadable photometry verdict cache %s: %s",
                    self.path,
                    error,
                )

    def __len__(self) -> int:
        return len(self._assets)

    def known(self, assets: Sequence[AssetSummary]) -> dict[str, bool]:
        """Return the verdicts already held for ``assets``, keyed by asset path."""
        return {asset.path: self._assets[asset.asset_id] for asset in assets if asset.asset_id in self._assets}

    def unknown(self, assets: Sequence[AssetSummary]) -> list[AssetSummary]:
        """Return the assets whose verdict is not held yet."""
        return [asset for asset in assets if asset.asset_id not in self._assets]

    def record(self, verdicts_by_asset_id: dict[str, bool | None]) -> None:
        """Take note of newly computed asset verdicts, ignoring the reads that failed."""
        self._assets.update({key: value for key, value in verdicts_by_asset_id.items() if value is not None})

    def dandiset_verdict(self, reference: "DandisetReference") -> bool | None:
        """Return what is known about a whole dandiset, or None when it must be read.

        A remembered positive always stands. A remembered negative stands only while the
        dandiset still holds the number of assets it held when it was read.
        """
        remembered = self._dandisets.get(reference.identifier)
        if remembered is None:
            return None
        holds, asset_count = remembered
        if holds:
            return True
        return False if asset_count == reference.asset_count else None

    def record_dandiset(self, reference: "DandisetReference", holds: bool) -> None:
        """Take note of a whole dandiset's settled verdict."""
        self._dandisets[reference.identifier] = [holds, reference.asset_count]

    def save(self) -> None:
        """Write the verdicts out, replacing whatever was there."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"assets": self._assets, "dandisets": self._dandisets}))
        except OSError as error:
            logger.warning("Could not write the photometry verdict cache %s: %s", self.path, error)


def default_verdict_cache_path() -> Path:
    """Return the file GuPPy caches DANDI photometry verdicts in."""
    return Path(user_cache_dir("guppy", "LernerLab")) / "dandi_photometry_verdicts.json"


def scan_assets_for_photometry(
    assets: Sequence[AssetSummary],
    *,
    process_count: int = SCAN_PROCESS_COUNT,
    progress_callback: Callable[[int], None] | None = None,
    cache: PhotometryVerdictCache | None = None,
) -> dict[str, bool]:
    """Report which of ``assets`` hold fiber photometry.

    Parameters
    ----------
    assets : sequence of AssetSummary
        The assets to scan.
    process_count : int, optional
        How many assets to scan at once.
    progress_callback : callable or None, optional
        Called with the number of assets scanned so far, each time one finishes.
    cache : PhotometryVerdictCache or None, optional
        Verdicts already known, which are returned instead of being rescanned and extended
        with whatever this run computes. No caching happens when omitted.

    Returns
    -------
    dict of {str: bool}
        Asset path mapped to whether that asset holds fiber photometry.
    """
    if not assets:
        return {}
    verdicts = dict(cache.known(assets)) if cache is not None else {}
    outstanding = cache.unknown(assets) if cache is not None else list(assets)
    if progress_callback is not None and verdicts:
        progress_callback(len(verdicts))
    if outstanding:
        with ProcessPoolExecutor(max_workers=min(process_count, len(outstanding))) as pool:
            asset_by_future = {pool.submit(asset_holds_photometry, asset): asset for asset in outstanding}
            for future in as_completed(asset_by_future):
                asset = asset_by_future[future]
                holds = future.result()
                verdicts[asset.path] = holds
                if cache is not None:
                    cache.record({asset.asset_id: holds})
                if progress_callback is not None:
                    progress_callback(len(verdicts))
    if cache is not None:
        cache.save()
    return verdicts


@dataclass(frozen=True)
class DandisetReference:
    """The little about a dandiset that deciding whether to scan it needs.

    The crawl visits every dandiset on the archive, and fetching each one's metadata to do
    that would cost a request per dandiset for information only the confirmed ones ever
    display. The archive's own listing carries these three fields already.

    Attributes
    ----------
    identifier : str
        Six-digit dandiset ID.
    version : str
        The version to read: its newest published one, else its draft.
    asset_count : int
        How many assets that version holds.
    """

    identifier: str
    version: str
    asset_count: int

    @classmethod
    def from_summary(cls, summary: DandisetSummary) -> "DandisetReference":
        """Return the reference describing an already-summarized dandiset."""
        return cls(
            identifier=summary.identifier,
            version=summary.version,
            asset_count=summary.file_count,
        )


def order_for_verification(
    references: Sequence[DandisetReference],
) -> list[DandisetReference]:
    """Order dandisets so that the ones likeliest to settle quickly are read first.

    Smallest first: confirming a dandiset takes one file, but ruling one out means reading
    every asset it has, so the largest dandisets are the slowest to settle either way. Reading
    them last means the answer fills in steadily from the start rather than stalling on one
    dataset of thousands of files.

    Parameters
    ----------
    references : sequence of DandisetReference
        The dandisets to order.

    Returns
    -------
    list of DandisetReference
        The same references, smallest first.
    """
    return sorted(references, key=lambda reference: reference.asset_count)


def verify_dandisets(
    references: Sequence[DandisetReference],
    *,
    list_assets_function: object = list_nwb_assets,
    cache: PhotometryVerdictCache | None = None,
    process_count: int = SCAN_PROCESS_COUNT,
    on_verdict: Callable[[DandisetReference, bool | None], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, bool | None]:
    """Report which of ``references`` hold fiber photometry GuPPy can read.

    Reading stops at the first asset that holds photometry, so confirming a dandiset is
    usually a handful of files, while ruling one out means reading every asset it has. A
    dandiset whose assets could not all be read is reported as None rather than as empty.

    Parameters
    ----------
    references : sequence of DandisetReference
        The dandisets to verify, in the order they should be visited.
    list_assets_function : callable, optional
        Injection point for the asset listing.
    cache : PhotometryVerdictCache or None, optional
        Verdicts already known, which are consulted instead of rereading and extended with
        whatever this run computes. No caching happens when omitted.
    process_count : int, optional
        How many assets to scan at once.
    on_verdict : callable or None, optional
        Called with each reference and its verdict as soon as that dandiset settles.
    should_stop : callable or None, optional
        Consulted before each dandiset; a true answer ends the run early, returning the
        verdicts reached so far.

    Returns
    -------
    dict of {str: bool or None}
        Dandiset identifier mapped to whether it holds fiber photometry, or to None when its
        assets could not all be read. A dandiset the run stopped before reaching is absent.
    """
    verdicts: dict[str, bool | None] = {}
    if not references:
        return verdicts
    # Listings are network-bound and independent, and one of them can take seconds on a
    # dandiset with thousands of assets. Fetching them ahead of the scan that needs them keeps
    # the process pool fed instead of idling between dandisets.
    with (
        ProcessPoolExecutor(max_workers=process_count) as pool,
        ThreadPoolExecutor(max_workers=LISTING_PREFETCH_COUNT) as listers,
    ):
        listings = {
            reference.identifier: listers.submit(
                _list_for_verification,
                reference=reference,
                list_assets_function=list_assets_function,
                cache=cache,
            )
            for reference in references
        }
        for reference in references:
            if should_stop is not None and should_stop():
                for pending in listings.values():
                    pending.cancel()
                break
            holds = _dandiset_holds_photometry(
                reference=reference,
                pool=pool,
                assets=listings[reference.identifier].result(),
                cache=cache,
                chunk_size=process_count,
            )
            verdicts[reference.identifier] = holds
            if on_verdict is not None:
                on_verdict(reference, holds)
    if cache is not None:
        cache.save()
    return verdicts


def _list_for_verification(
    *,
    reference: DandisetReference,
    list_assets_function: object,
    cache: PhotometryVerdictCache | None,
) -> list[AssetSummary] | None:
    """Fetch the assets a dandiset must be read from.

    None means no listing was taken: either the cache already answers for this dandiset, or
    the archive would not say. The caller checks the cache first, so a None that reaches it
    is the second case, which leaves the dandiset unresolved rather than empty.
    """
    if cache is not None and cache.dandiset_verdict(reference) is not None:
        return None
    try:
        return list_assets_function(dandiset_id=reference.identifier, version=reference.version)
    except Exception as error:
        logger.warning(
            "Could not list dandiset %s while verifying: %s",
            reference.identifier,
            error,
        )
        return None


def scan_order(assets: Sequence[AssetSummary]) -> list[AssetSummary]:
    """Order a dandiset's assets so that a few reads span every size it holds.

    Which asset carries the photometry depends on what else the dandiset carries. Where the
    recordings are the bulk of it, they are the largest files and the behavior-only sidecars
    the smallest. Where photometry accompanies electrophysiology, it is the other way around:
    in dandiset 000689 the photometry files are 5 MB against 19 GB of ephys, and rank 33rd of
    53 by size. Reading from either end alone therefore misses one of those layouts entirely.

    Sorting by size and then walking the ends and repeatedly bisecting what is left visits a
    spread of sizes immediately, so whichever band the photometry occupies is reached within a
    handful of reads rather than after a scan of everything above it.

    Parameters
    ----------
    assets : sequence of AssetSummary
        The dandiset's assets.

    Returns
    -------
    list of AssetSummary
        Every asset, reordered.
    """
    by_size = sorted(assets, key=lambda asset: -asset.size_in_bytes)
    return [by_size[index] for index in _spread_indices(len(by_size))]


def _spread_indices(count: int) -> list[int]:
    """Return ``0..count-1`` ordered so that any prefix spans the whole range."""
    if count <= 2:
        return list(range(count))
    order = [0, count - 1]
    taken = set(order)
    segments = [(0, count - 1)]
    while segments:
        low, high = segments.pop(0)
        middle = (low + high) // 2
        if middle not in taken:
            taken.add(middle)
            order.append(middle)
        if middle - low > 1:
            segments.append((low, middle))
        if high - middle > 1:
            segments.append((middle, high))
    return order


def _dandiset_holds_photometry(
    *,
    reference: DandisetReference,
    pool: ProcessPoolExecutor,
    assets: list[AssetSummary] | None,
    cache: PhotometryVerdictCache | None,
    chunk_size: int,
) -> bool | None:
    """Whether a dandiset holds photometry: True, False, or None when it could not be settled.

    Reading stops at the first asset that holds photometry, so confirming a dandiset is
    usually a handful of files. Ruling one out has no such shortcut and admits no gaps: every
    asset has to have actually answered, so any that could not be read are read again before
    the dandiset is called empty, and it is left unresolved rather than negative if they still
    will not answer.
    """
    if cache is not None:
        remembered = cache.dandiset_verdict(reference)
        if remembered is not None:
            return remembered
    if assets is None:
        return None
    if not assets:
        return False

    outstanding = scan_order(assets)
    if cache is not None:
        known = cache.known(outstanding)
        if any(known.values()):
            cache.record_dandiset(reference, True)
            return True
        outstanding = cache.unknown(outstanding)

    for retry_pass in range(SCAN_RETRY_PASSES + 1):
        if retry_pass:
            # Only reached when nothing has been found and some reads failed, which is when a
            # blip would otherwise masquerade as a definite answer.
            time.sleep(SCAN_RETRY_DELAY_IN_SECONDS * 2 ** (retry_pass - 1))
        unresolved = []
        for start in range(0, len(outstanding), chunk_size):
            chunk = outstanding[start : start + chunk_size]
            results = list(pool.map(asset_holds_photometry, chunk))
            if cache is not None:
                cache.record({asset.asset_id: holds for asset, holds in zip(chunk, results, strict=True)})
            if any(holds for holds in results if holds is not None):
                if cache is not None:
                    cache.record_dandiset(reference, True)
                return True
            unresolved += [asset for asset, holds in zip(chunk, results, strict=True) if holds is None]
        if not unresolved:
            if cache is not None:
                cache.record_dandiset(reference, False)
            return False
        outstanding = unresolved

    logger.warning(
        "Left dandiset %s unresolved: %d asset(s) could not be read.",
        reference.identifier,
        len(outstanding),
    )
    return None


def filter_assets(
    assets: Sequence[AssetSummary],
    *,
    photometry_by_path: dict[str, bool],
    photometry_only: bool,
) -> list[AssetSummary]:
    """Narrow an asset listing to the scanned assets that hold fiber photometry.

    Parameters
    ----------
    assets : sequence of AssetSummary
        The listing to narrow.
    photometry_by_path : dict of {str: bool}
        Scan verdicts, as :func:`scan_assets_for_photometry` returns them.
    photometry_only : bool
        When False, or when nothing has been scanned, every asset is kept.

    Returns
    -------
    list of AssetSummary
        The matching assets, in their original order.
    """
    if not photometry_only or not photometry_by_path:
        return list(assets)
    return [asset for asset in assets if photometry_by_path.get(asset.path)]

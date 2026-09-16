"""Search the DANDI Archive for fiber photometry datasets and inspect their NWB assets.

Four layers live here. The catalog layer talks to the DANDI REST API: it runs the archive's
full-text search, pulls each hit's dandiset metadata, and reduces it to a
:class:`DandisetSummary` that the browser can tabulate and filter, and it lists a dandiset's
NWB assets with the URLs their bytes are readable from. The scan layer answers, for every
asset of a dandiset at once, which of them hold fiber photometry at all. The verification
layer asks the same question of whole dandisets, stopping at the first asset that answers yes
and caching what it learns. The probe layer opens one NWB asset's HDF5 header and reports what
that file holds in detail -- the channels, their brain regions and indicators, and a short
slice of each trace.

The split exists because DANDI's structured metadata does not describe fiber photometry.
``assetsSummary.variableMeasured`` is built by dandi-cli from the core NWB types it knows, and
``FiberPhotometryResponseSeries`` is not among them: no dandiset in the archive lists it, so a
search for photometry datasets can only go through the free text a submitter wrote. Anything
authoritative -- which files carry photometry data, from which sites, with which indicator --
has to come from the files.
"""

import io
import json
import logging
import re
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path

import h5py
import numpy as np
import requests
from dandi.dandiapi import DandiAPIClient
from platformdirs import user_cache_dir
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

# Terms handed to the archive's full-text search to assemble the candidate catalog. A fiber
# photometry dandiset names the method somewhere in its title, description or keywords, and
# nothing else measurably adds to it: sensor families return only dandisets the one term already
# returns, while broader words like "gcamp" and "fluorescence" pull in two-photon and widefield
# imaging datasets that GuPPy cannot read. What the search misses is recovered by the crawl
# rather than by more terms.
PHOTOMETRY_SEARCH_TERMS = ("photometry",)

# How many samples a preview trace is decimated to before it is plotted.
DEFAULT_TRACE_POINTS = 2000
# Seconds of each channel a preview reads off the front of the recording.
DEFAULT_TRACE_DURATION_IN_SECONDS = 60.0

# Excitation wavelengths at or below this are isosbestic for the green calcium and dopamine
# sensors, whose isosbestic point sits near 405-415 nm. Used only to suggest a GuPPy store label.
ISOSBESTIC_WAVELENGTH_CEILING_IN_NM = 420.0

# Canonical brain region -> the spellings and abbreviations searched for in a dandiset's free
# text. Every pattern is matched on word boundaries, so "LH" does not fire inside "LHb".
BRAIN_REGION_VOCABULARY: dict[str, tuple[str, ...]] = {
    "Dorsal striatum": (
        "dorsal striatum",
        "dorsomedial striatum",
        "dorsolateral striatum",
        "dms",
        "dls",
    ),
    "Ventral striatum": (
        "ventral striatum",
        "nucleus accumbens",
        "nac",
        "nacc",
        "accumbens",
    ),
    "Striatum": ("striatum", "striatal"),
    "Substantia nigra": ("substantia nigra", "snc", "snr"),
    "Ventral tegmental area": ("ventral tegmental area", "vta"),
    "Prefrontal cortex": (
        "prefrontal cortex",
        "mpfc",
        "pfc",
        "prelimbic",
        "infralimbic",
    ),
    "Orbitofrontal cortex": ("orbitofrontal cortex", "ofc"),
    "Anterior cingulate cortex": ("anterior cingulate", "acc"),
    "Motor cortex": ("motor cortex", "m1", "m2"),
    "Somatosensory cortex": ("somatosensory cortex", "barrel cortex", "s1"),
    "Visual cortex": ("visual cortex", "v1"),
    "Auditory cortex": ("auditory cortex", "a1"),
    "Insular cortex": ("insular cortex", "insula"),
    "Hippocampus": ("hippocampus", "hippocampal", "ca1", "ca3", "dentate gyrus"),
    "Entorhinal cortex": ("entorhinal",),
    "Amygdala": ("amygdala", "bla", "cea", "basolateral amygdala", "central amygdala"),
    "Bed nucleus of the stria terminalis": (
        "bed nucleus of the stria terminalis",
        "bnst",
    ),
    "Septum": ("septum", "septal"),
    "Basal forebrain": ("basal forebrain",),
    "Globus pallidus": ("globus pallidus", "gpe", "gpi"),
    "Subthalamic nucleus": ("subthalamic nucleus", "stn"),
    "Pedunculopontine nucleus": ("pedunculopontine", "ppn", "ppt"),
    "Thalamus": ("thalamus", "thalamic", "mediodorsal thalamus"),
    "Habenula": ("habenula", "lhb", "mhb"),
    "Hypothalamus": (
        "hypothalamus",
        "hypothalamic",
        "lateral hypothalamus",
        "pvn",
        "arcuate",
        "vmh",
    ),
    "Locus coeruleus": ("locus coeruleus", "lc"),
    "Dorsal raphe": ("dorsal raphe", "raphe", "drn"),
    "Periaqueductal gray": ("periaqueductal", "pag"),
    "Superior colliculus": ("superior colliculus",),
    "Parabrachial nucleus": ("parabrachial",),
    "Cerebellum": ("cerebellum", "cerebellar", "purkinje"),
    "Olfactory bulb": ("olfactory bulb",),
    "Zona incerta": ("zona incerta",),
    "Spinal cord": ("spinal cord",),
}

# Canonical indicator -> the spellings searched for in a dandiset's free text. GuPPy's users
# choose datasets by sensor as much as by region, and the sensor is almost never in DANDI's
# structured metadata.
INDICATOR_VOCABULARY: dict[str, tuple[str, ...]] = {
    "GCaMP": ("gcamp", "jgcamp"),
    "dLight": ("dlight", "rdlight"),
    "GRAB-DA": ("grab-da", "grabda", "grab_da", "grab da", "grabda2m", "grabda3h"),
    "GRAB-ACh": ("grab-ach", "grabach", "grab ach", "ach3.0", "ach 3.0", "gach"),
    "GRAB-NE": ("grab-ne", "grabne", "grab ne", "nlight", "grabnE2h"),
    "GRAB-5HT": ("grab-5ht", "grab5ht", "grab 5ht"),
    "GRAB-eCB": ("grab-ecb", "grabecb", "ecb2.0"),
    "iGluSnFR": ("iglusnfr",),
    "iAChSnFR": ("iachsnfr",),
    "iSeroSnFR": ("iserosnfr",),
    "jRGECO": ("jrgeco",),
    "RCaMP": ("rcamp",),
    "GFP": ("gfp", "eyfp"),
    "tdTomato": ("tdtomato",),
}


def _compile_vocabulary(
    vocabulary: dict[str, tuple[str, ...]], *, allow_suffix: bool = False
) -> dict[str, re.Pattern[str]]:
    """Compile one alternation per canonical term in ``vocabulary``.

    Every pattern is anchored on its left at a word boundary. ``allow_suffix`` leaves the
    right end open, which is what indicator names need: a sensor is named by its family plus
    a variant suffix ("GCaMP6f", "dLight1.3b", "GRAB-DA2m") that the family term must still
    match. Region abbreviations need the closing boundary, so that "LH" does not fire on the
    habenula's "LHb".
    """
    compiled = {}
    trailing = "" if allow_suffix else r"(?![\w-])"
    for canonical, patterns in vocabulary.items():
        alternation = "|".join(re.escape(pattern) for pattern in patterns)
        compiled[canonical] = re.compile(rf"(?<![\w-])({alternation}){trailing}", re.IGNORECASE)
    return compiled


_BRAIN_REGION_PATTERNS = _compile_vocabulary(BRAIN_REGION_VOCABULARY)
_INDICATOR_PATTERNS = _compile_vocabulary(INDICATOR_VOCABULARY, allow_suffix=True)


def find_vocabulary_terms(*, text: str, vocabulary: dict[str, re.Pattern[str]]) -> tuple[str, ...]:
    """Return every canonical term in ``vocabulary`` whose pattern appears in ``text``.

    Parameters
    ----------
    text : str
        Free text to scan.
    vocabulary : dict of {str: re.Pattern}
        Canonical term to the compiled pattern that recognizes it.

    Returns
    -------
    tuple of str
        The matching canonical terms, in the vocabulary's own order.
    """
    return tuple(canonical for canonical, pattern in vocabulary.items() if pattern.search(text))


# ----------------------------------------------------------------------------------------------
# Catalog layer: the DANDI REST API
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DandisetSummary:
    """One row of the photometry catalog, reduced from a dandiset's DANDI metadata.

    Attributes
    ----------
    identifier : str
        Six-digit dandiset ID.
    version : str
        Version the metadata was read from -- a published version string, or ``"draft"``.
    name : str
        Dandiset title.
    description : str
        Dandiset abstract.
    species : tuple of str
        Species names from ``assetsSummary``, shortened to the common binomial.
    approaches : tuple of str
        Experimental approaches and measurement techniques from ``assetsSummary``.
    keywords : tuple of str
        Submitter-supplied keywords.
    brain_regions : tuple of str
        Canonical regions recognized in the dandiset's free text.
    indicators : tuple of str
        Canonical indicators recognized in the dandiset's free text.
    subject_count, file_count, size_in_bytes : int
        Totals from ``assetsSummary``; ``0`` for a dandiset whose summary is still empty.
    contributors : tuple of str
        Names credited in the citation.
    license_terms : tuple of str
        SPDX license identifiers.
    url : str
        Landing page on dandiarchive.org.
    is_published : bool
        Whether a published version exists, as opposed to a draft-only dandiset.
    searchable_text : str
        Lowercased concatenation of every text field, for local free-text filtering.
    """

    identifier: str
    version: str
    name: str
    description: str
    species: tuple[str, ...]
    approaches: tuple[str, ...]
    keywords: tuple[str, ...]
    brain_regions: tuple[str, ...]
    indicators: tuple[str, ...]
    subject_count: int
    file_count: int
    size_in_bytes: int
    contributors: tuple[str, ...]
    license_terms: tuple[str, ...]
    url: str
    is_published: bool
    searchable_text: str


def _names(entries: Iterable[object]) -> tuple[str, ...]:
    """Pull the ``name`` off each schema entry, dropping duplicates and keeping order."""
    names = []
    for entry in entries:
        name = entry.get("name") if isinstance(entry, dict) else entry
        if name and name not in names:
            names.append(name)
    return tuple(names)


def _summarize_dandiset(
    *,
    identifier: str,
    version: str,
    metadata: dict[str, object],
    file_count: int | None = None,
    size_in_bytes: int | None = None,
) -> DandisetSummary:
    """Reduce one dandiset's raw DANDI metadata to a :class:`DandisetSummary`.

    ``file_count`` and ``size_in_bytes`` override the totals in ``assetsSummary``, which a
    dandiset that has never been published can leave at zero while holding assets. The
    version record the archive's dandiset listing returns always carries both.
    """
    assets_summary = metadata.get("assetsSummary") or {}
    keywords = tuple(metadata.get("keywords") or ())
    study_targets = tuple(metadata.get("studyTarget") or ())
    about = _names(metadata.get("about") or ())
    name = metadata.get("name") or ""
    description = metadata.get("description") or ""
    # Everything a submitter wrote about the dataset, in one blob: what the vocabularies are
    # scanned against and what the browser's free-text filter searches.
    searchable_text = " ".join(
        (
            name,
            description,
            " ".join(keywords),
            " ".join(study_targets),
            " ".join(about),
            identifier,
        )
    ).lower()

    return DandisetSummary(
        identifier=identifier,
        version=version,
        name=name,
        description=description,
        species=tuple(species.split(" - ")[0] for species in _names(assets_summary.get("species") or ())),
        approaches=_names(assets_summary.get("approach") or ())
        + _names(assets_summary.get("measurementTechnique") or ()),
        keywords=keywords,
        brain_regions=find_vocabulary_terms(text=searchable_text, vocabulary=_BRAIN_REGION_PATTERNS),
        indicators=find_vocabulary_terms(text=searchable_text, vocabulary=_INDICATOR_PATTERNS),
        subject_count=int(assets_summary.get("numberOfSubjects") or 0),
        file_count=int(file_count if file_count is not None else (assets_summary.get("numberOfFiles") or 0)),
        size_in_bytes=int(size_in_bytes if size_in_bytes is not None else (assets_summary.get("numberOfBytes") or 0)),
        contributors=_names(metadata.get("contributor") or ()),
        license_terms=tuple(metadata.get("license") or ()),
        url=metadata.get("url") or f"https://dandiarchive.org/dandiset/{identifier}",
        is_published=version != "draft",
        searchable_text=searchable_text,
    )


def _search_rows(*, client: DandiAPIClient, terms: Sequence[str], page_size: int = 100) -> list[dict[str, object]]:
    """Return the dandiset listing rows matching any of ``terms``, in first-match order."""
    rows_by_identifier: dict[str, dict[str, object]] = {}
    for term in terms:
        for row in client.paginate("/dandisets/", params={"search": term, "page_size": page_size}):
            rows_by_identifier.setdefault(row["identifier"], row)
    return list(rows_by_identifier.values())


def _version_record(row: dict[str, object]) -> dict[str, object]:
    """Return the version of a listing row to read: its newest published one, else its draft."""
    return row.get("most_recent_published_version") or row["draft_version"]


def _fetch_summary_for_row(*, client: DandiAPIClient, row: dict[str, object]) -> DandisetSummary:
    """Fetch the metadata for one dandiset listing row and summarize it."""
    identifier = row["identifier"]
    version_record = _version_record(row)
    resolved_version = version_record["version"]
    info = client.get(f"/dandisets/{identifier}/versions/{resolved_version}/info/")
    return _summarize_dandiset(
        identifier=identifier,
        version=resolved_version,
        metadata=info.get("metadata") or {},
        file_count=version_record["asset_count"],
        size_in_bytes=version_record["size"],
    )


def search_dandisets(
    *,
    terms: Sequence[str] = PHOTOMETRY_SEARCH_TERMS,
    max_results: int = 200,
    max_workers: int = 8,
) -> list[DandisetSummary]:
    """Search the DANDI Archive and summarize every dandiset that matches.

    Each term is run through the archive's full-text search over dandiset metadata and the
    results are unioned. Every hit's full metadata is then fetched -- the search endpoint
    returns only a name and an asset count -- which is what the catalog's columns and filters
    read.

    Parameters
    ----------
    terms : sequence of str, optional
        Search terms. Defaults to :data:`PHOTOMETRY_SEARCH_TERMS`, which assembles the
        fiber photometry catalog.
    max_results : int, optional
        Cap on how many matching dandisets are summarized.
    max_workers : int, optional
        How many metadata requests are issued concurrently.

    Returns
    -------
    list of DandisetSummary
        One row per matching dandiset, ordered by identifier.
    """
    with DandiAPIClient() as client:
        rows = _search_rows(client=client, terms=terms)[:max_results]
        logger.info("DANDI search for %r matched %d dandiset(s)", list(terms), len(rows))

        def summarize(row: dict[str, object]) -> DandisetSummary:
            return _fetch_summary_for_row(client=client, row=row)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            summaries = list(executor.map(summarize, rows))
    return sorted(summaries, key=lambda summary: summary.identifier)


def summarize_dandisets(identifiers: Sequence[str], *, max_workers: int = 8) -> list[DandisetSummary]:
    """Fetch and summarize the named dandisets.

    The crawl works from identifiers alone and only needs a dandiset's metadata once it has
    confirmed the dandiset is worth showing, which is what this fetches.

    Parameters
    ----------
    identifiers : sequence of str
        Six-digit dandiset IDs.
    max_workers : int, optional
        How many metadata requests to make at once.

    Returns
    -------
    list of DandisetSummary
        One summary per identifier, in the order given. Identifiers the archive does not
        return are omitted.
    """
    if not identifiers:
        return []
    with DandiAPIClient() as client:
        rows = []
        for identifier in identifiers:
            try:
                rows.append(client.get(f"/dandisets/{identifier}/"))
            except Exception as error:
                logger.warning("Could not fetch dandiset %s: %s", identifier, error)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(lambda row: _fetch_summary_for_row(client=client, row=row), rows))


def filter_dandisets(
    summaries: Sequence[DandisetSummary],
    *,
    query: str = "",
    species: Sequence[str] = (),
    brain_regions: Sequence[str] = (),
    indicators: Sequence[str] = (),
    approaches: Sequence[str] = (),
    minimum_subjects: int = 0,
    minimum_files: int = 0,
    published_only: bool = False,
) -> list[DandisetSummary]:
    """Narrow a catalog locally, without going back to the archive.

    Every criterion is a conjunction; within one criterion, a summary matches when it holds
    *any* of the requested values. ``query`` is matched as a substring against the summary's
    whole text blob, so it finds terms the vocabularies do not know.

    Parameters
    ----------
    summaries : sequence of DandisetSummary
        The catalog to narrow.
    query : str, optional
        Free text. Every whitespace-separated word must appear somewhere in the summary.
    species, brain_regions, indicators, approaches : sequence of str, optional
        Values to keep. An empty sequence disables that criterion.
    minimum_subjects, minimum_files : int, optional
        Lower bounds on the dandiset's subject and file counts.
    published_only : bool, optional
        Keep only dandisets that have a published version.

    Returns
    -------
    list of DandisetSummary
        The matching summaries, in their original order.
    """
    words = query.lower().split()

    def matches(summary: DandisetSummary) -> bool:
        if any(word not in summary.searchable_text for word in words):
            return False
        for requested, available in (
            (species, summary.species),
            (brain_regions, summary.brain_regions),
            (indicators, summary.indicators),
            (approaches, summary.approaches),
        ):
            if requested and not set(requested) & set(available):
                return False
        if summary.subject_count < minimum_subjects or summary.file_count < minimum_files:
            return False
        return not (published_only and not summary.is_published)

    return [summary for summary in summaries if matches(summary)]


def collect_filter_options(
    summaries: Sequence[DandisetSummary],
) -> dict[str, list[str]]:
    """Return the values each categorical filter can take across ``summaries``.

    Offering only values that are present keeps every option in the browser's dropdowns a
    choice that narrows the table rather than emptying it.

    Parameters
    ----------
    summaries : sequence of DandisetSummary
        The catalog the options are drawn from.

    Returns
    -------
    dict of {str: list of str}
        Sorted option lists keyed ``"species"``, ``"brain_regions"``, ``"indicators"`` and
        ``"approaches"``.
    """
    options: dict[str, list[str]] = {}
    for key, attribute in (
        ("species", "species"),
        ("brain_regions", "brain_regions"),
        ("indicators", "indicators"),
        ("approaches", "approaches"),
    ):
        values: set[str] = set()
        for summary in summaries:
            values.update(getattr(summary, attribute))
        options[key] = sorted(values)
    return options


@dataclass(frozen=True)
class AssetSummary:
    """One NWB asset of a dandiset, as the archive's asset listing describes it.

    Attributes
    ----------
    asset_id : str
        The asset's UUID, which addresses it independently of its path.
    path : str
        Path within the dandiset.
    size_in_bytes : int
        Size on the archive.
    content_url : str
        Directly readable URL for the asset's bytes, carried in the listing so that reading
        an asset costs no further metadata request.
    """

    asset_id: str
    path: str
    size_in_bytes: int
    content_url: str = ""


def list_nwb_assets(
    *,
    dandiset_id: str,
    version: str = "draft",
    glob_pattern: str = "*.nwb",
    max_assets: int = 20000,
) -> list[AssetSummary]:
    """List a dandiset's NWB assets, with their sizes and directly readable URLs.

    The archive's asset listing carries each asset's metadata inline when asked for it, so one
    paginated request supplies everything the browser and the photometry scan need. Resolving
    the same URLs one asset at a time would cost a redirect per asset.

    Parameters
    ----------
    dandiset_id : str
        Six-digit dandiset ID.
    version : str, optional
        Dandiset version to list.
    glob_pattern : str, optional
        Glob the archive filters asset paths by.
    max_assets : int, optional
        Cap on how many assets are returned.

    Returns
    -------
    list of AssetSummary
        One entry per matching asset, in the archive's own order.
    """
    with DandiAPIClient() as client:
        # Touch the dandiset first so an unknown ID raises NotFoundError before we paginate.
        client.get_dandiset(dandiset_id, version)
        assets = []
        rows = client.paginate(
            f"/dandisets/{dandiset_id}/versions/{version}/assets/",
            params={"glob": glob_pattern, "metadata": "true", "page_size": 1000},
        )
        for row in rows:
            assets.append(
                AssetSummary(
                    asset_id=row["asset_id"],
                    path=row["path"],
                    size_in_bytes=int(row["size"]),
                    content_url=_direct_content_url(row.get("metadata") or {}),
                )
            )
            if len(assets) >= max_assets:
                break
    return assets


def _direct_content_url(metadata: dict[str, object]) -> str:
    """Pick the asset URL that serves bytes directly, rather than redirecting to one."""
    content_urls = metadata.get("contentUrl") or []
    for url in content_urls:
        if not url.endswith("/download/"):
            return url
    return content_urls[0] if content_urls else ""


def format_byte_size(size_in_bytes: int) -> str:
    """Render a byte count in the largest unit that keeps it above 1.

    Parameters
    ----------
    size_in_bytes : int
        The count to render.

    Returns
    -------
    str
        A string such as ``"236 KB"`` or ``"23.5 GB"``.
    """
    size = float(size_in_bytes)
    for unit, decimals in (("B", 0), ("KB", 0), ("MB", 1)):
        if size < 1024:
            return f"{size:.{decimals}f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


# Excluded from coverage: resolves the URL over the network. Exercised by the local-only live
# suite tests/unit/utils/test_dandi_catalog_live.py (marker ``dandi_live``, deselected in CI).
def asset_content_url(*, dandiset_id: str, asset_path: str, version: str = "draft") -> str:  # pragma: no cover
    """Resolve the S3 URL an asset's bytes can be range-read from.

    Reading a public asset this way needs no DANDI API key: the redirect target is a public
    S3 object.

    Parameters
    ----------
    dandiset_id : str
        Six-digit dandiset ID.
    asset_path : str
        Path of the asset within the dandiset.
    version : str, optional
        Dandiset version the asset belongs to.

    Returns
    -------
    str
        A directly readable URL for the asset's bytes.
    """
    with DandiAPIClient() as client:
        asset = client.get_dandiset(dandiset_id, version).get_asset_by_path(asset_path)
        return asset.get_content_url(follow_redirects=1, strip_query=False)


# ----------------------------------------------------------------------------------------------
# Scan layer: which of a dandiset's assets hold fiber photometry
# ----------------------------------------------------------------------------------------------

# The container ndx-fiber-photometry writes its FiberPhotometryTable into, as a neurodata
# type rather than a name: the name is whichever one the file's author passed, and differs
# between writers. Every FiberPhotometryResponseSeries references a region of that table, so
# the container's presence under /general is equivalent to the file holding photometry, and
# costs an attribute read per child of /general -- all of them served out of the prefetched
# tail window, where walking acquisition and the processing modules instead would reach the
# series' own object headers, which sit beside their data in the middle of the file.
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


def asset_holds_photometry(asset: AssetSummary) -> bool:
    """Report whether one remote asset holds fiber photometry, without downloading it.

    Runs in a worker process, so it takes and returns only picklable values.

    Parameters
    ----------
    asset : AssetSummary
        The asset to inspect, carrying the URL its bytes are read from.

    Returns
    -------
    bool
        Whether the file declares a fiber photometry table.
    """
    try:
        reader = PrefetchedRemoteFile(content_url=asset.content_url, size_in_bytes=asset.size_in_bytes)
        with h5py.File(reader, mode="r") as file:
            return _find_fiber_photometry_container(file) is not None
    except Exception as error:
        # One unreadable asset -- embargoed, truncated, mid-upload -- should not abandon the
        # scan of every other asset in the dandiset.
        logger.warning("Could not scan %s for fiber photometry: %s", asset.path, error)
        return False


class PhotometryVerdictCache:
    """Verdicts already known for individual assets, kept between sessions.

    DANDI asset IDs address immutable blobs, so an answer never needs recomputing. That makes
    the archive-wide crawl a one-time cost rather than a recurring one, and it means the
    catalog's own verification and the crawl each shorten the other: whatever one of them
    scanned, the other skips.

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
        self._verdicts: dict[str, bool] = {}
        if self.path.is_file():
            try:
                self._verdicts = {key: bool(value) for key, value in json.loads(self.path.read_text()).items()}
            except (OSError, ValueError) as error:
                logger.warning(
                    "Ignoring unreadable photometry verdict cache %s: %s",
                    self.path,
                    error,
                )

    def __len__(self) -> int:
        return len(self._verdicts)

    def known(self, assets: Sequence[AssetSummary]) -> dict[str, bool]:
        """Return the verdicts already held for ``assets``, keyed by asset path."""
        return {asset.path: self._verdicts[asset.asset_id] for asset in assets if asset.asset_id in self._verdicts}

    def unknown(self, assets: Sequence[AssetSummary]) -> list[AssetSummary]:
        """Return the assets whose verdict is not held yet."""
        return [asset for asset in assets if asset.asset_id not in self._verdicts]

    def record(self, verdicts_by_asset_id: dict[str, bool]) -> None:
        """Take note of newly computed verdicts."""
        self._verdicts.update(verdicts_by_asset_id)

    def save(self) -> None:
        """Write the verdicts out, replacing whatever was there."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._verdicts))
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


def list_dandiset_references(*, page_size: int = 1000) -> list[DandisetReference]:
    """List every dandiset on the archive that holds at least one asset.

    Parameters
    ----------
    page_size : int, optional
        How many rows to request per page.

    Returns
    -------
    list of DandisetReference
        One entry per dandiset with assets, in the archive's own order.
    """
    references = []
    with DandiAPIClient() as client:
        for row in client.paginate("/dandisets/", params={"page_size": page_size}):
            version = row.get("most_recent_published_version") or row.get("draft_version")
            asset_count = (version or {}).get("asset_count") or 0
            if asset_count:
                references.append(
                    DandisetReference(
                        identifier=row["identifier"],
                        version=version["version"],
                        asset_count=asset_count,
                    )
                )
    return references


def order_for_crawl(references: Sequence[DandisetReference], *, first: Sequence[str] = ()) -> list[DandisetReference]:
    """Order a crawl so that the dandisets likeliest to settle quickly come first.

    ``first`` leads, in its own order, because those are the candidates something cheaper has
    already flagged. The rest follow smallest-first: proving a dandiset empty means reading
    every asset it has, so the handful of enormous dandisets -- which between them hold most
    of the archive, and are electrophysiology and imaging rather than photometry -- go last,
    where they delay nothing.

    Parameters
    ----------
    references : sequence of DandisetReference
        The dandisets to order.
    first : sequence of str, optional
        Identifiers to visit before the rest.

    Returns
    -------
    list of DandisetReference
        The same references, reordered.
    """
    priority = {identifier: index for index, identifier in enumerate(first)}
    return sorted(
        references,
        key=lambda reference: (
            priority.get(reference.identifier, len(priority)),
            reference.asset_count,
        ),
    )


def verify_dandisets(
    references: Sequence[DandisetReference],
    *,
    list_assets_function: object = list_nwb_assets,
    cache: PhotometryVerdictCache | None = None,
    process_count: int = SCAN_PROCESS_COUNT,
    on_verdict: Callable[[DandisetReference, bool], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, bool]:
    """Report which of ``references`` hold fiber photometry GuPPy can read.

    Each dandiset's assets are read largest first and the scan stops at the first one that
    holds photometry, so confirming a dandiset is usually a single file. Ruling one out has no
    such shortcut -- it means reading every asset -- which is why the answers arrive as
    confirmations quickly and rejections slowly.

    Parameters
    ----------
    references : sequence of DandisetReference
        The dandisets to verify, in the order they should be visited.
    list_assets_function : callable, optional
        Injection point for the asset listing.
    cache : PhotometryVerdictCache or None, optional
        Verdicts already known, which are consulted instead of rescanning and extended with
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
    dict of {str: bool}
        Dandiset identifier mapped to whether it holds fiber photometry. A dandiset the run
        stopped before reaching is absent rather than False.
    """
    verdicts: dict[str, bool] = {}
    if not references:
        return verdicts
    with ProcessPoolExecutor(max_workers=process_count) as pool:
        for reference in references:
            if should_stop is not None and should_stop():
                break
            holds = _dandiset_holds_photometry(
                reference=reference,
                pool=pool,
                list_assets_function=list_assets_function,
                cache=cache,
                chunk_size=process_count,
            )
            verdicts[reference.identifier] = holds
            if on_verdict is not None:
                on_verdict(reference, holds)
    if cache is not None:
        cache.save()
    return verdicts


def _dandiset_holds_photometry(
    *,
    reference: DandisetReference,
    pool: ProcessPoolExecutor,
    list_assets_function: object,
    cache: PhotometryVerdictCache | None,
    chunk_size: int,
) -> bool:
    """Whether any asset of one dandiset holds photometry, stopping at the first that does."""
    try:
        assets = list_assets_function(dandiset_id=reference.identifier, version=reference.version)
    except Exception as error:
        logger.warning("Could not list dandiset %s while crawling: %s", reference.identifier, error)
        return False

    if cache is not None:
        known = cache.known(assets)
        if any(known.values()):
            return True
        assets = cache.unknown(assets)

    # Largest first: a recording carries traces and a behavior-only sidecar does not, so this
    # is the order that reaches a positive soonest. It decides nothing -- every file it
    # reaches is still read -- so a dandiset whose photometry sits in its smallest file is
    # found too, just later.
    assets = sorted(assets, key=lambda asset: -asset.size_in_bytes)
    for start in range(0, len(assets), chunk_size):
        chunk = assets[start : start + chunk_size]
        results = list(pool.map(asset_holds_photometry, chunk))
        if cache is not None:
            cache.record({asset.asset_id: holds for asset, holds in zip(chunk, results, strict=True)})
        if any(results):
            return True
    return False


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


# ----------------------------------------------------------------------------------------------
# Probe layer: what an NWB file actually holds
# ----------------------------------------------------------------------------------------------

FIBER_PHOTOMETRY_RESPONSE_SERIES = "FiberPhotometryResponseSeries"
FIBER_PHOTOMETRY_TABLE = "FiberPhotometryTable"
# Event containers GuPPy's NWB reader turns into PSTH event stores: the ndx-events v0.2 types
# and the core ``EventsTable`` that replaced them.
EVENT_NEURODATA_TYPES = frozenset({"Events", "EventsTable", "LabeledEvents", "AnnotatedEventsTable"})


@dataclass(frozen=True)
class ChannelInfo:
    """One fiber photometry channel: a column of one response series, and the fiber behind it.

    Attributes
    ----------
    store_name : str
        The store id GuPPy's Label Stores page will show for this channel.
    series_name : str
        Name of the ``FiberPhotometryResponseSeries`` the channel belongs to.
    column_index : int or None
        Column within a multi-channel series; None for a single-channel series.
    location : str or None
        Recording site, from the fiber photometry table's ``location`` column.
    indicator : str or None
        Indicator label, from the ``Indicator`` object the table row points at.
    excitation_wavelength_in_nm, emission_wavelength_in_nm : float or None
        Wavelengths for the channel, where the file records them.
    suggested_label : str or None
        A ``signal_<site>`` / ``control_<site>`` store label consistent with the channel's
        excitation wavelength, or None when the wavelength or the site is unknown.
    """

    store_name: str
    series_name: str
    column_index: int | None
    location: str | None
    indicator: str | None
    excitation_wavelength_in_nm: float | None
    emission_wavelength_in_nm: float | None
    suggested_label: str | None


@dataclass(frozen=True)
class SeriesInfo:
    """Timing and shape of one ``FiberPhotometryResponseSeries``.

    Attributes
    ----------
    name : str
        Object name within the NWB file.
    sample_count : int
        Length of the series' leading dimension.
    channel_count : int
        Number of columns; 1 for a single-channel series.
    sampling_rate_in_hz : float or None
        Rate, when the series stores a regular ``starting_time``/``rate`` pair.
    duration_in_seconds : float or None
        Recording length, from the rate or from the timestamps' span.
    """

    name: str
    sample_count: int
    channel_count: int
    sampling_rate_in_hz: float | None
    duration_in_seconds: float | None


@dataclass(frozen=True)
class PhotometryProbe:
    """What one NWB file holds, read from its header.

    Attributes
    ----------
    series : tuple of SeriesInfo
        Every ``FiberPhotometryResponseSeries`` in the file.
    channels : tuple of ChannelInfo
        One entry per channel of every series, in GuPPy store order.
    event_names : tuple of str
        Names of the event containers GuPPy can align a PSTH to.
    session_description, session_start_time, identifier : str or None
        The file's own session-level fields.
    subject : dict of {str: str}
        The ``Subject`` group's fields, as strings.
    """

    series: tuple[SeriesInfo, ...] = ()
    channels: tuple[ChannelInfo, ...] = ()
    event_names: tuple[str, ...] = ()
    session_description: str | None = None
    session_start_time: str | None = None
    identifier: str | None = None
    subject: dict[str, str] = field(default_factory=dict)

    @property
    def has_photometry(self) -> bool:
        """Whether the file holds any fiber photometry response series."""
        return bool(self.series)

    @property
    def locations(self) -> tuple[str, ...]:
        """Distinct recording sites across the file's channels, in channel order."""
        return _unique(channel.location for channel in self.channels)

    @property
    def indicators(self) -> tuple[str, ...]:
        """Distinct indicators across the file's channels, in channel order."""
        return _unique(channel.indicator for channel in self.channels)


def _unique(values: Iterable[str | None]) -> tuple[str, ...]:
    """Return the non-empty values in order, without duplicates."""
    seen: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return tuple(seen)


def _decode(value: object) -> str | None:
    """Render an HDF5 scalar as a plain string, decoding bytes."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _read_scalar_dataset(file: h5py.File, path: str) -> str | None:
    """Read a scalar dataset as a string, or None when the file does not have it."""
    dataset = file.get(path)
    return None if dataset is None else _decode(dataset[()])


def _find_typed_child(group: h5py.Group | None, neurodata_type: str) -> h5py.Group | None:
    """Return the first child of ``group`` carrying ``neurodata_type``, or None."""
    if group is None:
        return None
    for member in group.values():
        if member.attrs.get("neurodata_type") == neurodata_type:
            return member
    return None


def _find_fiber_photometry_container(file: h5py.File) -> h5py.Group | None:
    """Return the file's ``FiberPhotometry`` container, whatever its author named it."""
    return _find_typed_child(file.get("general"), FIBER_PHOTOMETRY_LAB_META_DATA)


def _find_fiber_photometry_table(file: h5py.File) -> h5py.Group | None:
    """Return the file's ``FiberPhotometryTable``, whatever its author named it or its parent."""
    return _find_typed_child(_find_fiber_photometry_container(file), FIBER_PHOTOMETRY_TABLE)


def _find_photometry_series(file: h5py.File) -> dict[str, h5py.Group]:
    """Return every ``FiberPhotometryResponseSeries`` group, keyed by object name.

    Searches ``/acquisition`` and the modules under ``/processing`` -- the two places NWB puts
    time series -- rather than walking the whole file, so the scan costs a handful of metadata
    reads even when the file is streamed over the network.
    """
    series: dict[str, h5py.Group] = {}
    for group in _series_containers(file):
        for name, member in group.items():
            if member.attrs.get("neurodata_type") == FIBER_PHOTOMETRY_RESPONSE_SERIES:
                series[name] = member
    return series


def _series_containers(file: h5py.File) -> list[h5py.Group]:
    """Return the groups that directly contain time series: acquisition and each module."""
    containers = []
    acquisition = file.get("acquisition")
    if acquisition is not None:
        containers.append(acquisition)
    processing = file.get("processing")
    if processing is not None:
        containers.extend(processing.values())
    return containers


def _find_event_names(file: h5py.File) -> tuple[str, ...]:
    """Return the names of the event containers GuPPy can align a PSTH to.

    Looks wherever the supported event types land: ndx-events objects sit in ``/acquisition``
    or a processing module, while a core ``EventsTable`` sits in the file's ``/events`` group.
    """
    containers = _series_containers(file)
    events_group = file.get("events")
    if events_group is not None:
        containers.append(events_group)
    names = []
    for group in containers:
        for name, member in group.items():
            if member.attrs.get("neurodata_type") in EVENT_NEURODATA_TYPES and name not in names:
                names.append(name)
    return tuple(names)


def _read_fiber_photometry_table(file: h5py.File) -> list[dict[str, object]]:
    """Read the fiber photometry table into one dictionary per row.

    Each row describes one fiber: where it sat, which indicator it recorded, and the
    wavelengths involved. The indicator and excitation source are object references, and
    their labels and wavelengths live as attributes on the objects they point at.
    """
    table = _find_fiber_photometry_table(file)
    if table is None:
        return []
    locations = table["location"][:] if "location" in table else []
    rows = []
    for row_index in range(len(locations)):
        indicator = _referenced_attribute(file=file, table=table, column="indicator", row=row_index, key="label")
        excitation = _column_value(table=table, column="excitation_wavelength_in_nm", row=row_index)
        if excitation is None:
            excitation = _referenced_attribute(
                file=file,
                table=table,
                column="excitation_source",
                row=row_index,
                key="excitation_wavelength_in_nm",
            )
        rows.append(
            {
                "location": _decode(locations[row_index]),
                "indicator": _decode(indicator),
                "excitation_wavelength_in_nm": _as_float(excitation),
                "emission_wavelength_in_nm": _as_float(
                    _column_value(table=table, column="emission_wavelength_in_nm", row=row_index)
                ),
            }
        )
    return rows


def _column_value(*, table: h5py.Group, column: str, row: int) -> object:
    """Read one cell of a dynamic-table column, or None when the column is absent."""
    dataset = table.get(column)
    return None if dataset is None else dataset[row]


def _referenced_attribute(*, file: h5py.File, table: h5py.Group, column: str, row: int, key: str) -> object:
    """Follow a table cell's object reference and read ``key`` off the object it points at."""
    reference = _column_value(table=table, column=column, row=row)
    if reference is None:
        return None
    return file[reference].attrs.get(key)


def _as_float(value: object) -> float | None:
    """Render an HDF5 numeric scalar as a float, or None when it is missing."""
    return None if value is None else float(value)


def _suggested_label(*, location: str | None, excitation_wavelength_in_nm: float | None) -> str | None:
    """Name the GuPPy store label a channel's wavelength and site imply."""
    if not location or excitation_wavelength_in_nm is None:
        return None
    role = "control" if excitation_wavelength_in_nm <= ISOSBESTIC_WAVELENGTH_CEILING_IN_NM else "signal"
    return f"{role}_{location}"


def _series_timing(series: h5py.Group, sample_count: int) -> tuple[float | None, float | None]:
    """Return the series' sampling rate and duration, from whichever timing it stores."""
    starting_time = series.get("starting_time")
    if starting_time is not None:
        rate = starting_time.attrs.get("rate")
        if rate:
            return float(rate), sample_count / float(rate)
    timestamps = series.get("timestamps")
    if timestamps is not None and len(timestamps) > 1:
        first, last = float(timestamps[0]), float(timestamps[-1])
        span = last - first
        return (len(timestamps) - 1) / span if span > 0 else None, span
    return None, None


def probe_photometry(*, file: h5py.File) -> PhotometryProbe:
    """Report the fiber photometry content of an open NWB file.

    Reads only header metadata -- object names, dataset shapes, table columns and device
    attributes -- so the call is affordable against a file streamed from the archive.

    Parameters
    ----------
    file : h5py.File
        An open NWB file, local or backed by a remote reader.

    Returns
    -------
    PhotometryProbe
        The file's response series, per-channel fiber details, event containers and
        session-level fields. ``has_photometry`` is False for a file that holds none.
    """
    table_rows = _read_fiber_photometry_table(file)
    series_infos = []
    channels = []
    for name, series in _find_photometry_series(file).items():
        data = series["data"]
        sample_count = int(data.shape[0])
        channel_count = int(data.shape[1]) if data.ndim == 2 else 1
        sampling_rate, duration = _series_timing(series, sample_count)
        series_infos.append(
            SeriesInfo(
                name=name,
                sample_count=sample_count,
                channel_count=channel_count,
                sampling_rate_in_hz=sampling_rate,
                duration_in_seconds=duration,
            )
        )
        region = series.get("fiber_photometry_table_region")
        row_indices = [int(index) for index in region[:]] if region is not None else []
        for column_index in range(channel_count):
            # A single-channel series keeps its own name as the store id; a multi-channel one
            # contributes one store per column, which is how GuPPy's NWB reader names them.
            store_name = name if data.ndim == 1 else f"{name}_{column_index}"
            row = table_rows[row_indices[column_index]] if column_index < len(row_indices) else {}
            channels.append(
                ChannelInfo(
                    store_name=store_name,
                    series_name=name,
                    column_index=None if data.ndim == 1 else column_index,
                    location=row.get("location"),
                    indicator=row.get("indicator"),
                    excitation_wavelength_in_nm=row.get("excitation_wavelength_in_nm"),
                    emission_wavelength_in_nm=row.get("emission_wavelength_in_nm"),
                    suggested_label=_suggested_label(
                        location=row.get("location"),
                        excitation_wavelength_in_nm=row.get("excitation_wavelength_in_nm"),
                    ),
                )
            )

    subject = file.get("general/subject")
    return PhotometryProbe(
        series=tuple(series_infos),
        channels=tuple(channels),
        event_names=_find_event_names(file),
        session_description=_read_scalar_dataset(file, "session_description"),
        session_start_time=_read_scalar_dataset(file, "session_start_time"),
        identifier=_read_scalar_dataset(file, "identifier"),
        subject={} if subject is None else {key: _decode(subject[key][()]) for key in subject},
    )


@dataclass(frozen=True)
class ExampleTraces:
    """A decimated slice off the front of one response series, ready to plot.

    Attributes
    ----------
    series_name : str
        The series the slice came from.
    timestamps : numpy.ndarray
        Times in seconds, one per returned sample.
    traces : dict of {str: numpy.ndarray}
        Store name to that channel's samples.
    """

    series_name: str
    timestamps: np.ndarray
    traces: dict[str, np.ndarray]


def read_example_traces(
    *,
    file: h5py.File,
    probe: PhotometryProbe,
    series_name: str | None = None,
    duration_in_seconds: float = DEFAULT_TRACE_DURATION_IN_SECONDS,
    max_points: int = DEFAULT_TRACE_POINTS,
) -> ExampleTraces:
    """Read the opening seconds of one response series, decimated for display.

    Parameters
    ----------
    file : h5py.File
        The open NWB file ``probe`` was taken from.
    probe : PhotometryProbe
        The file's probe, which supplies the series' timing and channel names.
    series_name : str or None, optional
        Series to read. Defaults to the file's first response series.
    duration_in_seconds : float, optional
        How much of the recording to read from its start. A series whose file records no
        sampling rate is read whole, since there is then nothing to convert seconds into
        samples with.
    max_points : int, optional
        Upper bound on the samples returned per channel; the slice is strided down to it.

    Returns
    -------
    ExampleTraces
        Timestamps and one trace per channel of the chosen series.
    """
    series_info = next(info for info in probe.series if series_name in (None, info.name))
    series = _find_photometry_series(file)[series_info.name]
    data = series["data"]

    sample_count = series_info.sample_count
    if series_info.sampling_rate_in_hz:
        sample_count = min(sample_count, int(duration_in_seconds * series_info.sampling_rate_in_hz))
    sample_count = max(sample_count, 1)
    step = max(1, ceil(sample_count / max_points))

    values = data[:sample_count:step]
    timestamps_dataset = series.get("timestamps")
    if timestamps_dataset is not None:
        timestamps = np.asarray(timestamps_dataset[:sample_count:step], dtype=float)
    else:
        starting_time = series.get("starting_time")
        offset = float(starting_time[()]) if starting_time is not None else 0.0
        rate = series_info.sampling_rate_in_hz or 1.0
        timestamps = offset + np.arange(len(values)) * step / rate

    channels = [channel for channel in probe.channels if channel.series_name == series_info.name]
    traces = {
        channel.store_name: np.asarray(values if values.ndim == 1 else values[:, channel.column_index], dtype=float)
        for channel in channels
    }
    return ExampleTraces(series_name=series_info.name, timestamps=timestamps, traces=traces)


@dataclass(frozen=True)
class AssetPreview:
    """A probed asset and the example traces read in the same streaming session.

    Attributes
    ----------
    dandiset_id, asset_path : str
        Where the asset came from.
    probe : PhotometryProbe
        What the file holds.
    traces : ExampleTraces or None
        Opening slice of the file's first response series; None when it holds no photometry.
    """

    dandiset_id: str
    asset_path: str
    probe: PhotometryProbe
    traces: ExampleTraces | None


# Excluded from coverage: opens a network connection to the archive. Exercised by the local-only
# live suite (marker ``dandi_live``, deselected in CI); offline tests call the probe and trace
# readers against local NWB files instead.
def open_remote_hdf5(*, content_url: str) -> h5py.File:  # pragma: no cover
    """Open an HDF5 file over HTTP range requests, without downloading it.

    Parameters
    ----------
    content_url : str
        Directly readable URL for the file's bytes, as
        :func:`asset_content_url` returns.

    Returns
    -------
    h5py.File
        An open file the caller owns and must close.
    """
    import remfile

    return h5py.File(remfile.File(content_url), mode="r")


# Excluded from coverage: streams from the archive. Exercised by the local-only live suite
# tests/unit/utils/test_dandi_catalog_live.py (marker ``dandi_live``, deselected in CI); the
# probe and trace readers it composes are unit-tested against local NWB files.
def preview_asset(  # pragma: no cover
    *,
    dandiset_id: str,
    asset_path: str,
    version: str = "draft",
    series_name: str | None = None,
    trace_duration_in_seconds: float = DEFAULT_TRACE_DURATION_IN_SECONDS,
    max_points: int = DEFAULT_TRACE_POINTS,
) -> AssetPreview:
    """Stream one asset's header and the opening seconds of its traces.

    Both reads share a single streaming session, so the preview costs one file open rather
    than two. No DANDI API key is needed for a public dandiset.

    Parameters
    ----------
    dandiset_id : str
        Six-digit dandiset ID.
    asset_path : str
        Path of the asset within the dandiset.
    version : str, optional
        Dandiset version the asset belongs to.
    series_name : str or None, optional
        Response series to read the traces from. Defaults to the file's first.
    trace_duration_in_seconds : float, optional
        Seconds of the recording to read for the example traces.
    max_points : int, optional
        Upper bound on the samples returned per channel.

    Returns
    -------
    AssetPreview
        The probe, and the example traces when the asset holds photometry.
    """
    content_url = asset_content_url(dandiset_id=dandiset_id, asset_path=asset_path, version=version)
    file = open_remote_hdf5(content_url=content_url)
    try:
        probe = probe_photometry(file=file)
        traces = (
            read_example_traces(
                file=file,
                probe=probe,
                series_name=series_name,
                duration_in_seconds=trace_duration_in_seconds,
                max_points=max_points,
            )
            if probe.has_photometry
            else None
        )
    finally:
        file.close()
    return AssetPreview(dandiset_id=dandiset_id, asset_path=asset_path, probe=probe, traces=traces)

"""Search the DANDI Archive for fiber photometry datasets and list their NWB assets.

Everything here talks to the DANDI REST API. It runs the archive's full-text search, pulls each
hit's dandiset metadata and reduces it to a :class:`DandisetSummary` the search panel can tabulate
and filter, and it lists a dandiset's NWB assets with the URLs their bytes are readable from.

The search can only go through free text, because DANDI's structured metadata does not describe
fiber photometry: ``assetsSummary.variableMeasured`` is built by dandi-cli from the core NWB types
it knows, and ``FiberPhotometryResponseSeries`` is not among them, so no dandiset in the archive
lists it. Anything authoritative -- which files carry photometry, from which sites, with which
indicator -- has to come from the files themselves, which is what :mod:`guppy.utils.dandi_filter`
and :mod:`guppy.utils.dandi_preview` read.
"""

import logging
import re
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from dandi.dandiapi import DandiAPIClient

logger = logging.getLogger(__name__)

# Terms handed to the archive's full-text search to assemble the candidate catalog. A fiber
# photometry dandiset names the method somewhere in its title, description or keywords, and
# nothing else measurably adds to it: sensor families return only dandisets the one term already
# returns, while broader words like "gcamp" and "fluorescence" pull in two-photon and widefield
# imaging datasets that GuPPy cannot read. What the search misses is recovered by the crawl
# rather than by more terms.
PHOTOMETRY_SEARCH_TERMS = ("photometry",)

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

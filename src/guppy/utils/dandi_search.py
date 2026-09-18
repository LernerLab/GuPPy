"""Search the DANDI Archive for fiber photometry datasets and list their NWB assets.

Everything here talks to the DANDI REST API. It runs the archive's full-text search, pulls each
hit's dandiset metadata and reduces it to a :class:`DandisetSummary` the search panel can
tabulate, and it lists a dandiset's NWB assets with the URLs their bytes are readable from.

The search can only go through free text, because DANDI's structured metadata does not describe
fiber photometry: ``assetsSummary.variableMeasured`` is built by dandi-cli from the core NWB types
it knows, and ``FiberPhotometryResponseSeries`` is not among them, so no dandiset in the archive
lists it. A summary therefore reports only what DANDI itself asserts -- title, abstract, keywords,
species, approaches and totals. Anything authoritative, including which files carry photometry and
from which site with which indicator, has to come from the files themselves, which is what
:mod:`guppy.utils.dandi_filter` and :mod:`guppy.utils.dandi_preview` read.
"""

import logging
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
    """

    identifier: str
    version: str
    name: str
    description: str
    species: tuple[str, ...]
    approaches: tuple[str, ...]
    keywords: tuple[str, ...]
    subject_count: int
    file_count: int
    size_in_bytes: int
    contributors: tuple[str, ...]
    license_terms: tuple[str, ...]
    url: str
    is_published: bool


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
    name = metadata.get("name") or ""
    description = metadata.get("description") or ""

    return DandisetSummary(
        identifier=identifier,
        version=version,
        name=name,
        description=description,
        species=tuple(species.split(" - ")[0] for species in _names(assets_summary.get("species") or ())),
        approaches=_names(assets_summary.get("approach") or ())
        + _names(assets_summary.get("measurementTechnique") or ()),
        keywords=keywords,
        subject_count=int(assets_summary.get("numberOfSubjects") or 0),
        file_count=int(file_count if file_count is not None else (assets_summary.get("numberOfFiles") or 0)),
        size_in_bytes=int(size_in_bytes if size_in_bytes is not None else (assets_summary.get("numberOfBytes") or 0)),
        contributors=_names(metadata.get("contributor") or ()),
        license_terms=tuple(metadata.get("license") or ()),
        url=metadata.get("url") or f"https://dandiarchive.org/dandiset/{identifier}",
        is_published=version != "draft",
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
    returns only a name and an asset count -- which is what the catalog's columns and the
    dandiset page read.

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

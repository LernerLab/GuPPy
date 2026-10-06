"""Mixin of contract tests for :mod:`guppy.utils.dandi_search`.

The contract is what has to hold of any archive the search reads, so it is bound twice: once to
the in-memory stand-in in ``test_dandi_search.py`` and once to the real DANDI Archive in
``test_dandi_search_live.py``. Everything here is about the shape of what comes back, because
that is what an in-memory fake cannot vouch for -- it agrees with whatever we wrote it to return,
including a field the archive has since renamed.

Child test classes must define the following class-level attributes:

search_terms : tuple of str
    Terms handed to the search.
expected_identifier : str
    A dandiset the search must surface, described in detail by the contract.
listing_dandiset_id : str
    The dandiset whose assets are listed.
expected_asset_path : str
    An NWB asset that listing must contain.

and may override the ``archive_ready`` fixture to install a stand-in before the search runs.
"""

import pytest

from guppy.utils.dandi_search import list_nwb_assets, search_dandisets


class DandiSearchTestMixin:
    """Contract tests for searching an archive and listing one dandiset's NWB assets."""

    search_terms: tuple[str, ...]
    expected_identifier: str
    listing_dandiset_id: str
    expected_asset_path: str

    @pytest.fixture
    def archive_ready(self):
        """Nothing to install; the live binding reads the archive itself."""
        return None

    @pytest.fixture
    def summaries(self, archive_ready):
        return search_dandisets(terms=self.search_terms)

    @pytest.fixture
    def assets(self, archive_ready):
        return list_nwb_assets(dandiset_id=self.listing_dandiset_id)

    def test_the_search_surfaces_the_expected_dandiset(self, summaries):
        assert self.expected_identifier in {summary.identifier for summary in summaries}

    def test_every_hit_is_summarized(self, summaries):
        assert summaries
        assert all(summary.name for summary in summaries)
        assert all(summary.identifier.isdigit() for summary in summaries)
        assert all(isinstance(summary.species, tuple) for summary in summaries)
        assert all(isinstance(term, str) for summary in summaries for term in summary.species)

    def test_the_expected_dandiset_carries_what_the_catalog_shows(self, summaries):
        summary = next(summary for summary in summaries if summary.identifier == self.expected_identifier)
        assert summary.name
        assert summary.version
        assert summary.url.startswith("https://")
        assert summary.subject_count > 0
        assert summary.file_count > 0
        assert summary.size_in_bytes > 0

    def test_the_listing_holds_nwb_assets_only(self, assets):
        assert assets
        assert all(asset.path.endswith(".nwb") for asset in assets)
        assert self.expected_asset_path in {asset.path for asset in assets}

    def test_every_listed_asset_carries_an_id_and_a_size(self, assets):
        assert all(asset.asset_id for asset in assets)
        assert all(asset.size_in_bytes > 0 for asset in assets)

    def test_the_listing_carries_a_url_the_bytes_can_be_read_from(self, assets):
        # The archive serves each asset from two URLs: one that redirects and the S3 blob the
        # scan range-reads. Taking the redirect would cost a round trip per read.
        urls = [asset.content_url for asset in assets]
        assert all(url.startswith("https://") for url in urls)
        assert not any(url.endswith("/download/") for url in urls)

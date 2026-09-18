"""Live contract tests for :mod:`guppy.utils.dandi_search`.

These hit the real DANDI Archive and are intended for **local use only**. They are deselected in
CI (see ``.github/workflows/run-tests.yml``) because the archive is subject to network flakiness
and its contents change as datasets are published.

Run locally with::

    pytest tests/unit/utils/test_dandi_search_live.py -v -m dandi_live

The contract itself lives in ``dandi_search_test_mixin.py`` and is inherited, so what is written
here is only the archive-specific detail: the dandiset and asset pinned, and the values that only
the real archive can confirm. They pin the same dandiset and asset as the live streaming suite
(``tests/unit/extractors/test_dandi_nwb_live.py``), so the two describe one recording.
"""

import pytest

from guppy.utils.dandi_search import PHOTOMETRY_SEARCH_TERMS

from .dandi_search_test_mixin import DandiSearchTestMixin

DANDISET_ID = "000971"
ASSET_PATH = "sub-112-283/sub-112-283_ses-FP-PS-2019-06-20T09-32-04_behavior.nwb"


@pytest.mark.dandi_live
class TestDandiSearchAgainstTheArchive(DandiSearchTestMixin):
    """The search contract, bound to the real DANDI REST API."""

    search_terms = PHOTOMETRY_SEARCH_TERMS
    expected_identifier = DANDISET_ID
    listing_dandiset_id = DANDISET_ID
    expected_asset_path = ASSET_PATH

    def test_the_pinned_dandiset_is_summarized_from_its_metadata(self, summaries):
        summary = next(summary for summary in summaries if summary.identifier == DANDISET_ID)
        assert summary.species == ("Mus musculus",)
        assert "Dorsal striatum" in summary.brain_regions

    def test_the_photometry_search_returns_a_catalog_worth_showing(self, summaries):
        assert len(summaries) > 10

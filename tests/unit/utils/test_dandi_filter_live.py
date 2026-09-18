"""Live contract tests for :mod:`guppy.utils.dandi_filter`.

These hit the real DANDI Archive and are intended for **local use only**. They are deselected in
CI (see ``.github/workflows/run-tests.yml``) because the archive is subject to network flakiness
and its contents change as datasets are published.

Run locally with::

    pytest tests/unit/utils/test_dandi_filter_live.py -v -m dandi_live

The contract itself lives in ``dandi_filter_test_mixin.py`` and is inherited. The asset scan is
bound to one subject's folder rather than the whole dandiset, because ruling an asset out is a
read apiece and 000971 holds four thousand of them.
"""

import pytest

from guppy.utils.dandi_filter import (
    SCAN_PROCESS_COUNT,
    DandisetReference,
    verify_dandisets,
)
from guppy.utils.dandi_search import list_nwb_assets

from .dandi_filter_test_mixin import DandiFilterTestMixin

DANDISET_ID = "000971"
ASSET_PATH = "sub-112-283/sub-112-283_ses-FP-PS-2019-06-20T09-32-04_behavior.nwb"
SUBJECT_FOLDER = "sub-112-283/"


@pytest.mark.dandi_live
class TestDandiFilterAgainstTheArchive(DandiFilterTestMixin):
    """The filter contract, bound to real assets streamed from the archive."""

    expected_photometry_paths = [ASSET_PATH]
    process_count = SCAN_PROCESS_COUNT

    @pytest.fixture(scope="class")
    def asset_listing(self):
        """One subject's assets: 37 recordings and behavior files, one of which holds traces."""
        assets = list_nwb_assets(dandiset_id=DANDISET_ID)
        return [asset for asset in assets if asset.path.startswith(SUBJECT_FOLDER)]

    @pytest.fixture(scope="class")
    def photometry_asset(self, asset_listing):
        return next(asset for asset in asset_listing if asset.path == ASSET_PATH)

    @pytest.fixture(scope="class")
    def behavior_only_asset(self, asset_listing):
        return next(asset for asset in asset_listing if asset.path != ASSET_PATH)

    @pytest.fixture(scope="class")
    def photometry_dandiset(self):
        return DandisetReference(identifier=DANDISET_ID, version="draft", asset_count=4139)

    def test_a_dandiset_that_only_writes_about_photometry_is_ruled_out(self):
        # 000251 mentions photometry in its text but stores fluorescence without the
        # ndx-fiber-photometry types, so GuPPy cannot read it and it must not be offered.
        # Ruling a dandiset out has no shortcut -- this reads all 513 of its assets.
        reference = DandisetReference(identifier="000251", version="draft", asset_count=513)
        assert verify_dandisets([reference]) == {"000251": False}

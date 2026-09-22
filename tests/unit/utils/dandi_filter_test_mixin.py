"""Mixin of contract tests for :mod:`guppy.utils.dandi_filter`.

Bound twice: once in ``test_dandi_filter.py`` against a local server that answers byte ranges as
the archive does, serving real NWB files, and once in ``test_dandi_filter_live.py`` against the
archive itself. The offline binding is a faithful substitute for the mechanism -- real sockets,
real ``Range`` headers, real h5py -- so what the live binding adds is that files written by other
people's conversions still answer the way ours do.

Child test classes must define the following class-level attributes:

expected_photometry_paths : list of str
    The assets in ``asset_listing`` that hold photometry, in scan order.
process_count : int
    How many assets to scan at once.

and the following fixtures:

photometry_asset : AssetSummary
    An asset known to hold a ``FiberPhotometryResponseSeries``.
behavior_only_asset : AssetSummary
    An asset known to hold none.
asset_listing : list of AssetSummary
    The assets a scan is run over.
photometry_dandiset : DandisetReference
    A dandiset known to hold photometry.

Child classes may also override ``list_assets_function`` to inject the asset listing.
"""

import pytest

from guppy.utils.dandi_filter import (
    asset_holds_photometry,
    scan_assets_for_photometry,
    verify_dandisets,
)
from guppy.utils.dandi_search import list_nwb_assets


class DandiFilterTestMixin:
    """Contract tests for deciding what holds photometry, per asset and per dandiset."""

    expected_photometry_paths: list[str]
    process_count: int = 2

    @pytest.fixture
    def list_assets_function(self):
        """The listing a dandiset is verified from; the live binding reads the archive."""
        return list_nwb_assets

    def test_a_photometry_asset_is_recognized(self, photometry_asset):
        assert asset_holds_photometry(photometry_asset) is True

    def test_a_behavior_only_asset_is_rejected(self, behavior_only_asset):
        assert asset_holds_photometry(behavior_only_asset) is False

    def test_every_scanned_asset_gets_a_verdict_keyed_by_its_path(self, asset_listing):
        verdicts = scan_assets_for_photometry(asset_listing, process_count=self.process_count)
        assert set(verdicts) == {asset.path for asset in asset_listing}

    def test_only_the_photometry_assets_answer_yes(self, asset_listing):
        verdicts = scan_assets_for_photometry(asset_listing, process_count=self.process_count)
        assert [path for path, holds in verdicts.items() if holds] == self.expected_photometry_paths

    def test_progress_is_reported_once_per_asset(self, asset_listing):
        completed = []
        scan_assets_for_photometry(
            asset_listing,
            process_count=self.process_count,
            progress_callback=completed.append,
        )
        assert completed == list(range(1, len(asset_listing) + 1))

    def test_a_dandiset_holding_photometry_is_confirmed(self, photometry_dandiset, list_assets_function):
        verdicts = verify_dandisets(
            [photometry_dandiset],
            list_assets_function=list_assets_function,
            process_count=self.process_count,
        )
        assert verdicts == {photometry_dandiset.identifier: True}

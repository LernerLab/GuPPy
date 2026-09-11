"""Live contract tests for the DANDI catalog search and the streaming asset preview.

These tests hit the real DANDI Archive and are intended for **local use only**. They are
deselected in CI (see ``.github/workflows/run-tests.yml``) because the archive is subject to
network flakiness and its contents change as datasets are published.

Run locally with::

    pytest tests/unit/utils/test_dandi_catalog_live.py -v -m dandi_live

They pin the same dandiset and asset as the live streaming suite
(``tests/unit/extractors/test_dandi_nwb_live.py``), so the two describe one recording.
Streaming a public asset for a preview needs no DANDI API key.
"""

import numpy as np
import pytest

from guppy.utils.dandi_catalog import (
    PHOTOMETRY_SEARCH_TERMS,
    filter_assets,
    list_nwb_assets,
    preview_asset,
    search_dandisets,
)

DANDISET_ID = "000971"
ASSET_PATH = "sub-112-283/sub-112-283_ses-FP-PS-2019-06-20T09-32-04_behavior.nwb"
# Site, indicator and wavelengths the how-to guide documents for this recording.
EXPECTED_STORE_NAMES = [f"fiber_photometry_response_series_{index}" for index in range(4)]
EXPECTED_SUGGESTED_LABELS = ["signal_DMS", "control_DMS", "signal_DLS", "control_DLS"]


@pytest.mark.dandi_live
class TestLiveCatalogSearch:
    def test_photometry_search_finds_the_pinned_dandiset(self):
        summaries = search_dandisets(terms=PHOTOMETRY_SEARCH_TERMS)
        by_id = {summary.identifier: summary for summary in summaries}
        assert DANDISET_ID in by_id
        summary = by_id[DANDISET_ID]
        assert summary.species == ("Mus musculus",)
        assert "Dorsal striatum" in summary.brain_regions
        assert summary.subject_count > 0
        assert summary.file_count > 0

    def test_every_hit_is_summarized(self):
        summaries = search_dandisets(terms=("photometry",))
        assert len(summaries) > 10
        assert all(summary.name for summary in summaries)
        assert all(summary.identifier.isdigit() for summary in summaries)


@pytest.mark.dandi_live
class TestLiveAssetListing:
    def test_lists_the_dandiset_assets_with_their_sizes(self):
        assets = list_nwb_assets(dandiset_id=DANDISET_ID)
        by_path = {asset.path: asset for asset in assets}
        assert ASSET_PATH in by_path
        assert all(asset.path.endswith(".nwb") for asset in assets)
        # The recordings are hundreds of megabytes; the behavior-only files are a few hundred KB.
        assert by_path[ASSET_PATH].size_in_bytes > 100_000_000

    def test_a_size_floor_isolates_the_recordings(self):
        assets = list_nwb_assets(dandiset_id=DANDISET_ID)
        recordings = filter_assets(assets, minimum_size_in_bytes=5 * 1024 * 1024)
        assert 0 < len(recordings) < len(assets) / 10


@pytest.mark.dandi_live
class TestLiveAssetPreview:
    @pytest.fixture(scope="class")
    def preview(self):
        return preview_asset(dandiset_id=DANDISET_ID, asset_path=ASSET_PATH)

    def test_reports_the_recordings_four_channels(self, preview):
        assert preview.probe.has_photometry is True
        assert [channel.store_name for channel in preview.probe.channels] == EXPECTED_STORE_NAMES
        assert preview.probe.locations == ("DMS", "DLS")
        assert preview.probe.indicators == ("GCaMP7b",)

    def test_suggested_labels_match_the_documented_mapping(self, preview):
        assert [channel.suggested_label for channel in preview.probe.channels] == EXPECTED_SUGGESTED_LABELS

    def test_reports_the_series_timing_and_the_subject(self, preview):
        (series,) = preview.probe.series
        assert series.channel_count == 4
        assert series.sampling_rate_in_hz == pytest.approx(1017.25, abs=0.01)
        assert series.duration_in_seconds == pytest.approx(3698.0, abs=1.0)
        assert preview.probe.subject["subject_id"] == "112.283"
        assert preview.probe.subject["species"] == "Mus musculus"

    def test_finds_the_behavioral_event_objects(self, preview):
        assert "right_nose_poke_times" in preview.probe.event_names

    def test_example_traces_cover_the_requested_window(self, preview):
        traces = preview.traces
        assert list(traces.traces) == EXPECTED_STORE_NAMES
        assert traces.timestamps[0] == pytest.approx(0.0)
        assert traces.timestamps[-1] == pytest.approx(60.0, abs=0.1)
        for values in traces.traces.values():
            assert values.shape == traces.timestamps.shape
            assert np.isfinite(values).all()

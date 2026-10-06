"""Live contract tests for :mod:`guppy.utils.dandi_preview`.

These hit the real DANDI Archive and are intended for **local use only**. They are deselected in
CI (see ``.github/workflows/run-tests.yml``) because the archive is subject to network flakiness
and its contents change as datasets are published.

Run locally with::

    pytest tests/unit/utils/test_dandi_preview_live.py -v -m dandi_live

The contract itself lives in ``dandi_preview_test_mixin.py`` and is inherited. What is written
here is the pinned asset and the values only a real conversion can confirm -- its sampling rate,
duration and subject. Streaming a public asset needs no DANDI API key. The asset is the one the
live streaming suite reads (``tests/unit/extractors/test_dandi_nwb_live.py``), so the two
describe one recording.
"""

import pytest

from guppy.utils.dandi_preview import DEFAULT_TRACE_DURATION_IN_SECONDS, preview_asset

from .dandi_preview_test_mixin import DandiPreviewTestMixin

DANDISET_ID = "000971"
ASSET_PATH = "sub-112-283/sub-112-283_ses-FP-PS-2019-06-20T09-32-04_behavior.nwb"


@pytest.mark.dandi_live
class TestDandiPreviewAgainstTheArchive(DandiPreviewTestMixin):
    """The preview contract, bound to a real asset streamed from the archive."""

    # Site, indicator and wavelengths the how-to guide documents for this recording.
    expected_store_names = [f"fiber_photometry_response_series_{index}" for index in range(4)]
    expected_suggested_labels = ["signal_DMS", "control_DMS", "signal_DLS", "control_DLS"]
    expected_locations = ("DMS", "DLS")
    expected_indicators = ("GCaMP7b",)
    expected_event_name = "right_nose_poke_times"
    trace_duration_in_seconds = DEFAULT_TRACE_DURATION_IN_SECONDS

    @pytest.fixture(scope="class")
    def preview(self):
        return preview_asset(dandiset_id=DANDISET_ID, asset_path=ASSET_PATH)

    @pytest.fixture
    def probe(self, preview):
        return preview.probe

    @pytest.fixture
    def traces(self, preview):
        return preview.traces

    def test_the_series_timing_matches_the_recording(self, probe):
        (series,) = probe.series
        assert series.sampling_rate_in_hz == pytest.approx(1017.25, abs=0.01)
        assert series.duration_in_seconds == pytest.approx(3698.0, abs=1.0)

    def test_the_subject_is_read(self, probe):
        assert probe.subject["subject_id"] == "112.283"
        assert probe.subject["species"] == "Mus musculus"

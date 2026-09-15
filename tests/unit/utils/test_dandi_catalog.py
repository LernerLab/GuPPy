"""Tests for the DANDI catalog search and the NWB photometry probe.

The catalog layer talks to the DANDI REST API, so ``DandiAPIClient`` is replaced with an
in-memory archive built from recorded response shapes. The probe layer reads HDF5, so it runs
against the real mock NWB files in ``stubbed_testing_data/nwb/`` — one per supported
ndx-fiber-photometry / events combination — with no network access anywhere.
"""

from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path

import h5py
import numpy as np
import pytest
from dandi.exceptions import NotFoundError

from guppy.utils import dandi_catalog
from guppy.utils.dandi_catalog import (
    AssetSummary,
    DandisetSummary,
    collect_filter_options,
    filter_assets,
    filter_dandisets,
    find_vocabulary_terms,
    format_byte_size,
    list_nwb_assets,
    probe_photometry,
    read_example_traces,
    search_dandisets,
)
from guppy_test_data import STUBBED_TESTING_DATA

NWB_DATA = STUBBED_TESTING_DATA / "nwb"
MOCK_NWB_FILES = {
    name: NWB_DATA / name / f"{name}.nwb"
    for name in (
        "mock_nwbfile_ndx_fiber_photometry_v0_1_ndx_events_v0_2",
        "mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2",
        "mock_nwbfile_ndx_fiber_photometry_v0_2_core_events",
    )
}


# ---------------------------------------------------------------------------------------------
# In-memory stand-in for the DANDI REST API
# ---------------------------------------------------------------------------------------------


class FakeRemoteAsset:
    def __init__(self, path, size):
        self.identifier = f"asset-{path}"
        self.path = path
        self.size = size


class FakeRemoteDandiset:
    def __init__(self, assets):
        self._assets = assets

    def get_assets_by_glob(self, pattern, order=None):
        return [asset for asset in self._assets if fnmatch(asset.path, pattern)]


class FakeDandiAPIClient:
    """In-memory stand-in for ``dandi.dandiapi.DandiAPIClient``.

    ``search_results`` maps a search term to the identifiers it matches; ``metadata_by_id``
    holds each dandiset's raw metadata; ``assets_by_id`` holds its asset listing.
    """

    search_results = {}
    metadata_by_id = {}
    assets_by_id = {}
    requested_paths = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def paginate(self, path, params=None):
        self.requested_paths.append((path, params))
        term = (params or {}).get("search", "")
        return [self._listing_row(identifier) for identifier in self.search_results.get(term, [])]

    def get(self, path):
        self.requested_paths.append((path, None))
        identifier = path.split("/")[2]
        if path.endswith("/info/"):
            return {"metadata": self.metadata_by_id[identifier]}
        return self._listing_row(identifier)

    def get_dandiset(self, dandiset_id, version=None):
        if dandiset_id not in self.assets_by_id:
            raise NotFoundError(f"Dandiset {dandiset_id} not found")
        return FakeRemoteDandiset(self.assets_by_id[dandiset_id])

    def _listing_row(self, identifier):
        assets = self.assets_by_id.get(identifier, [])
        version_record = {
            "version": self.metadata_by_id[identifier].get("version", "draft"),
            "asset_count": len(assets),
            "size": sum(asset.size for asset in assets),
        }
        published = version_record if version_record["version"] != "draft" else None
        return {
            "identifier": identifier,
            "draft_version": {"version": "draft", "asset_count": len(assets), "size": version_record["size"]},
            "most_recent_published_version": published,
        }


def _metadata(**overrides):
    """Build one dandiset's raw DANDI metadata, with the fields the summary reads."""
    metadata = {
        "name": "A photometry study",
        "description": "Fiber photometry of dopamine release.",
        "keywords": ["fiber photometry"],
        "about": [],
        "studyTarget": [],
        "contributor": [{"name": "Lerner, Talia"}],
        "license": ["spdx:CC-BY-4.0"],
        "url": "https://dandiarchive.org/dandiset/000001",
        "assetsSummary": {
            "species": [{"name": "Mus musculus - House mouse"}],
            "approach": [{"name": "behavioral approach"}],
            "measurementTechnique": [{"name": "analytical technique"}],
            "numberOfSubjects": 4,
        },
    }
    metadata.update(overrides)
    return metadata


@pytest.fixture
def archive(monkeypatch):
    """Replace the DANDI client with the in-memory archive and load it with three dandisets."""
    FakeDandiAPIClient.requested_paths = []
    FakeDandiAPIClient.search_results = {
        "photometry": ["000001", "000002"],
        "dLight": ["000002", "000003"],
        "GRAB-DA": [],
        "striatum": ["000001"],
    }
    FakeDandiAPIClient.metadata_by_id = {
        "000001": _metadata(
            name="Dorsomedial striatum dopamine",
            description="GCaMP7b recordings in the DMS during a reward task.",
            keywords=["fiber photometry", "dopamine"],
            version="0.240101.0000",
        ),
        "000002": _metadata(
            name="Ventral tegmental area dLight",
            description="dLight1.3b photometry in the VTA.",
            assetsSummary={
                "species": [{"name": "Rattus norvegicus - Norway rat"}],
                "approach": [{"name": "optogenetic approach"}],
                "numberOfSubjects": 12,
            },
        ),
        "000003": _metadata(name="No regions named here", description="Photometry.", keywords=[]),
    }
    FakeDandiAPIClient.assets_by_id = {
        "000001": [
            FakeRemoteAsset("sub-01/ses-1_behavior.nwb", 240_000),
            FakeRemoteAsset("sub-01/ses-2_behavior.nwb", 60_000_000),
            FakeRemoteAsset("sub-02/ses-1_behavior.nwb", 240_000),
            FakeRemoteAsset("README.md", 100),
        ],
        "000002": [FakeRemoteAsset("sub-a/data.nwb", 1_000)],
        "000003": [],
    }
    monkeypatch.setattr(dandi_catalog, "DandiAPIClient", FakeDandiAPIClient)
    return FakeDandiAPIClient


class TestFindVocabularyTerms:
    def test_matches_full_names_and_abbreviations(self):
        text = "recordings in the dms and the nucleus accumbens with gcamp6f"
        regions = find_vocabulary_terms(text=text, vocabulary=dandi_catalog._BRAIN_REGION_PATTERNS)
        indicators = find_vocabulary_terms(text=text, vocabulary=dandi_catalog._INDICATOR_PATTERNS)
        assert regions == ("Dorsal striatum", "Ventral striatum")
        # "gcamp6f" names the GCaMP family plus a variant suffix, which the family term matches.
        assert indicators == ("GCaMP",)

    def test_a_subregion_also_matches_the_region_that_contains_it(self):
        # A study of the DMS is a study of the striatum, so filtering on either finds it.
        terms = find_vocabulary_terms(
            text="dopamine in the dorsomedial striatum", vocabulary=dandi_catalog._BRAIN_REGION_PATTERNS
        )
        assert terms == ("Dorsal striatum", "Striatum")

    def test_abbreviation_does_not_match_inside_a_longer_word(self):
        # "LHb" is the habenula, not the lateral hypothalamus; "dmso" is neither.
        text = "fibers in lhb and a dmso vehicle"
        regions = find_vocabulary_terms(text=text, vocabulary=dandi_catalog._BRAIN_REGION_PATTERNS)
        assert regions == ("Habenula",)

    def test_no_match_returns_empty(self):
        assert find_vocabulary_terms(text="a study of nothing", vocabulary=dandi_catalog._INDICATOR_PATTERNS) == ()


class TestFormatByteSize:
    @pytest.mark.parametrize(
        ("size_in_bytes", "expected"),
        [(100, "100 B"), (240_000, "234 KB"), (60_000_000, "57.2 MB"), (23_491_138_657, "21.9 GB")],
    )
    def test_renders_in_the_largest_unit_above_one(self, size_in_bytes, expected):
        assert format_byte_size(size_in_bytes) == expected


class TestSearchDandisets:
    def test_unions_the_search_terms_and_sorts_by_identifier(self, archive):
        summaries = search_dandisets(terms=("photometry", "dLight"))
        assert [summary.identifier for summary in summaries] == ["000001", "000002", "000003"]

    def test_reads_the_published_version_when_there_is_one(self, archive):
        by_id = {summary.identifier: summary for summary in search_dandisets(terms=("photometry",))}
        assert by_id["000001"].version == "0.240101.0000"
        assert by_id["000001"].is_published is True
        assert by_id["000002"].version == "draft"
        assert by_id["000002"].is_published is False

    def test_summary_carries_the_metadata_the_catalog_shows(self, archive):
        (summary,) = search_dandisets(terms=("striatum",))
        assert summary.name == "Dorsomedial striatum dopamine"
        assert summary.species == ("Mus musculus",)
        assert summary.approaches == ("behavioral approach", "analytical technique")
        assert summary.keywords == ("fiber photometry", "dopamine")
        assert summary.subject_count == 4
        assert summary.contributors == ("Lerner, Talia",)
        assert summary.license_terms == ("spdx:CC-BY-4.0",)

    def test_regions_and_indicators_come_from_the_free_text(self, archive):
        by_id = {summary.identifier: summary for summary in search_dandisets(terms=("photometry", "dLight"))}
        assert by_id["000001"].brain_regions == ("Dorsal striatum", "Striatum")
        assert by_id["000001"].indicators == ("GCaMP",)
        assert by_id["000002"].brain_regions == ("Ventral tegmental area",)
        assert by_id["000002"].indicators == ("dLight",)
        assert by_id["000003"].brain_regions == ()

    def test_file_count_and_size_come_from_the_version_record(self, archive):
        # assetsSummary carries neither, and a draft-only dandiset leaves them at zero.
        by_id = {summary.identifier: summary for summary in search_dandisets(terms=("photometry",))}
        assert by_id["000001"].file_count == 4
        assert by_id["000001"].size_in_bytes == 60_480_100
        assert by_id["000002"].file_count == 1

    def test_repeated_and_blank_schema_names_are_collapsed(self, archive):
        archive.metadata_by_id["000003"]["contributor"] = [
            {"name": "Lerner, Talia"},
            {"name": "Lerner, Talia"},
            {"name": ""},
            {"schemaKey": "Person"},
        ]
        by_id = {summary.identifier: summary for summary in search_dandisets(terms=("dLight",))}
        assert by_id["000003"].contributors == ("Lerner, Talia",)

    def test_an_unmatched_term_contributes_nothing(self, archive):
        summaries = search_dandisets(terms=("GRAB-DA",))
        assert summaries == []

    def test_max_results_caps_the_metadata_requests(self, archive):
        summaries = search_dandisets(terms=("photometry", "dLight"), max_results=1)
        assert len(summaries) == 1
        assert sum(1 for path, _ in archive.requested_paths if path.endswith("/info/")) == 1


def _summary(**overrides):
    """Build a DandisetSummary with every field set, for the filter tests."""
    fields = {
        "identifier": "000001",
        "version": "draft",
        "name": "Study",
        "description": "",
        "species": ("Mus musculus",),
        "approaches": ("behavioral approach",),
        "keywords": (),
        "brain_regions": ("Dorsal striatum",),
        "indicators": ("GCaMP",),
        "subject_count": 10,
        "file_count": 100,
        "size_in_bytes": 1000,
        "contributors": (),
        "license_terms": (),
        "url": "",
        "is_published": False,
        "searchable_text": "study of dopamine in the dorsomedial striatum",
    }
    fields.update(overrides)
    return DandisetSummary(**fields)


class TestFilterDandisets:
    @pytest.fixture
    def summaries(self):
        return [
            _summary(identifier="000001"),
            _summary(
                identifier="000002",
                species=("Rattus norvegicus",),
                brain_regions=("Ventral tegmental area",),
                indicators=("dLight",),
                approaches=("optogenetic approach",),
                subject_count=2,
                file_count=5,
                is_published=True,
                searchable_text="dlight in the vta",
            ),
        ]

    def test_no_criteria_keeps_everything(self, summaries):
        assert filter_dandisets(summaries) == summaries

    def test_query_matches_every_word_against_the_text_blob(self, summaries):
        assert [s.identifier for s in filter_dandisets(summaries, query="dopamine striatum")] == ["000001"]
        assert filter_dandisets(summaries, query="dopamine vta") == []

    def test_categorical_criteria_keep_any_requested_value(self, summaries):
        assert [s.identifier for s in filter_dandisets(summaries, species=["Rattus norvegicus"])] == ["000002"]
        assert len(filter_dandisets(summaries, brain_regions=["Dorsal striatum", "Ventral tegmental area"])) == 2

    def test_criteria_are_conjunctive(self, summaries):
        assert filter_dandisets(summaries, species=["Mus musculus"], indicators=["dLight"]) == []

    def test_scale_bounds_and_published_only(self, summaries):
        assert [s.identifier for s in filter_dandisets(summaries, minimum_subjects=5)] == ["000001"]
        assert [s.identifier for s in filter_dandisets(summaries, minimum_files=50)] == ["000001"]
        assert [s.identifier for s in filter_dandisets(summaries, published_only=True)] == ["000002"]

    def test_collect_filter_options_reports_the_present_values_sorted(self, summaries):
        options = collect_filter_options(summaries)
        assert options["species"] == ["Mus musculus", "Rattus norvegicus"]
        assert options["brain_regions"] == ["Dorsal striatum", "Ventral tegmental area"]
        assert options["indicators"] == ["GCaMP", "dLight"]
        assert options["approaches"] == ["behavioral approach", "optogenetic approach"]


class TestListNwbAssets:
    def test_lists_paths_and_sizes_of_the_nwb_assets_only(self, archive):
        assets = list_nwb_assets(dandiset_id="000001")
        assert [(asset.path, asset.size_in_bytes) for asset in assets] == [
            ("sub-01/ses-1_behavior.nwb", 240_000),
            ("sub-01/ses-2_behavior.nwb", 60_000_000),
            ("sub-02/ses-1_behavior.nwb", 240_000),
        ]

    def test_max_assets_truncates(self, archive):
        assert len(list_nwb_assets(dandiset_id="000001", max_assets=2)) == 2

    def test_unknown_dandiset_raises_not_found(self, archive):
        with pytest.raises(NotFoundError):
            list_nwb_assets(dandiset_id="999999")


class TestFilterAssets:
    @pytest.fixture
    def assets(self):
        return [
            AssetSummary(asset_id="a", path="sub-01/ses-1_behavior.nwb", size_in_bytes=240_000),
            AssetSummary(asset_id="b", path="sub-01/ses-2_photometry.nwb", size_in_bytes=60_000_000),
            AssetSummary(asset_id="c", path="sub-02/ses-1_behavior.nwb", size_in_bytes=5_000_000),
        ]

    def test_no_criteria_keeps_everything(self, assets):
        assert filter_assets(assets) == assets

    def test_name_substring_is_case_insensitive_and_spans_the_path(self, assets):
        assert [a.asset_id for a in filter_assets(assets, name_contains="SUB-01")] == ["a", "b"]
        assert [a.asset_id for a in filter_assets(assets, name_contains="photometry")] == ["b"]

    def test_minimum_size_keeps_the_recordings(self, assets):
        assert [a.asset_id for a in filter_assets(assets, minimum_size_in_bytes=1_000_000)] == ["b", "c"]

    def test_criteria_combine(self, assets):
        assert filter_assets(assets, name_contains="sub-02", minimum_size_in_bytes=10_000_000) == []


# ---------------------------------------------------------------------------------------------
# Probe layer, against the real mock NWB files
# ---------------------------------------------------------------------------------------------


@pytest.fixture(params=sorted(MOCK_NWB_FILES))
def mock_nwb_file(request):
    """Open each mock NWB file in turn, so the probe is checked on every supported layout."""
    with h5py.File(MOCK_NWB_FILES[request.param], "r") as file:
        yield file


@pytest.fixture
def sparse_nwb_file(tmp_path):
    """An NWB-shaped file whose series carries timestamps and whose fiber table is bare.

    Covers the layouts the mock files do not: a single-channel series with irregular
    timestamps instead of a starting_time/rate pair, a fiber photometry table with only a
    ``location`` column, and no ``/processing`` group at all.
    """
    path = tmp_path / "sparse.nwb"
    with h5py.File(path, "w") as file:
        series = file.create_group("acquisition/series_with_timestamps")
        series.attrs["neurodata_type"] = "FiberPhotometryResponseSeries"
        series.create_dataset("data", data=np.arange(10, dtype=float))
        series.create_dataset("timestamps", data=np.arange(10) * 0.5)
        series.create_dataset("fiber_photometry_table_region", data=np.array([0]))
        table = file.create_group("general/fiber_photometry/fiber_photometry_table")
        table.attrs["neurodata_type"] = "FiberPhotometryTable"
        table.create_dataset("location", data=np.array([b"NAc"]))
    with h5py.File(path, "r") as file:
        yield file


class TestProbePhotometry:
    def test_finds_the_response_series_and_its_timing(self, mock_nwb_file):
        probe = probe_photometry(file=mock_nwb_file)
        assert probe.has_photometry is True
        (series,) = probe.series
        assert series.name == "fiber_photometry_response_series"
        assert series.sample_count == 3000
        assert series.channel_count == 2
        assert series.sampling_rate_in_hz == 30.0
        assert series.duration_in_seconds == 100.0

    def test_channels_carry_the_store_names_guppy_will_show(self, mock_nwb_file):
        probe = probe_photometry(file=mock_nwb_file)
        assert [channel.store_name for channel in probe.channels] == [
            "fiber_photometry_response_series_0",
            "fiber_photometry_response_series_1",
        ]
        assert [channel.column_index for channel in probe.channels] == [0, 1]

    def test_channels_carry_the_fiber_photometry_table_details(self, mock_nwb_file):
        probe = probe_photometry(file=mock_nwb_file)
        assert probe.locations == ("VTA",)
        assert probe.indicators == ("GCamp6f",)
        assert [channel.excitation_wavelength_in_nm for channel in probe.channels] == [405.0, 470.0]

    def test_isosbestic_excitation_suggests_a_control_label(self, mock_nwb_file):
        probe = probe_photometry(file=mock_nwb_file)
        assert [channel.suggested_label for channel in probe.channels] == ["control_VTA", "signal_VTA"]

    def test_session_fields_are_read(self, mock_nwb_file):
        probe = probe_photometry(file=mock_nwb_file)
        assert "Mock session for NWB extractor testing" in probe.session_description
        assert probe.identifier == Path(mock_nwb_file.filename).stem
        # The mock files are regenerated, so only the shape of the timestamp is fixed.
        assert datetime.fromisoformat(probe.session_start_time).year >= 2024
        # These mock files carry no Subject group, which the probe reports as no fields.
        assert probe.subject == {}

    def test_event_objects_are_found_wherever_they_live(self, mock_nwb_file):
        probe = probe_photometry(file=mock_nwb_file)
        # ndx-events objects sit in /acquisition; core EventsTables sit in /events.
        assert set(probe.event_names) in (
            {"AnnotatedEventsTable", "events", "labeled_events"},
            {"annotated_events", "simple_events", "strobe_events"},
        )

    def test_emission_wavelength_is_read_when_the_table_has_it(self):
        # The column exists only in ndx-fiber-photometry v0.2.
        with h5py.File(MOCK_NWB_FILES["mock_nwbfile_ndx_fiber_photometry_v0_2_core_events"], "r") as file:
            probe = probe_photometry(file=file)
        assert [channel.emission_wavelength_in_nm for channel in probe.channels] == [525.0, 525.0]
        with h5py.File(MOCK_NWB_FILES["mock_nwbfile_ndx_fiber_photometry_v0_1_ndx_events_v0_2"], "r") as file:
            probe = probe_photometry(file=file)
        assert [channel.emission_wavelength_in_nm for channel in probe.channels] == [None, None]

    def test_timestamps_are_used_when_the_series_has_no_rate(self, sparse_nwb_file):
        probe = probe_photometry(file=sparse_nwb_file)
        (series,) = probe.series
        # Timestamps run 0, 0.5, ... 4.5, so the span is 4.5 s over 9 intervals: 2 Hz.
        assert series.sampling_rate_in_hz == 2.0
        assert series.duration_in_seconds == 4.5

    def test_a_single_channel_series_keeps_its_own_name_as_the_store(self, sparse_nwb_file):
        probe = probe_photometry(file=sparse_nwb_file)
        (channel,) = probe.channels
        assert channel.store_name == "series_with_timestamps"
        assert channel.column_index is None

    def test_table_columns_the_file_omits_are_reported_as_unknown(self, sparse_nwb_file):
        probe = probe_photometry(file=sparse_nwb_file)
        (channel,) = probe.channels
        assert channel.location == "NAc"
        assert channel.indicator is None
        assert channel.excitation_wavelength_in_nm is None
        assert channel.emission_wavelength_in_nm is None
        # Without a wavelength there is no basis for calling the channel signal or control.
        assert channel.suggested_label is None

    def test_a_file_without_photometry_reports_none(self, tmp_path):
        path = tmp_path / "empty.nwb"
        with h5py.File(path, "w") as file:
            file.create_group("acquisition")
            file.create_dataset("session_description", data="no photometry here")
        with h5py.File(path, "r") as file:
            probe = probe_photometry(file=file)
        assert probe.has_photometry is False
        assert probe.channels == ()
        assert probe.series == ()
        assert probe.session_description == "no photometry here"


class TestReadExampleTraces:
    def test_reads_the_requested_duration_of_every_channel(self, mock_nwb_file):
        probe = probe_photometry(file=mock_nwb_file)
        traces = read_example_traces(file=mock_nwb_file, probe=probe, duration_in_seconds=10.0, max_points=1000)
        # 10 s at 30 Hz is 300 samples, which is under max_points, so nothing is strided away.
        assert list(traces.traces) == [
            "fiber_photometry_response_series_0",
            "fiber_photometry_response_series_1",
        ]
        assert traces.timestamps.shape == (300,)
        np.testing.assert_allclose(traces.timestamps[:3], [0.0, 1 / 30, 2 / 30])
        for values in traces.traces.values():
            assert values.shape == (300,)

    def test_decimates_down_to_max_points(self, mock_nwb_file):
        probe = probe_photometry(file=mock_nwb_file)
        traces = read_example_traces(file=mock_nwb_file, probe=probe, duration_in_seconds=100.0, max_points=100)
        # 100 s at 30 Hz is the full 3000 samples, strided by ceil(3000/100) = 30.
        assert traces.timestamps.shape == (100,)
        np.testing.assert_allclose(traces.timestamps[:3], [0.0, 1.0, 2.0])

    def test_traces_hold_the_series_columns(self, mock_nwb_file):
        probe = probe_photometry(file=mock_nwb_file)
        traces = read_example_traces(file=mock_nwb_file, probe=probe, duration_in_seconds=1.0, max_points=1000)
        expected = mock_nwb_file["acquisition/fiber_photometry_response_series/data"][:30]
        np.testing.assert_allclose(traces.traces["fiber_photometry_response_series_0"], expected[:, 0])
        np.testing.assert_allclose(traces.traces["fiber_photometry_response_series_1"], expected[:, 1])

    def test_reads_a_timestamped_series_from_its_own_timestamps(self, sparse_nwb_file):
        probe = probe_photometry(file=sparse_nwb_file)
        traces = read_example_traces(file=sparse_nwb_file, probe=probe, max_points=4)
        # All 10 samples fall inside the default 60 s window, strided by ceil(10/4) = 3.
        np.testing.assert_allclose(traces.timestamps, [0.0, 1.5, 3.0, 4.5])
        np.testing.assert_allclose(traces.traces["series_with_timestamps"], [0.0, 3.0, 6.0, 9.0])

    def test_never_returns_more_than_max_points(self, mock_nwb_file):
        probe = probe_photometry(file=mock_nwb_file)
        traces = read_example_traces(file=mock_nwb_file, probe=probe, duration_in_seconds=100.0, max_points=7)
        assert traces.timestamps.shape == (7,)

    def test_named_series_is_honored(self, mock_nwb_file):
        probe = probe_photometry(file=mock_nwb_file)
        traces = read_example_traces(
            file=mock_nwb_file,
            probe=probe,
            series_name="fiber_photometry_response_series",
            duration_in_seconds=1.0,
        )
        assert traces.series_name == "fiber_photometry_response_series"

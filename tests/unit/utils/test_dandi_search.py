"""Tests for searching the DANDI Archive and listing a dandiset's NWB assets.

Everything here talks to the DANDI REST API in production, so ``DandiAPIClient`` is replaced with
an in-memory archive built from recorded response shapes. No network access anywhere; the same
search and listing run against the real archive in ``test_dandi_search_live.py``.
"""

from fnmatch import fnmatch

import pytest
from dandi.exceptions import NotFoundError

from guppy.utils import dandi_search
from guppy.utils.dandi_search import (
    DandisetSummary,
    _direct_content_url,
    collect_filter_options,
    filter_dandisets,
    find_vocabulary_terms,
    format_byte_size,
    list_nwb_assets,
    search_dandisets,
)

from .dandi_search_test_mixin import DandiSearchTestMixin


class FakeRemoteAsset:
    def __init__(self, path, size):
        self.identifier = f"asset-{path}"
        self.path = path
        self.size = size

    def as_listing_row(self):
        """Render as the archive's asset listing renders an asset, metadata included."""
        return {
            "asset_id": self.identifier,
            "path": self.path,
            "size": self.size,
            "metadata": {
                "contentUrl": [
                    f"https://api.dandiarchive.org/api/assets/{self.identifier}/download/",
                    f"https://dandiarchive.s3.amazonaws.com/blobs/{self.identifier}",
                ]
            },
        }


class FakeRemoteDandiset:
    def __init__(self, assets):
        self._assets = assets


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
        if path.endswith("/assets/"):
            identifier = path.split("/")[2]
            pattern = (params or {}).get("glob", "*")
            return [
                asset.as_listing_row()
                for asset in self.assets_by_id.get(identifier, [])
                if fnmatch(asset.path, pattern)
            ]
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
            "draft_version": {
                "version": "draft",
                "asset_count": len(assets),
                "size": version_record["size"],
            },
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
    monkeypatch.setattr(dandi_search, "DandiAPIClient", FakeDandiAPIClient)
    return FakeDandiAPIClient


class TestFindVocabularyTerms:
    def test_matches_full_names_and_abbreviations(self):
        text = "recordings in the dms and the nucleus accumbens with gcamp6f"
        regions = find_vocabulary_terms(text=text, vocabulary=dandi_search._BRAIN_REGION_PATTERNS)
        indicators = find_vocabulary_terms(text=text, vocabulary=dandi_search._INDICATOR_PATTERNS)
        assert regions == ("Dorsal striatum", "Ventral striatum")
        # "gcamp6f" names the GCaMP family plus a variant suffix, which the family term matches.
        assert indicators == ("GCaMP",)

    def test_a_subregion_also_matches_the_region_that_contains_it(self):
        # A study of the DMS is a study of the striatum, so filtering on either finds it.
        terms = find_vocabulary_terms(
            text="dopamine in the dorsomedial striatum",
            vocabulary=dandi_search._BRAIN_REGION_PATTERNS,
        )
        assert terms == ("Dorsal striatum", "Striatum")

    def test_abbreviation_does_not_match_inside_a_longer_word(self):
        # "LHb" is the habenula, not the lateral hypothalamus; "dmso" is neither.
        text = "fibers in lhb and a dmso vehicle"
        regions = find_vocabulary_terms(text=text, vocabulary=dandi_search._BRAIN_REGION_PATTERNS)
        assert regions == ("Habenula",)

    def test_no_match_returns_empty(self):
        assert find_vocabulary_terms(text="a study of nothing", vocabulary=dandi_search._INDICATOR_PATTERNS) == ()


class TestFormatByteSize:
    @pytest.mark.parametrize(
        ("size_in_bytes", "expected"),
        [
            (100, "100 B"),
            (240_000, "234 KB"),
            (60_000_000, "57.2 MB"),
            (23_491_138_657, "21.9 GB"),
        ],
    )
    def test_renders_in_the_largest_unit_above_one(self, size_in_bytes, expected):
        assert format_byte_size(size_in_bytes) == expected


class TestSearchDandisets:
    def test_unions_the_search_terms_and_sorts_by_identifier(self, archive):
        summaries = search_dandisets(terms=("photometry", "dLight"))
        assert [summary.identifier for summary in summaries] == [
            "000001",
            "000002",
            "000003",
        ]

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
        assert (
            len(
                filter_dandisets(
                    summaries,
                    brain_regions=["Dorsal striatum", "Ventral tegmental area"],
                )
            )
            == 2
        )

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

    def test_listing_carries_the_url_the_bytes_are_read_from(self, archive):
        assets = list_nwb_assets(dandiset_id="000001")
        assert [asset.content_url for asset in assets] == [
            "https://dandiarchive.s3.amazonaws.com/blobs/asset-sub-01/ses-1_behavior.nwb",
            "https://dandiarchive.s3.amazonaws.com/blobs/asset-sub-01/ses-2_behavior.nwb",
            "https://dandiarchive.s3.amazonaws.com/blobs/asset-sub-02/ses-1_behavior.nwb",
        ]

    def test_the_listing_asks_the_archive_for_asset_metadata(self, archive):
        list_nwb_assets(dandiset_id="000001")
        assets_request = next(params for path, params in archive.requested_paths if path.endswith("/assets/"))
        assert assets_request["metadata"] == "true"
        assert assets_request["glob"] == "*.nwb"

    def test_max_assets_truncates(self, archive):
        assert len(list_nwb_assets(dandiset_id="000001", max_assets=2)) == 2

    def test_unknown_dandiset_raises_not_found(self, archive):
        with pytest.raises(NotFoundError):
            list_nwb_assets(dandiset_id="999999")


class TestDirectContentUrl:
    def test_prefers_the_url_that_serves_bytes_over_the_one_that_redirects(self):
        metadata = {
            "contentUrl": [
                "https://api.dandiarchive.org/api/assets/abc/download/",
                "https://dandiarchive.s3.amazonaws.com/blobs/abc",
            ]
        }
        assert _direct_content_url(metadata) == "https://dandiarchive.s3.amazonaws.com/blobs/abc"

    def test_falls_back_to_the_only_url_there_is(self):
        metadata = {"contentUrl": ["https://api.dandiarchive.org/api/assets/abc/download/"]}
        assert _direct_content_url(metadata) == "https://api.dandiarchive.org/api/assets/abc/download/"

    def test_an_asset_without_urls_has_none(self):
        assert _direct_content_url({}) == ""


class TestDandiSearchAgainstTheInMemoryArchive(DandiSearchTestMixin):
    """The search contract, bound to the in-memory stand-in for the REST API."""

    search_terms = ("photometry",)
    expected_identifier = "000001"
    listing_dandiset_id = "000001"
    expected_asset_path = "sub-01/ses-1_behavior.nwb"

    @pytest.fixture
    def archive_ready(self, archive):
        return archive

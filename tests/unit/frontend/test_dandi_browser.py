"""Unit tests for the DANDI catalog browser and the shared photometry preview pane.

Every network call the browser makes — the catalog search, the photometry verification, the
archive-wide crawl's dandiset listing, the metadata fetch for what the crawl confirms, the asset
listing and the streaming preview — is a constructor injection point, so every test here runs
against in-memory stand-ins with no archive access.
"""

import holoviews as hv
import numpy as np
import pytest

from guppy.frontend.dandi_browser import (
    CATALOG_COLUMNS,
    CHANNEL_COLUMNS,
    DandiBrowser,
    PhotometryPreviewPane,
    build_trace_overlay,
    catalog_dataframe,
    channel_dataframe,
    describe_dandiset,
    describe_probe,
)
from guppy.utils.dandi_catalog import (
    AssetPreview,
    AssetSummary,
    ChannelInfo,
    DandisetReference,
    DandisetSummary,
    ExampleTraces,
    PhotometryProbe,
    PhotometryVerdictCache,
    SeriesInfo,
)


def make_summary(**overrides):
    """Build a DandisetSummary with every field populated."""
    fields = {
        "identifier": "000001",
        "version": "0.240101.0000",
        "name": "Dorsomedial striatum dopamine",
        "description": "GCaMP7b recordings in the DMS.",
        "species": ("Mus musculus",),
        "approaches": ("behavioral approach",),
        "keywords": ("fiber photometry",),
        "brain_regions": ("Dorsal striatum",),
        "indicators": ("GCaMP",),
        "subject_count": 10,
        "file_count": 100,
        "size_in_bytes": 60_000_000,
        "contributors": ("Lerner, Talia",),
        "license_terms": ("spdx:CC-BY-4.0",),
        "url": "https://dandiarchive.org/dandiset/000001",
        "is_published": True,
        "searchable_text": "dorsomedial striatum dopamine gcamp7b recordings in the dms",
    }
    fields.update(overrides)
    return DandisetSummary(**fields)


SUMMARIES = [
    make_summary(),
    make_summary(
        identifier="000002",
        name="Ventral tegmental area dLight",
        version="draft",
        species=("Rattus norvegicus",),
        brain_regions=("Ventral tegmental area",),
        indicators=("dLight",),
        approaches=("optogenetic approach",),
        subject_count=2,
        file_count=5,
        is_published=False,
        searchable_text="ventral tegmental area dlight",
    ),
]


def make_probe(*, channel_count=2, has_photometry=True):
    """Build a probe of one two-channel series recorded in the VTA, or of an empty file."""
    if not has_photometry:
        return PhotometryProbe(session_description="behavior only", event_names=("nose_poke",))
    channels = tuple(
        ChannelInfo(
            store_name=f"fiber_photometry_response_series_{index}",
            series_name="fiber_photometry_response_series",
            column_index=index,
            location="VTA",
            indicator="GCaMP7b",
            excitation_wavelength_in_nm=405.0 if index == 0 else 465.0,
            emission_wavelength_in_nm=525.0,
            suggested_label="control_VTA" if index == 0 else "signal_VTA",
        )
        for index in range(channel_count)
    )
    return PhotometryProbe(
        series=(
            SeriesInfo(
                name="fiber_photometry_response_series",
                sample_count=3000,
                channel_count=channel_count,
                sampling_rate_in_hz=30.0,
                duration_in_seconds=100.0,
            ),
        ),
        channels=channels,
        event_names=("nose_poke",),
        session_description="A reward task",
        session_start_time="2024-01-01T00:00:00+00:00",
        identifier="session-1",
        subject={"subject_id": "mouse-1", "species": "Mus musculus", "sex": "F"},
    )


def make_preview(*, probe=None, series_name="fiber_photometry_response_series", with_traces=True):
    """Build an AssetPreview, optionally carrying example traces for ``series_name``."""
    probe = probe if probe is not None else make_probe()
    traces = None
    if with_traces and probe.has_photometry:
        timestamps = np.linspace(0.0, 10.0, 50)
        traces = ExampleTraces(
            series_name=series_name,
            timestamps=timestamps,
            traces={channel.store_name: np.sin(timestamps) for channel in probe.channels},
        )
    return AssetPreview(dandiset_id="000001", asset_path="sub-01/ses-1.nwb", probe=probe, traces=traces)


class RecordingPreview:
    """Stand-in for ``preview_asset`` that records its calls and returns a canned preview."""

    def __init__(self, preview=None):
        self.preview = preview if preview is not None else make_preview()
        self.calls = []

    def __call__(self, *, dandiset_id, asset_path, series_name=None):
        self.calls.append(
            {
                "dandiset_id": dandiset_id,
                "asset_path": asset_path,
                "series_name": series_name,
            }
        )
        return self.preview


class TestCatalogDataframe:
    def test_columns_and_rendered_values(self):
        frame = catalog_dataframe(SUMMARIES)
        assert list(frame.columns) == list(CATALOG_COLUMNS)
        assert list(frame["Dandiset"]) == ["000001", "000002"]
        assert list(frame["Size"]) == ["57.2 MB", "57.2 MB"]
        assert list(frame["Brain regions"]) == [
            "Dorsal striatum",
            "Ventral tegmental area",
        ]

    def test_empty_catalog_still_has_the_columns(self):
        assert list(catalog_dataframe([]).columns) == list(CATALOG_COLUMNS)


class TestChannelDataframe:
    def test_one_row_per_channel(self):
        frame = channel_dataframe(make_probe())
        assert list(frame.columns) == list(CHANNEL_COLUMNS)
        assert list(frame["Store name"]) == [
            "fiber_photometry_response_series_0",
            "fiber_photometry_response_series_1",
        ]
        assert list(frame["Suggested label"]) == ["control_VTA", "signal_VTA"]

    def test_missing_fields_render_as_blanks(self):
        probe = PhotometryProbe(
            channels=(
                ChannelInfo(
                    store_name="series",
                    series_name="series",
                    column_index=None,
                    location=None,
                    indicator=None,
                    excitation_wavelength_in_nm=None,
                    emission_wavelength_in_nm=None,
                    suggested_label=None,
                ),
            )
        )
        frame = channel_dataframe(probe)
        assert list(frame.loc[0]) == ["series", "", "", "", "", ""]


class TestDescribeDandiset:
    def test_reports_identity_scale_and_detected_terms(self):
        text = describe_dandiset(make_summary())
        assert "000001 — Dorsomedial striatum dopamine" in text
        assert "https://dandiarchive.org/dandiset/000001" in text
        assert "**Species:** Mus musculus" in text
        assert "**Subjects:** 10" in text
        assert "**Brain regions:** Dorsal striatum" in text
        assert "**Indicators:** GCaMP" in text
        assert "GCaMP7b recordings in the DMS." in text

    def test_spdx_prefix_is_stripped_from_the_license(self):
        assert "CC-BY-4.0" in describe_dandiset(make_summary())
        assert "spdx:" not in describe_dandiset(make_summary())

    def test_absent_fields_render_as_a_dash(self):
        text = describe_dandiset(make_summary(species=(), indicators=(), keywords=(), license_terms=()))
        assert "**Species:** —" in text
        assert "**Indicators:** —" in text
        assert "license not stated" in text

    def test_long_contributor_list_is_abbreviated(self):
        text = describe_dandiset(make_summary(contributors=tuple(f"Author {index}" for index in range(6))))
        assert "Author 3, et al." in text
        assert "Author 4" not in text


class TestDescribeProbe:
    def test_reports_every_series_with_its_timing(self):
        text = describe_probe(preview=make_preview())
        assert "`sub-01/ses-1.nwb`" in text
        assert "2 channel(s)" in text
        assert "3,000 samples" in text
        assert "30.00 Hz" in text
        assert "1.7 min" in text

    def test_reports_subject_and_session(self):
        text = describe_probe(preview=make_preview())
        assert "mouse-1 · Mus musculus · F" in text
        assert "**Session:** A reward task" in text
        assert "**Event objects:** nose_poke" in text

    def test_file_without_photometry_is_called_out(self):
        text = describe_probe(preview=make_preview(probe=make_probe(has_photometry=False)))
        assert "no `FiberPhotometryResponseSeries`" in text
        assert "the recordings are the large ones" in text
        assert "**Event objects:** nose_poke" in text

    def test_file_with_neither_photometry_nor_events_says_only_that(self):
        preview = make_preview(probe=PhotometryProbe(session_description="empty"))
        text = describe_probe(preview=preview)
        assert "no `FiberPhotometryResponseSeries`" in text
        assert "Event objects" not in text

    def test_irregular_timing_is_named_rather_than_faked(self):
        probe = PhotometryProbe(
            series=(
                SeriesInfo(
                    name="series",
                    sample_count=10,
                    channel_count=1,
                    sampling_rate_in_hz=None,
                    duration_in_seconds=None,
                ),
            )
        )
        text = describe_probe(preview=make_preview(probe=probe, with_traces=False))
        assert "irregular timestamps" in text
        assert "unknown length" in text


class TestBuildTraceOverlay:
    def test_one_curve_per_channel_over_a_shared_time_axis(self):
        overlay = build_trace_overlay(preview=make_preview())
        assert isinstance(overlay, hv.NdOverlay)
        assert list(overlay.keys()) == [
            "fiber_photometry_response_series_0",
            "fiber_photometry_response_series_1",
        ]
        curve = overlay["fiber_photometry_response_series_0"]
        np.testing.assert_allclose(curve.dimension_values(0), np.linspace(0.0, 10.0, 50))

    def test_title_names_the_series_and_the_window(self):
        overlay = build_trace_overlay(preview=make_preview())
        assert overlay.opts.get("plot").kwargs["title"] == "First 10 s of fiber_photometry_response_series"


@pytest.fixture
def preview_pane(panel_extension):
    return PhotometryPreviewPane(preview_function=RecordingPreview())


class TestPhotometryPreviewPane:
    def test_starts_hidden_and_empty(self, preview_pane):
        assert preview_pane.panel.visible is False
        assert preview_pane.preview is None
        assert preview_pane.channel_table.value.empty

    def test_show_fills_every_part(self, preview_pane):
        preview_pane.show(preview=make_preview())
        assert preview_pane.panel.visible is True
        assert "2 channel(s)" in preview_pane.summary.object
        assert len(preview_pane.channel_table.value) == 2
        assert isinstance(preview_pane.trace_pane.object, hv.NdOverlay)

    def test_single_series_file_hides_the_series_picker(self, preview_pane):
        preview_pane.show(preview=make_preview())
        assert preview_pane.series_select.visible is False
        assert preview_pane.series_select.value == "fiber_photometry_response_series"

    def test_multi_series_file_shows_the_series_picker(self, preview_pane):
        probe = make_probe()
        extra = SeriesInfo(
            name="second_series",
            sample_count=10,
            channel_count=1,
            sampling_rate_in_hz=30.0,
            duration_in_seconds=1.0,
        )
        probe = PhotometryProbe(
            series=(*probe.series, extra),
            channels=probe.channels,
            event_names=probe.event_names,
        )
        preview_pane.show(preview=make_preview(probe=probe))
        assert preview_pane.series_select.visible is True
        assert preview_pane.series_select.options == [
            "fiber_photometry_response_series",
            "second_series",
        ]

    def test_showing_a_preview_does_not_restream_it(self, preview_pane):
        preview_pane.show(preview=make_preview())
        assert preview_pane.preview_function.calls == []

    def test_picking_another_series_restreams_that_series(self, preview_pane):
        second = SeriesInfo(
            name="second_series",
            sample_count=10,
            channel_count=1,
            sampling_rate_in_hz=30.0,
            duration_in_seconds=1.0,
        )
        probe = PhotometryProbe(series=(*make_probe().series, second), channels=make_probe().channels)
        preview_pane.preview_function.preview = make_preview(probe=probe, series_name="second_series")
        preview_pane.show(preview=make_preview(probe=probe))

        preview_pane.series_select.value = "second_series"

        assert preview_pane.preview_function.calls == [
            {
                "dandiset_id": "000001",
                "asset_path": "sub-01/ses-1.nwb",
                "series_name": "second_series",
            }
        ]
        assert preview_pane.preview.traces.series_name == "second_series"

    def test_file_without_photometry_hides_the_channel_table_and_plot(self, preview_pane):
        preview_pane.show(preview=make_preview(probe=make_probe(has_photometry=False)))
        assert preview_pane.panel.visible is True
        assert preview_pane.channel_table.visible is False
        assert preview_pane.trace_pane.object is None

    def test_clear_empties_and_hides(self, preview_pane):
        preview_pane.show(preview=make_preview())
        preview_pane.clear()
        assert preview_pane.panel.visible is False
        assert preview_pane.preview is None
        assert preview_pane.summary.object == ""
        assert preview_pane.channel_table.value.empty
        assert preview_pane.trace_pane.object is None


class RecordingSearch:
    """Stand-in for ``search_dandisets`` that records the terms it was asked for."""

    def __init__(self, summaries=SUMMARIES):
        self.summaries = summaries
        self.calls = []

    def __call__(self, *, terms):
        self.calls.append(tuple(terms))
        return list(self.summaries)


ASSETS = [
    AssetSummary(
        asset_id="a",
        path="sub-01/ses-1.nwb",
        size_in_bytes=240_000,
        content_url="url-a",
    ),
    AssetSummary(
        asset_id="b",
        path="sub-01/ses-2.nwb",
        size_in_bytes=60_000_000,
        content_url="url-b",
    ),
]

# Dandisets the fake verification reports as holding photometry. Everything the browser is
# handed is confirmed unless a test narrows this, so the tests that are about filtering and
# selection are not also about verification.
CONFIRMED_IDENTIFIERS = {"000001", "000002", "000003"}


class RecordingVerification:
    """Stand-in for ``verify_dandisets`` over a fixed set of verdicts."""

    def __init__(self, confirmed=None):
        self.confirmed = set(CONFIRMED_IDENTIFIERS if confirmed is None else confirmed)
        self.calls = []

    def __call__(self, references, *, cache=None, on_verdict=None, should_stop=None, **kwargs):
        self.calls.append([reference.identifier for reference in references])
        verdicts = {}
        for reference in references:
            if should_stop is not None and should_stop():
                break
            verdicts[reference.identifier] = reference.identifier in self.confirmed
            if on_verdict is not None:
                on_verdict(reference, verdicts[reference.identifier])
        return verdicts


@pytest.fixture
def browser(panel_extension, tmp_path):
    chosen = []
    browser = DandiBrowser(
        on_dandiset_selected=chosen.append,
        search_function=RecordingSearch(),
        list_assets_function=lambda **kwargs: list(ASSETS),
        preview_function=RecordingPreview(),
        verify_function=RecordingVerification(),
        # The archive holds the two the search found plus one whose text never says
        # "photometry", which is the whole reason the crawl exists.
        references_function=lambda: [
            DandisetReference(identifier="000001", version="draft", asset_count=100),
            DandisetReference(identifier="000002", version="draft", asset_count=5),
            DandisetReference(identifier="000003", version="draft", asset_count=2),
        ],
        summarize_function=lambda identifiers: [
            make_summary(identifier=identifier, name=f"Crawled {identifier}") for identifier in identifiers
        ],
        verdict_cache=PhotometryVerdictCache(path=tmp_path / "verdicts.json"),
    )
    browser.chosen = chosen
    return browser


def load_catalog(browser):
    """Search, then run whatever verification it started to completion and draw the result.

    Verification runs on a worker thread that a periodic callback would normally poll; here
    the thread is joined and the poll is called once, which is the same sequence without the
    event loop. Searching the whole archive verifies nothing, so there may be no thread.
    """
    browser.refresh_catalog()
    finish_verification(browser)


def finish_verification(browser):
    """Wait for the browser's running verification, if any, and fold its result in."""
    if "thread" not in browser._verification:
        return
    browser._verification["thread"].join()
    browser._poll_verification()


class TestDandiBrowser:
    def test_starts_empty_and_prompts_for_a_search(self, browser):
        assert browser.summaries == []
        assert browser.results_table.value.empty
        assert "Search DANDI" in browser.status.object
        assert browser.action_row.visible is False

    def test_search_loads_the_photometry_terms_by_default(self, browser):
        load_catalog(browser)
        assert browser.search_function.calls == [("photometry",)]
        assert len(browser.results_table.value) == 2
        assert browser.status.object.startswith("Read **2** of 2 candidate(s): **2** hold fiber photometry.")

    def test_unchecking_photometry_only_sends_the_query_to_the_archive(self, browser):
        browser.photometry_only.value = False
        browser.query_input.value = "zebrafish"
        load_catalog(browser)
        assert browser.search_function.calls == [("zebrafish",)]

    def test_whole_archive_search_needs_a_term(self, browser):
        browser.photometry_only.value = False
        load_catalog(browser)
        assert browser.search_function.calls == []
        assert "needs a search term" in browser.status.object

    def test_filter_options_are_the_values_the_catalog_holds(self, browser):
        load_catalog(browser)
        assert browser.species_filter.options == ["Mus musculus", "Rattus norvegicus"]
        assert browser.brain_region_filter.options == [
            "Dorsal striatum",
            "Ventral tegmental area",
        ]
        assert browser.indicator_filter.options == ["GCaMP", "dLight"]

    def test_filtering_narrows_the_table_without_researching(self, browser):
        load_catalog(browser)
        browser.indicator_filter.value = ["dLight"]
        assert list(browser.results_table.value["Dandiset"]) == ["000002"]
        assert "Showing **1** of 2" in browser.status.object
        assert len(browser.search_function.calls) == 1

    def test_query_filters_the_photometry_catalog_locally(self, browser):
        load_catalog(browser)
        browser.query_input.value = "dms"
        assert list(browser.results_table.value["Dandiset"]) == ["000001"]
        assert len(browser.search_function.calls) == 1

    def test_toggling_photometry_only_refilters_at_once(self, browser):
        load_catalog(browser)
        browser.query_input.value = "dms"
        assert list(browser.results_table.value["Dandiset"]) == ["000001"]
        # The query now belongs to the archive, so it stops narrowing the loaded catalog.
        browser.photometry_only.value = False
        assert list(browser.results_table.value["Dandiset"]) == ["000001", "000002"]

    def test_query_is_not_applied_twice_when_the_archive_ran_it(self, browser):
        # The archive matched on fields the summary does not carry, so filtering locally on
        # the same query would drop rows the archive found.
        browser.photometry_only.value = False
        browser.query_input.value = "zebrafish"
        load_catalog(browser)
        assert len(browser.results_table.value) == 2

    def test_scale_filters_apply(self, browser):
        load_catalog(browser)
        browser.minimum_subjects.value = 5
        assert list(browser.results_table.value["Dandiset"]) == ["000001"]
        browser.minimum_subjects.value = 0
        browser.published_only.value = True
        assert list(browser.results_table.value["Dandiset"]) == ["000001"]

    def test_a_new_search_drops_filter_values_the_catalog_no_longer_offers(self, browser):
        load_catalog(browser)
        browser.indicator_filter.value = ["dLight"]
        browser.search_function.summaries = [make_summary()]
        load_catalog(browser)
        assert browser.indicator_filter.options == ["GCaMP"]
        assert browser.indicator_filter.value == []
        assert list(browser.results_table.value["Dandiset"]) == ["000001"]

    def test_selecting_a_row_shows_its_metadata(self, browser):
        load_catalog(browser)
        browser.results_table.selection = [1]
        assert browser.selected_summary.identifier == "000002"
        assert "000002 — Ventral tegmental area dLight" in browser.dandiset_details.object
        assert browser.action_row.visible is True

    def test_selection_indexes_the_filtered_rows(self, browser):
        load_catalog(browser)
        browser.indicator_filter.value = ["dLight"]
        browser.results_table.selection = [0]
        assert browser.selected_summary.identifier == "000002"

    def test_filtering_clears_a_stale_selection(self, browser):
        load_catalog(browser)
        browser.results_table.selection = [1]
        browser.indicator_filter.value = ["GCaMP"]
        assert browser.results_table.selection == []
        assert browser.selected_summary is None
        assert browser.dandiset_details.object == ""
        assert browser.action_row.visible is False

    def test_analyze_hands_the_identifier_to_the_callback(self, browser):
        load_catalog(browser)
        browser.results_table.selection = [0]
        browser._on_use_clicked()
        assert browser.chosen == ["000001"]

    def test_analyze_without_a_callback_is_a_no_op(self, panel_extension, tmp_path):
        # The browser is embeddable on its own, with nothing wired to its Analyze button.
        standalone = DandiBrowser(
            search_function=RecordingSearch(),
            verify_function=RecordingVerification(),
            verdict_cache=PhotometryVerdictCache(path=tmp_path / "verdicts.json"),
        )
        load_catalog(standalone)
        standalone.results_table.selection = [0]
        standalone._on_use_clicked()
        assert standalone.selected_summary.identifier == "000001"

    def test_only_verified_dandisets_reach_the_table(self, browser):
        browser.verify_function = RecordingVerification(confirmed={"000002"})
        load_catalog(browser)
        assert list(browser.results_table.value["Dandiset"]) == ["000002"]

    def test_a_dandiset_with_no_assets_is_never_verified(self, browser):
        browser.search_function = RecordingSearch(
            [
                make_summary(identifier="000001", file_count=0),
                make_summary(identifier="000002"),
            ]
        )
        load_catalog(browser)
        assert browser.verify_function.calls == [["000002"]]

    def test_verification_reports_what_it_read_and_what_it_missed(self, browser):
        browser.verify_function = RecordingVerification(confirmed={"000001"})
        load_catalog(browser)
        assert browser.status.object == (
            "Read **2** of 2 candidate(s): **1** holds fiber photometry. Datasets whose "
            "description never mentions photometry are not in this list; **Search every "
            "dandiset** reads the rest of the archive to find them."
        )

    def test_progress_is_shown_while_verifying_and_hidden_after(self, browser):
        browser.refresh_catalog()
        assert browser.verification_progress.visible
        assert browser.verification_progress.max == 2
        assert browser.search_button.disabled
        assert browser.stop_button.visible
        finish_verification(browser)
        assert not browser.verification_progress.visible
        assert not browser.search_button.disabled
        assert not browser.stop_button.visible

    def test_the_crawl_reads_every_dandiset_and_adds_what_it_finds(self, browser):
        load_catalog(browser)
        assert list(browser.results_table.value["Dandiset"]) == ["000001", "000002"]
        browser.crawl_archive()
        finish_verification(browser)
        # 000003 is only reachable by crawling; the search never returned it.
        assert list(browser.results_table.value["Dandiset"]) == ["000001", "000002", "000003"]
        assert browser.status.object == "Read **3** of 3 dandiset(s): **3** hold fiber photometry."

    def test_the_crawl_visits_the_search_hits_first(self, browser):
        load_catalog(browser)
        browser.crawl_archive()
        finish_verification(browser)
        # The two hits already on screen lead, then everything else smallest-first.
        assert browser.verify_function.calls[-1] == ["000001", "000002", "000003"]

    def test_stopping_ends_the_run_and_says_so(self, browser):
        browser.refresh_catalog()
        browser.stop_verification()
        finish_verification(browser)
        assert browser.status.object.startswith("Stopped after")
        assert not browser.search_button.disabled

    def test_analyze_without_a_selection_warns(self, browser):
        load_catalog(browser)
        browser._on_use_clicked()
        assert browser.chosen == []
        assert "Select a dandiset row first" in browser.status.object

    def test_inspect_streams_the_largest_asset(self, browser):
        load_catalog(browser)
        browser.results_table.selection = [0]
        browser.inspect_selected_dandiset()
        assert browser.preview_function.calls == [
            {
                "dandiset_id": "000001",
                "asset_path": "sub-01/ses-2.nwb",
                "series_name": None,
            }
        ]
        assert browser.preview_pane.panel.visible is True
        assert "the largest of 2 NWB asset(s)" in browser.status.object

    def test_inspect_without_a_selection_warns(self, browser):
        load_catalog(browser)
        browser.inspect_selected_dandiset()
        assert browser.preview_function.calls == []
        assert "Select a dandiset row first" in browser.status.object

    def test_inspect_reports_a_dandiset_with_no_nwb_assets(self, browser):
        browser.list_assets_function = lambda **kwargs: []
        load_catalog(browser)
        browser.results_table.selection = [0]
        browser.inspect_selected_dandiset()
        assert "holds no NWB assets" in browser.status.object
        assert browser.preview_pane.panel.visible is False

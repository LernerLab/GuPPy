"""Unit tests for the DANDI catalog browser and the shared photometry preview pane.

Every network call the browser makes — the catalog search, the photometry verification, the
archive-wide crawl's dandiset listing, the metadata fetch for what the crawl confirms, the asset
listing and the streaming preview — is a constructor injection point, so every test here runs
against in-memory stand-ins with no archive access.
"""

from threading import Event

import holoviews as hv
import numpy as np
import pytest

from guppy.frontend.dandi_browser import (
    CATALOG_COLUMNS,
    CHANNEL_COLUMNS,
    DEFAULT_SEARCH_TERM,
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
        assert list(frame["Species"]) == ["Mus musculus", "Rattus norvegicus"]
        assert list(frame["Subjects"]) == [10, 2]

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
    """Stand-in for ``search_dandisets`` over a fixed catalog."""

    def __init__(self, summaries=None):
        self.summaries = list(SUMMARIES if summaries is None else summaries)
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

# Which dandisets the stand-in read reports as holding GuPPy-readable photometry.
READABLE_IDENTIFIERS = {"000001"}


class RecordingVerification:
    """Stand-in for ``verify_dandisets`` over a fixed set of verdicts.

    Blocks on ``gate`` so a test can hold a read mid-flight and inspect what the widgets show
    while it runs, rather than only after it finishes.
    """

    def __init__(self, readable=None, unresolved=()):
        self.readable = set(READABLE_IDENTIFIERS if readable is None else readable)
        self.unresolved = set(unresolved)
        self.calls = []
        self.gate = Event()
        self.gate.set()

    def __call__(self, references, *, cache=None, on_verdict=None, should_stop=None, **kwargs):
        self.calls.append([reference.identifier for reference in references])
        self.gate.wait()
        verdicts = {}
        for reference in references:
            if should_stop is not None and should_stop():
                break
            if reference.identifier in self.unresolved:
                verdicts[reference.identifier] = None
            else:
                verdicts[reference.identifier] = reference.identifier in self.readable
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
        verdict_cache=PhotometryVerdictCache(path=tmp_path / "verdicts.json"),
    )
    browser.chosen = chosen
    return browser


def finish_read(browser):
    """Wait for the browser's running read and fold its verdicts into the table.

    The read runs on a worker thread that a periodic callback would normally poll; joining the
    thread and polling once is the same sequence without an event loop.
    """
    browser._verification["thread"].join()
    browser._poll_verification()


def listed(browser):
    """Return the dandiset identifiers the table is showing."""
    table = browser.results_table.value
    return list(table["Dandiset"]) if len(table) else []


class TestDandiBrowserSearch:
    def test_the_catalog_starts_empty_until_it_is_opened(self, browser):
        assert listed(browser) == []
        assert browser.search_function.calls == []

    def test_opening_runs_the_photometry_search_once(self, browser):
        browser.open_catalog()
        browser.open_catalog()
        assert browser.search_function.calls == [(DEFAULT_SEARCH_TERM,)]
        assert listed(browser) == ["000001", "000002"]

    def test_the_search_box_starts_on_the_photometry_term(self, browser):
        assert browser.query_input.value == DEFAULT_SEARCH_TERM

    def test_searching_sends_whatever_the_box_holds_to_the_archive(self, browser):
        browser.query_input.value = "neurotensin"
        browser.refresh_catalog()
        assert browser.search_function.calls == [("neurotensin",)]

    def test_a_six_digit_id_is_just_another_search(self, browser):
        browser.query_input.value = "000971"
        browser.refresh_catalog()
        assert browser.search_function.calls == [("000971",)]

    def test_an_empty_box_asks_for_a_term(self, browser):
        browser.query_input.value = "   "
        browser.refresh_catalog()
        assert browser.search_function.calls == []
        assert "Type something to search for" in browser.status.object

    def test_dandisets_holding_no_assets_are_dropped(self, panel_extension, tmp_path):
        browser = DandiBrowser(
            search_function=RecordingSearch(
                [
                    make_summary(identifier="000001", file_count=0),
                    make_summary(identifier="000002"),
                ]
            ),
            verify_function=RecordingVerification(),
            verdict_cache=PhotometryVerdictCache(path=tmp_path / "verdicts.json"),
        )
        browser.refresh_catalog()
        assert listed(browser) == ["000002"]

    def test_the_status_counts_what_is_listed(self, browser):
        browser.refresh_catalog()
        assert browser.status.object == "**2** dandiset(s). Select one to see its metadata."


class TestDandiBrowserNavigation:
    def test_the_list_is_the_opening_screen(self, browser):
        assert browser.list_view.visible is True
        assert browser.dandiset_view.visible is False

    def test_selecting_a_row_opens_that_dandiset(self, browser):
        browser.refresh_catalog()
        browser.results_table.selection = [0]
        assert browser.list_view.visible is False
        assert browser.dandiset_view.visible is True
        assert "000001 — Dorsomedial striatum dopamine" in browser.dandiset_details.object

    def test_going_back_returns_to_the_list_and_clears_the_selection(self, browser):
        browser.refresh_catalog()
        browser.results_table.selection = [0]
        browser.show_results()
        assert browser.list_view.visible is True
        assert browser.dandiset_view.visible is False
        assert browser.results_table.selection == []
        assert browser.dandiset_details.object == ""

    def test_analyze_hands_the_identifier_to_the_callback(self, browser):
        browser.refresh_catalog()
        browser.results_table.selection = [1]
        browser._on_use_clicked()
        assert browser.chosen == ["000002"]

    def test_analyze_without_a_selection_warns(self, browser):
        browser.refresh_catalog()
        browser._on_use_clicked()
        assert browser.chosen == []
        assert "Select a dandiset first" in browser.status.object

    def test_analyze_without_a_callback_is_a_no_op(self, panel_extension, tmp_path):
        # The browser is embeddable on its own, with nothing wired to its Analyze button.
        standalone = DandiBrowser(
            search_function=RecordingSearch(),
            verify_function=RecordingVerification(),
            verdict_cache=PhotometryVerdictCache(path=tmp_path / "verdicts.json"),
        )
        standalone.refresh_catalog()
        standalone.results_table.selection = [0]
        standalone._on_use_clicked()
        assert standalone.selected_summary.identifier == "000001"

    def test_a_new_search_returns_to_the_list(self, browser):
        browser.refresh_catalog()
        browser.results_table.selection = [0]
        browser.refresh_catalog()
        assert browser.list_view.visible is True
        assert browser.dandiset_view.visible is False


class TestDandiBrowserVerification:
    def test_the_filter_is_off_until_it_is_asked_for(self, browser):
        browser.refresh_catalog()
        assert browser.verify_listed.value is False
        assert browser.verify_function.calls == []
        assert listed(browser) == ["000001", "000002"]

    def test_ticking_it_reads_the_listed_dandisets(self, browser):
        browser.refresh_catalog()
        browser.verify_listed.value = True
        finish_read(browser)
        assert browser.verify_function.calls == [["000002", "000001"]]

    def test_the_smallest_dandisets_are_read_first(self, browser):
        # 000002 holds 5 files to 000001's 100, so it settles sooner and goes first.
        browser.refresh_catalog()
        browser.verify_listed.value = True
        finish_read(browser)
        assert browser.verify_function.calls[0] == ["000002", "000001"]

    def test_only_the_readable_dandisets_are_kept(self, browser):
        browser.refresh_catalog()
        browser.verify_listed.value = True
        finish_read(browser)
        assert listed(browser) == ["000001"]

    def test_unticking_it_brings_the_others_back(self, browser):
        browser.refresh_catalog()
        browser.verify_listed.value = True
        finish_read(browser)
        browser.verify_listed.value = False
        assert listed(browser) == ["000001", "000002"]

    def test_the_status_reports_what_was_read(self, browser):
        browser.refresh_catalog()
        browser.verify_listed.value = True
        finish_read(browser)
        assert browser.status.object == ("Read **2** of 2 listed dandiset(s): **1** holds fiber photometry.")

    def test_a_dandiset_that_could_not_be_read_is_reported_separately(self, browser):
        browser.verify_function = RecordingVerification(readable={"000001"}, unresolved={"000002"})
        browser.refresh_catalog()
        browser.verify_listed.value = True
        finish_read(browser)
        assert browser.status.object == (
            "Read **2** of 2 listed dandiset(s): **1** holds fiber photometry. 1 could not "
            "be read in full and are not accounted for either way; reading again retries them."
        )

    def test_progress_shows_while_reading_and_hides_after(self, browser):
        browser.refresh_catalog()
        browser.verify_function.gate.clear()
        browser.verify_listed.value = True
        assert browser.verification_progress.visible is True
        assert browser.verification_progress.max == 2
        assert browser.stop_button.visible is True
        assert browser.search_button.disabled is True

        browser.verify_function.gate.set()
        finish_read(browser)
        assert browser.verification_progress.visible is False
        assert browser.stop_button.visible is False
        assert browser.search_button.disabled is False

    def test_stopping_ends_the_read_and_says_so(self, browser):
        browser.refresh_catalog()
        browser.verify_listed.value = True
        browser.stop_verification()
        finish_read(browser)
        assert browser.status.object.startswith("Stopped after")
        assert browser.search_button.disabled is False

    def test_a_new_search_forgets_the_previous_verdicts(self, browser):
        browser.refresh_catalog()
        browser.verify_listed.value = True
        finish_read(browser)
        browser.refresh_catalog()
        assert browser.verdicts == {}
        assert browser.verify_listed.value is False
        assert listed(browser) == ["000001", "000002"]

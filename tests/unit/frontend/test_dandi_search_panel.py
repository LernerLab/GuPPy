"""Unit tests for the panel that finds a dandiset on the DANDI Archive.

Every network call it makes — the catalog search and the photometry verification that reads the
listed dandisets' files — is a constructor injection point, so every test here runs against
in-memory stand-ins with no archive access.
"""

from threading import Event

import pytest

from guppy.frontend.dandi_search_panel import (
    CATALOG_COLUMNS,
    DEFAULT_SEARCH_TERM,
    DandiSearchPanel,
    catalog_dataframe,
    describe_dandiset,
)
from guppy.utils.dandi_filter import PhotometryVerdictCache
from guppy.utils.dandi_search import DandisetSummary


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
        "subject_count": 10,
        "file_count": 100,
        "size_in_bytes": 60_000_000,
        "contributors": ("Lerner, Talia",),
        "license_terms": ("spdx:CC-BY-4.0",),
        "url": "https://dandiarchive.org/dandiset/000001",
        "is_published": True,
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
        approaches=("optogenetic approach",),
        subject_count=2,
        file_count=5,
        is_published=False,
    ),
]


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


class TestDescribeDandiset:
    def test_reports_identity_scale_and_detected_terms(self):
        text = describe_dandiset(make_summary())
        assert "000001 — Dorsomedial striatum dopamine" in text
        assert "https://dandiarchive.org/dandiset/000001" in text
        assert "**Species:** Mus musculus" in text
        assert "**Subjects:** 10" in text
        assert "GCaMP7b recordings in the DMS." in text

    def test_spdx_prefix_is_stripped_from_the_license(self):
        assert "CC-BY-4.0" in describe_dandiset(make_summary())
        assert "spdx:" not in describe_dandiset(make_summary())

    def test_absent_fields_render_as_a_dash(self):
        text = describe_dandiset(make_summary(species=(), keywords=(), license_terms=()))
        assert "**Species:** —" in text
        assert "**Keywords:** —" in text
        assert "license not stated" in text

    def test_long_contributor_list_is_abbreviated(self):
        text = describe_dandiset(make_summary(contributors=tuple(f"Author {index}" for index in range(6))))
        assert "Author 3, et al." in text
        assert "Author 4" not in text


class RecordingSearch:
    """Stand-in for ``search_dandisets`` over a fixed catalog."""

    def __init__(self, summaries=None):
        self.summaries = list(SUMMARIES if summaries is None else summaries)
        self.calls = []

    def __call__(self, *, terms):
        self.calls.append(tuple(terms))
        return list(self.summaries)


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
def search_panel(panel_extension, tmp_path):
    chosen = []
    search_panel = DandiSearchPanel(
        on_dandiset_selected=chosen.append,
        search_function=RecordingSearch(),
        verify_function=RecordingVerification(),
        verdict_cache=PhotometryVerdictCache(path=tmp_path / "verdicts.json"),
    )
    search_panel.chosen = chosen
    return search_panel


def finish_read(search_panel):
    """Wait for the search_panel's running read and fold its verdicts into the table.

    The read runs on a worker thread that a periodic callback would normally poll; joining the
    thread and polling once is the same sequence without an event loop.
    """
    search_panel._verification["thread"].join()
    search_panel._poll_verification()


def listed(search_panel):
    """Return the dandiset identifiers the table is showing."""
    table = search_panel.results_table.value
    return list(table["Dandiset"]) if len(table) else []


class TestDandiSearchPanelSearch:
    def test_the_catalog_starts_empty_until_it_is_opened(self, search_panel):
        assert listed(search_panel) == []
        assert search_panel.search_function.calls == []

    def test_opening_runs_the_photometry_search_once(self, search_panel):
        search_panel.open_catalog()
        search_panel.open_catalog()
        assert search_panel.search_function.calls == [(DEFAULT_SEARCH_TERM,)]
        assert listed(search_panel) == ["000001", "000002"]

    def test_the_search_box_starts_on_the_photometry_term(self, search_panel):
        assert search_panel.query_input.value == DEFAULT_SEARCH_TERM

    def test_searching_sends_whatever_the_box_holds_to_the_archive(self, search_panel):
        search_panel.query_input.value = "neurotensin"
        search_panel.refresh_catalog()
        assert search_panel.search_function.calls == [("neurotensin",)]

    def test_a_six_digit_id_is_just_another_search(self, search_panel):
        search_panel.query_input.value = "000971"
        search_panel.refresh_catalog()
        assert search_panel.search_function.calls == [("000971",)]

    def test_an_empty_box_asks_for_a_term(self, search_panel):
        search_panel.query_input.value = "   "
        search_panel.refresh_catalog()
        assert search_panel.search_function.calls == []
        assert "Type something to search for" in search_panel.status.object

    def test_dandisets_holding_no_assets_are_dropped(self, panel_extension, tmp_path):
        search_panel = DandiSearchPanel(
            search_function=RecordingSearch(
                [
                    make_summary(identifier="000001", file_count=0),
                    make_summary(identifier="000002"),
                ]
            ),
            verify_function=RecordingVerification(),
            verdict_cache=PhotometryVerdictCache(path=tmp_path / "verdicts.json"),
        )
        search_panel.refresh_catalog()
        assert listed(search_panel) == ["000002"]

    def test_the_status_counts_what_is_listed(self, search_panel):
        search_panel.refresh_catalog()
        assert search_panel.status.object == "**2** dandiset(s). Select one to see its metadata."


class TestDandiSearchPanelNavigation:
    def test_the_list_is_the_opening_screen(self, search_panel):
        assert search_panel.list_view.visible is True
        assert search_panel.dandiset_view.visible is False

    def test_selecting_a_row_opens_that_dandiset(self, search_panel):
        search_panel.refresh_catalog()
        search_panel.results_table.selection = [0]
        assert search_panel.list_view.visible is False
        assert search_panel.dandiset_view.visible is True
        assert "000001 — Dorsomedial striatum dopamine" in search_panel.dandiset_details.object

    def test_going_back_returns_to_the_list_and_clears_the_selection(self, search_panel):
        search_panel.refresh_catalog()
        search_panel.results_table.selection = [0]
        search_panel.show_results()
        assert search_panel.list_view.visible is True
        assert search_panel.dandiset_view.visible is False
        assert search_panel.results_table.selection == []
        assert search_panel.dandiset_details.object == ""

    def test_analyze_hands_the_identifier_to_the_callback(self, search_panel):
        search_panel.refresh_catalog()
        search_panel.results_table.selection = [1]
        search_panel._on_use_clicked()
        assert search_panel.chosen == ["000002"]

    def test_analyze_without_a_selection_warns(self, search_panel):
        search_panel.refresh_catalog()
        search_panel._on_use_clicked()
        assert search_panel.chosen == []
        assert "Select a dandiset first" in search_panel.status.object

    def test_analyze_without_a_callback_is_a_no_op(self, panel_extension, tmp_path):
        # The search_panel is embeddable on its own, with nothing wired to its Analyze button.
        standalone = DandiSearchPanel(
            search_function=RecordingSearch(),
            verify_function=RecordingVerification(),
            verdict_cache=PhotometryVerdictCache(path=tmp_path / "verdicts.json"),
        )
        standalone.refresh_catalog()
        standalone.results_table.selection = [0]
        standalone._on_use_clicked()
        assert standalone.selected_summary.identifier == "000001"

    def test_a_new_search_returns_to_the_list(self, search_panel):
        search_panel.refresh_catalog()
        search_panel.results_table.selection = [0]
        search_panel.refresh_catalog()
        assert search_panel.list_view.visible is True
        assert search_panel.dandiset_view.visible is False


class TestDandiSearchPanelVerification:
    def test_the_filter_is_off_until_it_is_asked_for(self, search_panel):
        search_panel.refresh_catalog()
        assert search_panel.verify_listed.value is False
        assert search_panel.verify_function.calls == []
        assert listed(search_panel) == ["000001", "000002"]

    def test_ticking_it_reads_the_listed_dandisets(self, search_panel):
        search_panel.refresh_catalog()
        search_panel.verify_listed.value = True
        finish_read(search_panel)
        assert search_panel.verify_function.calls == [["000002", "000001"]]

    def test_the_smallest_dandisets_are_read_first(self, search_panel):
        # 000002 holds 5 files to 000001's 100, so it settles sooner and goes first.
        search_panel.refresh_catalog()
        search_panel.verify_listed.value = True
        finish_read(search_panel)
        assert search_panel.verify_function.calls[0] == ["000002", "000001"]

    def test_only_the_readable_dandisets_are_kept(self, search_panel):
        search_panel.refresh_catalog()
        search_panel.verify_listed.value = True
        finish_read(search_panel)
        assert listed(search_panel) == ["000001"]

    def test_unticking_it_brings_the_others_back(self, search_panel):
        search_panel.refresh_catalog()
        search_panel.verify_listed.value = True
        finish_read(search_panel)
        search_panel.verify_listed.value = False
        assert listed(search_panel) == ["000001", "000002"]

    def test_the_status_reports_what_was_read(self, search_panel):
        search_panel.refresh_catalog()
        search_panel.verify_listed.value = True
        finish_read(search_panel)
        assert search_panel.status.object == ("Read **2** of 2 listed dandiset(s): **1** holds fiber photometry.")

    def test_a_dandiset_that_could_not_be_read_is_reported_separately(self, search_panel):
        search_panel.verify_function = RecordingVerification(readable={"000001"}, unresolved={"000002"})
        search_panel.refresh_catalog()
        search_panel.verify_listed.value = True
        finish_read(search_panel)
        assert search_panel.status.object == (
            "Read **2** of 2 listed dandiset(s): **1** holds fiber photometry. 1 could not "
            "be read in full and are not accounted for either way; reading again retries them."
        )

    def test_progress_shows_while_reading_and_hides_after(self, search_panel):
        search_panel.refresh_catalog()
        search_panel.verify_function.gate.clear()
        search_panel.verify_listed.value = True
        assert search_panel.verification_progress.visible is True
        assert search_panel.verification_progress.max == 2
        assert search_panel.stop_button.visible is True
        assert search_panel.search_button.disabled is True

        search_panel.verify_function.gate.set()
        finish_read(search_panel)
        assert search_panel.verification_progress.visible is False
        assert search_panel.stop_button.visible is False
        assert search_panel.search_button.disabled is False

    def test_stopping_ends_the_read_and_says_so(self, search_panel):
        search_panel.refresh_catalog()
        search_panel.verify_listed.value = True
        search_panel.stop_verification()
        finish_read(search_panel)
        assert search_panel.status.object.startswith("Stopped after")
        assert search_panel.search_button.disabled is False

    def test_a_new_search_forgets_the_previous_verdicts(self, search_panel):
        search_panel.refresh_catalog()
        search_panel.verify_listed.value = True
        finish_read(search_panel)
        search_panel.refresh_catalog()
        assert search_panel.verdicts == {}
        assert search_panel.verify_listed.value is False
        assert listed(search_panel) == ["000001", "000002"]

import panel as pn

from guppy.orchestration import transients_view
from guppy.utils import page_links
from guppy.utils.page_links import Page


def test_open_transients_view_offers_the_page_for_the_resolved_runs(panel_extension, monkeypatch):
    monkeypatch.setattr(transients_view, "resolve_run_folders", lambda session_folders, params: ["/a/output_1"])
    peaks_page = pn.pane.Markdown("peaks")
    calls = []

    def build_peaks_view_page(run_folders, select_for_transients):
        calls.append((run_folders, select_for_transients))
        return peaks_page

    monkeypatch.setattr(transients_view, "build_peaks_view_page", build_peaks_view_page)

    with page_links.collect() as pages:
        transients_view.open_transients_view(
            ["/a"], {"selectForTransientsComputation": "z_score", "combine_data": False}
        )

    assert calls == [(["/a/output_1"], "z_score")]
    assert pages == [Page(name="Transient peaks", content=peaks_page)]

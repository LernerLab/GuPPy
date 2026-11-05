import panel as pn

from guppy.orchestration import preprocess_view
from guppy.utils import page_links
from guppy.utils.page_links import Page


def test_open_preprocess_view_offers_the_page_for_the_resolved_runs(panel_extension, monkeypatch):
    monkeypatch.setattr(preprocess_view, "resolve_run_folders", lambda session_folders, params: ["/a/output_1"])
    view_page = pn.pane.Markdown("preprocessing")
    calls = []

    def build_preprocess_view_page(*, run_folders):
        calls.append(run_folders)
        return view_page

    monkeypatch.setattr(preprocess_view, "build_preprocess_view_page", build_preprocess_view_page)

    with page_links.collect() as pages:
        preprocess_view.open_preprocess_view(["/a"], {"combine_data": False})

    assert calls == [["/a/output_1"]]
    assert pages == [Page(name="Preprocessing", content=view_page)]

import panel as pn

from guppy.orchestration import artifact_view
from guppy.utils import page_links
from guppy.utils.page_links import Page


def test_open_artifact_view_offers_the_page_for_the_resolved_runs(panel_extension, monkeypatch):
    monkeypatch.setattr(artifact_view, "resolve_run_folders", lambda session_folders, params: ["/a/output_1"])
    review_page = pn.pane.Markdown("review")
    calls = []

    def build_artifact_review_page(*, run_folders):
        calls.append(run_folders)
        return review_page

    monkeypatch.setattr(artifact_view, "build_artifact_review_page", build_artifact_review_page)

    with page_links.collect() as pages:
        artifact_view.open_artifact_view(["/a"], {"combine_data": False})

    assert calls == [["/a/output_1"]]
    assert pages == [Page(name="Artifact Removal", content=review_page)]

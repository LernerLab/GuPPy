from guppy.frontend.group_labeling import GroupLabelingPage
from guppy.orchestration import group_labeling
from guppy.orchestration.group_labeling import (
    build_group_labeling_page,
    orchestrate_group_labeling_page,
)


class TestBuildGroupLabelingPage:
    def test_builds_page_from_input_parameters(self, panel_extension, tmp_path):
        page = build_group_labeling_page(inputParameters={"abspath": str(tmp_path), "selected_group_folders": []})

        assert isinstance(page, GroupLabelingPage)


class TestOrchestrateGroupLabelingPage:
    def test_serves_the_page(self, panel_extension, monkeypatch, tmp_path):
        served_titles = []
        monkeypatch.setattr(group_labeling, "serve_page", lambda *, template, title: served_titles.append(title))

        orchestrate_group_labeling_page({"abspath": str(tmp_path), "selected_group_folders": []})

        assert served_titles == ["Label Groups"]

import panel as pn

from guppy.frontend.group_labeling import GroupLabelingPage
from guppy.orchestration.group_labeling import (
    build_group_labeling_page,
    orchestrate_group_labeling_page,
)
from guppy.utils import page_links


class TestBuildGroupLabelingPage:
    def test_builds_page_from_input_parameters(self, panel_extension, tmp_path):
        page = build_group_labeling_page(inputParameters={"abspath": str(tmp_path), "selected_group_folders": []})

        assert isinstance(page, GroupLabelingPage)


class TestOrchestrateGroupLabelingPage:
    def test_offers_the_page(self, panel_extension, tmp_path):
        with page_links.collect() as pages:
            orchestrate_group_labeling_page({"abspath": str(tmp_path), "selected_group_folders": []})

        (page,) = pages
        assert page.name == "Label Groups"
        (row,) = page.content
        assert isinstance(row, pn.Row)

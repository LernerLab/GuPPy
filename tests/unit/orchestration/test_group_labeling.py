import panel as pn

from guppy.frontend.group_labeling import GroupLabelingPage
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
        served_ports = []
        monkeypatch.setattr(pn.template.BootstrapTemplate, "show", lambda self, port: served_ports.append(port))

        orchestrate_group_labeling_page({"abspath": str(tmp_path), "selected_group_folders": []})

        assert len(served_ports) == 1

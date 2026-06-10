import socket
import time

import panel as pn
import pytest
from playwright.sync_api import expect

from guppy.frontend.frontend_utils import scanPortsAndFind
from guppy.orchestration.metadata import build_project_metadata_template


@pytest.fixture(scope="session")
def project_metadata_server_url(panel_extension, ui_base_dir, tmp_path_factory):
    """Serve the Project Metadata GUI (empty form) on a free local port."""
    temporary_folder = tmp_path_factory.mktemp("project_metadata_session")
    template = build_project_metadata_template(
        project_metadata={},
        project_yaml_path=str(temporary_folder / "nwb_project_metadata.yaml"),
    )
    port = scanPortsAndFind()
    pn.serve(template, port=port, show=False, threaded=True)

    for _ in range(50):
        try:
            connection = socket.create_connection(("localhost", port), timeout=0.1)
            connection.close()
            break
        except OSError:
            time.sleep(0.05)

    yield f"http://localhost:{port}"

    pn.state.kill_all_servers()


@pytest.mark.ui
def test_page_title_contains_project_metadata_gui(page, project_metadata_server_url):
    page.goto(project_metadata_server_url)
    assert "Project Metadata GUI" in page.title()


@pytest.mark.ui
def test_build_and_save_buttons_visible(page, project_metadata_server_url):
    page.goto(project_metadata_server_url)
    expect(page.get_by_role("button", name="Show / Refresh YAML from form above")).to_be_visible()
    expect(page.get_by_role("button", name="Save Project Metadata")).to_be_visible()
    expect(page.get_by_role("button", name="Load Existing Project YAML")).to_be_visible()


@pytest.mark.ui
def test_hardware_component_card_visible(page, project_metadata_server_url):
    page.goto(project_metadata_server_url)
    expect(page.get_by_text("Optical Fiber").first).to_be_visible()

import socket
import time

import panel as pn
import pytest
from playwright.sync_api import expect

from guppy.frontend.frontend_utils import scanPortsAndFind
from guppy.orchestration.metadata import build_session_metadata_template


@pytest.fixture(scope="session")
def session_metadata_server_url(panel_extension, ui_base_dir, tmp_path_factory):
    """Serve the Session Metadata GUI on a free local port."""
    temporary_folder = tmp_path_factory.mktemp("session_metadata_session")
    template = build_session_metadata_template(
        session_label="Photo_63 (1)",
        session_metadata={},
        session_yaml_path=str(temporary_folder / "nwb_session_metadata.yaml"),
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
def test_page_title_contains_session_metadata_gui(page, session_metadata_server_url):
    page.goto(session_metadata_server_url)
    assert "Session Metadata GUI" in page.title()


@pytest.mark.ui
def test_session_fields_and_buttons_visible(page, session_metadata_server_url):
    page.goto(session_metadata_server_url)
    expect(page.get_by_text("Session description").first).to_be_visible()
    expect(page.get_by_text("Subject ID").first).to_be_visible()
    expect(page.get_by_role("button", name="Save Session Metadata")).to_be_visible()

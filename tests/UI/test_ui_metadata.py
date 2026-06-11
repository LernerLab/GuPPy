import socket
import time

import panel as pn
import pytest
from playwright.sync_api import expect

from guppy.frontend.frontend_utils import scanPortsAndFind
from guppy.orchestration.metadata import build_metadata_template
from guppy.utils.nwb_metadata import Channel


@pytest.fixture(scope="session")
def metadata_server_url(panel_extension, ui_base_dir, tmp_path_factory):
    """Serve the Metadata GUI on a free local port (two fixed dms channels)."""
    temporary_folder = tmp_path_factory.mktemp("metadata_session")
    channels = [Channel("dms", "control", "Dv1A"), Channel("dms", "signal", "Dv2A")]
    template = build_metadata_template(
        session_label="Photo_63 (1)",
        channels=channels,
        metadata={},
        metadata_yaml_path=str(temporary_folder / "nwb_metadata.yaml"),
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
def test_page_title_contains_metadata_gui(page, metadata_server_url):
    page.goto(metadata_server_url)
    assert "Metadata GUI" in page.title()


@pytest.mark.ui
def test_session_subject_fields_visible(page, metadata_server_url):
    page.goto(metadata_server_url)
    # Labels are humanized from the metadata keys.
    expect(page.get_by_text("Session description").first).to_be_visible()
    expect(page.get_by_text("Subject ID").first).to_be_visible()


@pytest.mark.ui
def test_fixed_channel_rows_visible(page, metadata_server_url):
    page.goto(metadata_server_url)
    # Rows are fixed by storesList; each channel renders a chip with its source store name.
    expect(page.get_by_text("Dv1A").first).to_be_visible()
    expect(page.get_by_text("Dv2A").first).to_be_visible()
    expect(page.get_by_text("signal").first).to_be_visible()
    expect(page.get_by_text("control").first).to_be_visible()


@pytest.mark.ui
def test_buttons_and_device_library_visible(page, metadata_server_url):
    page.goto(metadata_server_url)
    expect(page.get_by_role("button", name="Build & preview YAML")).to_be_visible()
    expect(page.get_by_role("button", name="Save metadata")).to_be_visible()
    # The arbitrary-YAML loader is a file input introduced by a labelled utility strip.
    expect(page.get_by_text("Reuse metadata from another session").first).to_be_visible()
    expect(page.get_by_text("Optical fibers").first).to_be_visible()

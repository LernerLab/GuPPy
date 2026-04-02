import socket
import time

import panel as pn
import pytest
from playwright.sync_api import expect

from guppy.frontend.frontend_utils import scanPortsAndFind
from guppy.orchestration.storenames import build_storenames_template


@pytest.fixture(scope="session")
def storenames_server_url(panel_extension, ui_base_dir, tmp_path_factory):
    """Serve the Storenames GUI on a free local port.

    Yields the base URL of the running server.
    """
    temporary_folder = tmp_path_factory.mktemp("storenames_session")
    template = build_storenames_template(
        events=["Dv1A", "Dv1B"],
        flags=[],
        folder_path=str(temporary_folder),
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
def test_page_title_contains_storenames_gui(page, storenames_server_url):
    page.goto(storenames_server_url)
    expect(page.get_by_text("Select Storenames").first).to_be_visible()
    assert "Storenames GUI" in page.title()


@pytest.mark.ui
def test_storenames_to_repeat_checkbox_visible(page, storenames_server_url):
    page.goto(storenames_server_url)
    expect(page.get_by_text("Storenames to repeat").first).to_be_visible()


@pytest.mark.ui
def test_select_storenames_button_visible(page, storenames_server_url):
    page.goto(storenames_server_url)
    expect(page.get_by_role("button", name="Select Storenames")).to_be_visible()


# TODO: Debug this test
# @pytest.mark.ui
# def test_show_selected_configuration_button_visible(page, storenames_server_url):
#     page.goto(storenames_server_url)
#     expect(page.get_by_role("button", name="Show Selected Configuration")).to_be_visible()


@pytest.mark.ui
def test_save_button_visible(page, storenames_server_url):
    page.goto(storenames_server_url)
    expect(page.get_by_role("button", name="Save").first).to_be_visible()


@pytest.mark.ui
def test_selected_store_names_label_visible(page, storenames_server_url):
    page.goto(storenames_server_url)
    expect(page.get_by_text("Selected Store Names").first).to_be_visible()


@pytest.mark.ui
def test_location_to_stores_list_file_label_visible(page, storenames_server_url):
    page.goto(storenames_server_url)
    expect(page.get_by_text("Location to Stores List file").first).to_be_visible()


@pytest.mark.ui
def test_overwrite_button_visible(page, storenames_server_url):
    page.goto(storenames_server_url)
    expect(page.get_by_text("over-write storeslist file or create a new one?").first).to_be_visible()

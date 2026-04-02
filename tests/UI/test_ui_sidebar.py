import pytest
from playwright.sync_api import expect


@pytest.mark.ui
def test_save_to_file_button_visible(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_role("button", name="Save to file...")).to_be_visible()


@pytest.mark.ui
def test_open_storenames_gui_button_visible(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_role("button", name="Open Storenames GUI")).to_be_visible()


@pytest.mark.ui
def test_read_raw_data_button_visible(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_role("button", name="Read Raw Data")).to_be_visible()


@pytest.mark.ui
def test_preprocess_button_visible(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_role("button", name="Preprocess and Remove Artifacts")).to_be_visible()


@pytest.mark.ui
def test_psth_computation_button_visible(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_role("button", name="PSTH Computation")).to_be_visible()


@pytest.mark.ui
def test_open_visualization_gui_button_visible(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_role("button", name="Open Visualization GUI")).to_be_visible()


@pytest.mark.ui
def test_step_1_label_visible(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_text("Step 1 : Save Input Parameters").first).to_be_visible()


@pytest.mark.ui
def test_step_3_label_visible(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_text("Step 3 : Read Raw Data").first).to_be_visible()


@pytest.mark.ui
def test_step_5_label_visible(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_text("Step 5 : PSTH Computation").first).to_be_visible()

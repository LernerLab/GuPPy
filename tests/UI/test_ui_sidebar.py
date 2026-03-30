import pytest


@pytest.mark.ui
def test_save_to_file_button_visible(page, live_server_url):
    page.goto(live_server_url)
    page.wait_for_selector("text=Save to file...")
    assert page.get_by_role("button", name="Save to file...").is_visible()


@pytest.mark.ui
def test_open_storenames_gui_button_visible(page, live_server_url):
    page.goto(live_server_url)
    page.wait_for_selector("text=Open Storenames GUI")
    assert page.get_by_role("button", name="Open Storenames GUI").is_visible()


@pytest.mark.ui
def test_read_raw_data_button_visible(page, live_server_url):
    page.goto(live_server_url)
    page.wait_for_selector("text=Read Raw Data")
    assert page.get_by_role("button", name="Read Raw Data").is_visible()


@pytest.mark.ui
def test_preprocess_button_visible(page, live_server_url):
    page.goto(live_server_url)
    page.wait_for_selector("text=Preprocess and Remove Artifacts")
    assert page.get_by_role("button", name="Preprocess and Remove Artifacts").is_visible()


@pytest.mark.ui
def test_psth_computation_button_visible(page, live_server_url):
    page.goto(live_server_url)
    page.wait_for_selector("text=PSTH Computation")
    assert page.get_by_role("button", name="PSTH Computation").is_visible()


@pytest.mark.ui
def test_open_visualization_gui_button_visible(page, live_server_url):
    page.goto(live_server_url)
    page.wait_for_selector("text=Open Visualization GUI")
    assert page.get_by_role("button", name="Open Visualization GUI").is_visible()


@pytest.mark.ui
def test_step_1_label_visible(page, live_server_url):
    page.goto(live_server_url)
    page.wait_for_selector("text=Step 1")
    assert page.get_by_text("Step 1 : Save Input Parameters").first.is_visible()


@pytest.mark.ui
def test_step_3_label_visible(page, live_server_url):
    page.goto(live_server_url)
    page.wait_for_selector("text=Step 3")
    assert page.get_by_text("Step 3 : Read Raw Data").first.is_visible()


@pytest.mark.ui
def test_step_5_label_visible(page, live_server_url):
    page.goto(live_server_url)
    page.wait_for_selector("text=Step 5")
    assert page.get_by_text("Step 5 : PSTH Computation").first.is_visible()

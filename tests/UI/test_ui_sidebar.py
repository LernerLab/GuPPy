import pytest
from playwright.sync_api import expect


@pytest.mark.ui
def test_save_to_file_button_removed(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_role("button", name="Save to file...")).to_have_count(0)


@pytest.mark.ui
def test_label_stores_button_visible(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_role("button", name="Label Stores")).to_be_visible()


@pytest.mark.ui
def test_read_raw_data_button_visible(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_role("button", name="Read Raw Data")).to_be_visible()


@pytest.mark.ui
def test_preprocess_button_visible(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_role("button", name="Preprocess", exact=True)).to_be_visible()


@pytest.mark.ui
def test_select_artifact_windows_button_visible(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_role("button", name="Select Artifact Windows")).to_be_visible()


@pytest.mark.ui
def test_remove_artifacts_button_visible(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_role("button", name="Remove Artifacts", exact=True)).to_be_visible()


@pytest.mark.ui
def test_tonic_analysis_button_visible(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_role("button", name="Tonic Analysis")).to_be_visible()


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
    expect(page.get_by_text("Step 1 : Label Stores").first).to_be_visible()


@pytest.mark.ui
def test_step_2_label_visible(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_text("Step 2 : Read Raw Data").first).to_be_visible()


@pytest.mark.ui
def test_step_4_label_visible(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_text("Step 4 : PSTH Computation").first).to_be_visible()


@pytest.mark.ui
def test_step_3_label_visible(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_text("Step 3 : Preprocess").first).to_be_visible()


@pytest.mark.ui
def test_optional_artifact_step_labels_visible(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_text("Select Artifact Windows (optional)").first).to_be_visible()
    expect(page.get_by_text("Remove Artifacts (optional)").first).to_be_visible()

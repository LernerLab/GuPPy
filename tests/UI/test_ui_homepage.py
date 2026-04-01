import pytest
from playwright.sync_api import expect


@pytest.mark.ui
def test_page_title_contains_input_parameters_gui(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_text("Individual Analysis").first).to_be_visible()
    assert "Input Parameters GUI" in page.title()


@pytest.mark.ui
def test_individual_analysis_card_heading_visible(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_text("Individual Analysis").first).to_be_visible()


@pytest.mark.ui
def test_group_analysis_card_heading_visible(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_text("Group Analysis").first).to_be_visible()


@pytest.mark.ui
def test_visualization_parameters_card_heading_visible(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_text("Visualization Parameters").first).to_be_visible()


@pytest.mark.ui
def test_no_javascript_errors_on_load(page, live_server_url):
    errors = []
    page.on("pageerror", lambda error: errors.append(error))
    page.goto(live_server_url)
    expect(page.get_by_text("Individual Analysis").first).to_be_visible()
    assert errors == [], f"JavaScript errors on page load: {errors}"

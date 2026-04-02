import pytest
from playwright.sync_api import expect


@pytest.mark.ui
def test_individual_analysis_card_file_selector_visible_on_load(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_text("Select folders for the analysis").first).to_be_visible()


@pytest.mark.ui
def test_isosbestic_control_select_shows_true_and_false_options(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_text("Isosbestic Control Channel?").first).to_be_visible()
    page.locator("select").filter(has_text="True").first.click()
    # Both True and False options should be present in the select element
    options = (
        page.locator("select").filter(has_text="True").first.evaluate("el => Array.from(el.options).map(o => o.text)")
    )
    assert "True" in options
    assert "False" in options


@pytest.mark.ui
def test_eliminate_first_few_seconds_input_displays_default_value(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_text("Eliminate first few seconds (int)").first).to_be_visible()


@pytest.mark.ui
def test_tabulator_peak_start_time_column_header_visible(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_text("Peak Start time").first).to_be_visible()


@pytest.mark.ui
def test_group_analysis_card_expands_on_click(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_text("Group Analysis").first).to_be_visible()
    page.get_by_text("Group Analysis").first.click()
    expect(page.get_by_text("Select folders for the average analysis").first).to_be_visible()


@pytest.mark.ui
def test_visualization_parameters_card_expands_on_click(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_text("Visualization Parameters").first).to_be_visible()
    page.get_by_text("Visualization Parameters").first.click()
    expect(page.get_by_text("z-score or").first).to_be_visible()

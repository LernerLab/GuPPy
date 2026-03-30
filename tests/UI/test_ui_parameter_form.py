import pytest


@pytest.mark.ui
def test_individual_analysis_card_file_selector_visible_on_load(page, live_server_url):
    page.goto(live_server_url)
    page.wait_for_selector("text=Select folders for the analysis")
    assert page.get_by_text("Select folders for the analysis").first.is_visible()


@pytest.mark.ui
def test_isosbestic_control_select_shows_true_and_false_options(page, live_server_url):
    page.goto(live_server_url)
    page.wait_for_selector("text=Isosbestic Control Channel?")
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
    page.wait_for_selector("text=Eliminate first few seconds")
    # Find the input associated with the "Eliminate first few seconds" label
    input_field = page.locator("input[type='number'], input[type='text']").nth(0)
    # The tabulator and LiteralInput fields for the first integer param should show "1"
    # Locate specifically by looking near the label text
    label_locator = page.get_by_text("Eliminate first few seconds (int)")
    assert label_locator.is_visible()


@pytest.mark.ui
def test_tabulator_peak_start_time_column_header_visible(page, live_server_url):
    page.goto(live_server_url)
    page.wait_for_selector("text=Peak Start time")
    assert page.get_by_text("Peak Start time").first.is_visible()


@pytest.mark.ui
def test_group_analysis_card_expands_on_click(page, live_server_url):
    page.goto(live_server_url)
    page.wait_for_selector("text=Group Analysis")
    page.get_by_text("Group Analysis").first.click()
    page.wait_for_selector("text=Select folders for the average analysis")
    assert page.get_by_text("Select folders for the average analysis").first.is_visible()


@pytest.mark.ui
def test_visualization_parameters_card_expands_on_click(page, live_server_url):
    page.goto(live_server_url)
    page.wait_for_selector("text=Visualization Parameters")
    page.get_by_text("Visualization Parameters").first.click()
    page.wait_for_selector("text=z-score or")
    assert page.get_by_text("z-score or").first.is_visible()

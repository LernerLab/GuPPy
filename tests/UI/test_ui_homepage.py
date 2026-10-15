import panel as pn
import pytest
from playwright.sync_api import expect

from guppy.frontend.frontend_utils import scanPortsAndFind
from guppy.orchestration.home import build_homepage


@pytest.mark.ui
def test_page_title_contains_input_parameters_gui(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_text("Parameter Selection").first).to_be_visible()
    assert "Input Parameters GUI" in page.title()


@pytest.mark.ui
def test_individual_analysis_card_heading_visible(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_text("Parameter Selection").first).to_be_visible()


@pytest.mark.ui
def test_group_analysis_card_heading_visible(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_text("Group Output Folder Selection").first).to_be_visible()


@pytest.mark.ui
def test_no_javascript_errors_on_load(page, live_server_url):
    errors = []
    page.on("pageerror", lambda error: errors.append(error))
    page.goto(live_server_url)
    expect(page.get_by_text("Parameter Selection").first).to_be_visible()
    assert errors == [], f"JavaScript errors on page load: {errors}"


@pytest.fixture(scope="module")
def homepage_route_url(panel_extension, ui_base_dir):
    """Serve the homepage as a per-session route, the way ``guppy`` serves it.

    The shared ``live_server`` serves one prebuilt template, so it never builds a homepage
    inside a browser session.
    """
    port = scanPortsAndFind()
    server = pn.serve(
        {"/": lambda: build_homepage(input_root_folder=str(ui_base_dir))}, port=port, show=False, threaded=True
    )
    yield f"http://localhost:{port}"
    # Joining lets the session-wide kill_all_servers find this server already stopped.
    server.stop()
    server.join()


@pytest.mark.ui
def test_the_homepage_renders_on_every_load(browser, homepage_route_url):
    # Panel includes an extension's JavaScript in the first session's page on its own, but in
    # later sessions only when the extension is declared, so a reload is the case to check.
    for _ in range(2):
        page = browser.new_page()
        console_errors = []
        page.on(
            "console",
            lambda message, errors=console_errors: message.type == "error" and errors.append(message.text),
        )
        page.goto(homepage_route_url)
        expect(page.get_by_text("Parameter Selection").first).to_be_visible()
        assert console_errors == []
        page.close()

import re
import socket
import time

import panel as pn
import pytest
from playwright.sync_api import expect

from guppy.app import build_session_homepage
from guppy.frontend.frontend_utils import scanPortsAndFind


@pytest.mark.ui
def test_page_title_names_guppy(page, live_server_url):
    page.goto(live_server_url)
    expect(page.get_by_text("Parameter Selection").first).to_be_visible()
    assert page.title() == "GuPPy: Guided Photometry Analysis in Python"


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
def served_homepages():
    """The document and homepage of each browser session the homepage route has built, in order."""
    return []


@pytest.fixture(scope="module")
def homepage_route_url(panel_extension, ui_base_dir, served_homepages):
    """Serve the homepage as a per-session route, the way ``guppy`` serves it.

    The shared ``live_server`` serves one prebuilt template, so it never builds a homepage
    inside a browser session.
    """

    def build_page():
        template = build_session_homepage(input_root_folder=str(ui_base_dir))
        served_homepages.append((pn.state.curdoc, template))
        return template

    port = scanPortsAndFind()
    server = pn.serve({"/": build_page}, port=port, show=False, threaded=True)
    # The server thread starts listening a moment after pn.serve returns.
    for _ in range(50):
        try:
            connection = socket.create_connection(("localhost", port), timeout=0.1)
            connection.close()
            break
        except OSError:
            time.sleep(0.05)
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


@pytest.mark.ui
def test_refreshing_the_homepage_keeps_its_open_tabs(page, homepage_route_url, served_homepages):
    page.goto(homepage_route_url)
    expect(page.get_by_text("Parameter Selection").first).to_be_visible()
    expect(page).to_have_url(re.compile("bokeh-session-id="))
    document, template = served_homepages[-1]
    page_area = template._widgets["page_area"]
    # A sidebar button needs a selected session folder to open its page, so the page is opened
    # on the server, inside the session's document, the way a click's handler runs.
    document.add_next_tick_callback(
        lambda: page_area.show(key="open_visualization", name="Visualization", content=pn.pane.Markdown("dashboard"))
    )
    expect(page.locator(".bk-tab")).to_have_text(["Home", "Visualization"])
    built_before_refresh = len(served_homepages)

    page.reload()

    expect(page.get_by_text("Parameter Selection").first).to_be_visible()
    expect(page.locator(".bk-tab")).to_have_text(["Home", "Visualization"])
    assert len(served_homepages) == built_before_refresh

import panel as pn
import pytest

from guppy.orchestration import step_view
from guppy.orchestration.step_view import StepView, serve_page, serve_pages
from guppy.utils import page_links
from guppy.utils.page_links import Page, PageLink


class TestStepView:
    def test_open_registers_token_and_offers_a_link_to_the_view(self):
        view = StepView(route="my-view", title="Title", build_page=lambda sf, ip: pn.pane.Markdown("x"))

        with page_links.collect() as offers:
            view.open(["/a"], {"k": 1})

        (token,) = view.pending
        assert view.pending[token] == (["/a"], {"k": 1})
        assert offers.links == [PageLink(title="Title", href=f"my-view?token={token}")]

    def test_route_factory_expired_token_shows_notice(self, panel_extension, monkeypatch):
        view = StepView(route="my-view", title="Title", build_page=lambda sf, ip: pn.pane.Markdown("page"))
        monkeypatch.setattr(step_view, "_read_token", lambda: "bogus")

        template = view.route_factory()

        text = " ".join(str(o.object) for o in template.main if hasattr(o, "object"))
        assert "expired" in text.lower()

    def test_route_factory_valid_token_builds_page_from_context(self, panel_extension, monkeypatch):
        compose_calls = []
        view = StepView(
            route="my-view",
            title="Title",
            build_page=lambda sf, ip: compose_calls.append((sf, ip)) or pn.pane.Markdown("page"),
        )
        view.pending["tok"] = (["/a"], {"k": 1})
        monkeypatch.setattr(step_view, "_read_token", lambda: "tok")
        # No live session in the test, so stub the session-destroyed registration.
        monkeypatch.setattr(step_view.pn.state, "on_session_destroyed", lambda callback: None)

        view.route_factory()

        assert compose_calls == [(["/a"], {"k": 1})]


@pytest.fixture
def shown(monkeypatch):
    """Record how each page would be served; starting a real server is out of scope here."""
    served = []
    monkeypatch.setattr(
        pn.template.BootstrapTemplate, "show", lambda self, **kwargs: served.append((self.title, kwargs))
    )
    monkeypatch.setattr(step_view, "scanPortsAndFind", lambda start_port, end_port: 5123)
    return served


class TestServePage:
    def test_serves_without_opening_a_browser_and_offers_a_link_to_the_port(self, panel_extension, shown):
        with page_links.collect() as offers:
            serve_page(template=pn.template.BootstrapTemplate(title="GuPPy — Label Groups"), title="Label Groups")

        assert shown == [("GuPPy — Label Groups", {"port": 5123, "open": False})]
        assert offers.links == [PageLink(title="Label Groups", href="http://localhost:5123/")]


class TestServePages:
    def test_a_lone_page_is_served_as_it_is(self, panel_extension, shown):
        template = pn.template.BootstrapTemplate(title="Label Stores GUI - mouse_01")

        link = serve_pages(
            pages=[Page(name="mouse_01", template=template)], title="Label Stores", plural_noun="sessions"
        )

        assert shown == [("Label Stores GUI - mouse_01", {"port": 5123, "open": False})]
        assert link == PageLink(title="Label Stores — mouse_01", href="http://localhost:5123/")

    def test_several_pages_share_one_page_with_a_tab_each(self, panel_extension, shown):
        mouse_01 = pn.template.BootstrapTemplate(title="Label Stores GUI - mouse_01")
        mouse_01.main.append(pn.pane.Markdown("labels for mouse_01"))
        mouse_02 = pn.template.BootstrapTemplate(title="Label Stores GUI - mouse_02")
        mouse_02.main.append(pn.pane.Markdown("labels for mouse_02"))

        link = serve_pages(
            pages=[Page(name="mouse_01", template=mouse_01), Page(name="mouse_02", template=mouse_02)],
            title="Label Stores",
            plural_noun="sessions",
        )

        assert shown == [("Label Stores — 2 sessions", {"port": 5123, "open": False})]
        assert link == PageLink(title="Label Stores — 2 sessions", href="http://localhost:5123/")
        (tabs,) = mouse_01.main
        assert tabs._names == ["mouse_01", "mouse_02"]
        assert [column[0].object for column in tabs] == ["labels for mouse_01", "labels for mouse_02"]

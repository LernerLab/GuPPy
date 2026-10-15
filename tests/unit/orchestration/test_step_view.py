import panel as pn

from guppy.orchestration import step_view
from guppy.orchestration.step_view import (
    StepView,
    build_page_view,
    serve_page,
    serve_pages,
)
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


def served_template(link):
    """The template registered for ``link`` on the shared ``page`` route."""
    assert link.href.startswith("page?token=")
    return step_view._pending_pages[link.href.removeprefix("page?token=")]


class TestServePage:
    def test_registers_the_page_on_the_page_route_and_offers_a_link(self, panel_extension):
        template = pn.template.BootstrapTemplate(title="GuPPy — Label Groups")

        with page_links.collect() as offers:
            serve_page(template=template, title="Label Groups")

        (link,) = offers.links
        assert link.title == "Label Groups"
        assert served_template(link) is template


class TestServePages:
    def test_a_lone_page_is_served_as_it_is(self, panel_extension):
        template = pn.template.BootstrapTemplate(title="Label Stores GUI - mouse_01")

        link = serve_pages(
            pages=[Page(name="mouse_01", template=template)], title="Label Stores", plural_noun="sessions"
        )

        assert link.title == "Label Stores — mouse_01"
        assert served_template(link) is template

    def test_several_pages_share_one_page_with_a_tab_each(self, panel_extension):
        mouse_01 = pn.template.BootstrapTemplate(title="Label Stores GUI - mouse_01")
        mouse_01.main.append(pn.pane.Markdown("labels for mouse_01"))
        mouse_02 = pn.template.BootstrapTemplate(title="Label Stores GUI - mouse_02")
        mouse_02.main.append(pn.pane.Markdown("labels for mouse_02"))

        link = serve_pages(
            pages=[Page(name="mouse_01", template=mouse_01), Page(name="mouse_02", template=mouse_02)],
            title="Label Stores",
            plural_noun="sessions",
        )

        assert link.title == "Label Stores — 2 sessions"
        combined = served_template(link)
        assert combined.title == "Label Stores — 2 sessions"
        (tabs,) = combined.main
        assert tabs._names == ["mouse_01", "mouse_02"]
        assert [column[0].object for column in tabs] == ["labels for mouse_01", "labels for mouse_02"]


class TestBuildPageView:
    def test_serves_the_page_registered_for_the_token(self, panel_extension, monkeypatch):
        template = pn.template.BootstrapTemplate(title="Page")
        monkeypatch.setitem(step_view._pending_pages, "tok", template)
        monkeypatch.setattr(step_view, "_read_token", lambda: "tok")
        # No live session in the test, so stub the session-destroyed registration.
        monkeypatch.setattr(step_view.pn.state, "on_session_destroyed", lambda callback: None)

        assert build_page_view() is template

    def test_unknown_token_shows_expired_notice(self, panel_extension, monkeypatch):
        monkeypatch.setattr(step_view, "_read_token", lambda: "bogus")

        template = build_page_view()

        assert [pane.object for pane in template.main] == ["This view has expired. You can close this tab."]

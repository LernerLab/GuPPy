import panel as pn

from guppy.orchestration import step_view
from guppy.orchestration.step_view import StepView


class TestStepView:
    def test_open_registers_token_and_opens_view_url(self, monkeypatch):
        view = StepView(route="my-view", title="Title", build_page=lambda sf, ip: pn.pane.Markdown("x"))
        monkeypatch.setattr(step_view, "_current_href", lambda: "http://localhost:5006/")
        opened = []
        monkeypatch.setattr(step_view.webbrowser, "open", lambda url: opened.append(url))

        view.open(["/a"], {"k": 1})

        assert len(view.pending) == 1
        token = next(iter(view.pending))
        assert view.pending[token] == (["/a"], {"k": 1})
        assert opened == [f"http://localhost:5006/my-view?token={token}"]

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

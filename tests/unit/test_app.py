import panel as pn
import pytest

from guppy import app


@pytest.fixture
def served(monkeypatch):
    """Capture the route map instead of starting a real Bokeh server.

    ``pn.serve`` binds a port and opens a browser tab, so it is stubbed here; everything
    downstream of it (the route factories themselves) runs for real.
    """
    captured = {}
    monkeypatch.setattr(app.pn, "serve", lambda routes, **kwargs: captured.update(routes=routes, kwargs=kwargs))
    return captured


class TestServeApp:
    def test_serves_every_route_and_opens_the_homepage(self, served):
        app.serve_app()

        assert sorted(served["routes"]) == [
            "/",
            "/artifact-view",
            "/preprocess-view",
            "/select-artifact-windows",
            "/transients-view",
        ]
        assert served["kwargs"] == {"show": True}

    def test_view_routes_are_the_step_view_factories(self, served):
        app.serve_app()

        assert served["routes"]["/preprocess-view"] is app.build_preprocess_view
        assert served["routes"]["/transients-view"] is app.build_transients_view

    def test_homepage_route_roots_the_file_selector_at_start_path(self, served, panel_extension, tmp_path):
        """The ``/`` route is a closure over ``start_path``; invoking it must build a
        homepage whose session-folder selector opens at that directory."""
        app.serve_app(start_path=str(tmp_path))
        template = served["routes"]["/"]()

        assert isinstance(template, pn.template.BootstrapTemplate)
        assert template._widgets["files_1"].directory == str(tmp_path)

    def test_homepage_route_falls_back_to_the_default_root(self, served, panel_extension, tmp_path):
        """With no start_path the selector falls back to the default root rather than
        inheriting a stale directory."""
        app.serve_app()
        template = served["routes"]["/"]()

        assert template._widgets["files_1"].directory != str(tmp_path)

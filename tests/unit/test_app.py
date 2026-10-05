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
            "/tonic-analysis",
            "/transients-view",
        ]
        assert served["kwargs"] == {"show": True}

    def test_view_routes_are_the_step_view_factories(self, served):
        app.serve_app()

        assert served["routes"]["/preprocess-view"] is app.build_preprocess_view
        assert served["routes"]["/transients-view"] is app.build_transients_view

    def test_homepage_route_opens_the_session_browser_in_the_input_root(self, served, panel_extension, tmp_path):
        """The ``/`` route is a closure over the roots; invoking it must build a homepage
        whose session-folder browser opens inside the input root folder."""
        input_root_folder = tmp_path / "raw"
        input_root_folder.mkdir()

        app.serve_app(input_root_folder=str(input_root_folder))
        template = served["routes"]["/"]()

        assert isinstance(template, pn.template.BootstrapTemplate)
        assert template._widgets["files_1"].directory == str(input_root_folder)

    def test_homepage_route_falls_back_to_the_default_root(self, served, panel_extension, tmp_path):
        """With no input root the browser falls back to the default root rather than
        inheriting a stale directory."""
        app.serve_app()
        template = served["routes"]["/"]()

        assert template._widgets["files_1"].directory != str(tmp_path)

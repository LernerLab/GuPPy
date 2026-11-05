import panel as pn
import pytest

from guppy import app


@pytest.fixture
def served(monkeypatch):
    """Capture what would be served instead of starting a real Bokeh server.

    ``pn.serve`` binds a port and opens a browser tab, so it is stubbed here; everything
    downstream of it (the homepage factory itself) runs for real.
    """
    captured = {}
    monkeypatch.setattr(
        app.pn, "serve", lambda build_page, **kwargs: captured.update(build_page=build_page, kwargs=kwargs)
    )
    return captured


class TestServeApp:
    def test_serves_the_homepage_alone(self, served):
        app.serve_app()

        assert callable(served["build_page"])
        assert served["kwargs"] == {"port": 0, "address": None, "show": True, "verbose": True}

    def test_prints_how_to_open_and_stop_guppy(self, served, capsys):
        app.serve_app()

        assert capsys.readouterr().out == (
            "GuPPy is starting. Open the address below in your browser, and press Ctrl+C here to stop GuPPy.\n"
        )

    def test_homepage_opens_the_session_browser_in_the_input_root(self, served, panel_extension, tmp_path):
        """The served page is a closure over the roots; invoking it must build a homepage
        whose session-folder browser opens inside the input root folder."""
        input_root_folder = tmp_path / "raw"
        input_root_folder.mkdir()

        app.serve_app(input_root_folder=str(input_root_folder))
        template = served["build_page"]()

        assert isinstance(template, pn.template.BootstrapTemplate)
        assert template._widgets["files_1"].directory == str(input_root_folder)

    def test_homepage_falls_back_to_the_default_root(self, served, panel_extension, tmp_path):
        """With no input root the browser falls back to the default root rather than
        inheriting a stale directory."""
        app.serve_app()
        template = served["build_page"]()

        assert template._widgets["files_1"].directory != str(tmp_path)

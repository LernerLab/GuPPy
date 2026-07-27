import pytest

from guppy import app
from guppy import main as main_module
from guppy.main import main


@pytest.fixture
def served(monkeypatch):
    """Capture the route map instead of starting a real Bokeh server.

    Stubbing ``pn.serve`` (rather than ``serve_app``) lets ``main`` dispatch through the
    real application wiring, so the deferred ``from .app import serve_app`` is exercised.
    """
    captured = {}
    monkeypatch.setattr(app.pn, "serve", lambda routes, **kwargs: captured.update(routes=routes, kwargs=kwargs))
    return captured


@pytest.fixture
def exported(monkeypatch):
    """Record log exports instead of copying onto the user's real Desktop."""
    calls = []
    monkeypatch.setattr(main_module.logging_config, "export_log_file", lambda: calls.append(True))
    return calls


class TestMain:
    def test_no_arguments_serves_the_app(self, served, exported):
        main(argv=[])

        assert sorted(served["routes"]) == ["/", "/preprocess-view", "/transients-view"]
        assert exported == []

    def test_start_path_reaches_the_homepage_route(self, served, exported, panel_extension, tmp_path):
        main(argv=["--start-path", str(tmp_path)])
        template = served["routes"]["/"]()

        assert template._widgets["files_1"].directory == str(tmp_path)
        assert exported == []

    def test_export_logs_exports_without_starting_a_server(self, served, exported):
        main(argv=["--export-logs"])

        assert exported == [True]
        assert served == {}

    def test_unrecognized_argument_exits(self, served, exported):
        with pytest.raises(SystemExit):
            main(argv=["--not-a-flag"])

        assert served == {}
        assert exported == []

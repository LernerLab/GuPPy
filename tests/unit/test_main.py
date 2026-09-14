from importlib.metadata import version

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

        assert sorted(served["routes"]) == [
            "/",
            "/artifact-view",
            "/preprocess-view",
            "/select-artifact-windows",
            "/tonic-analysis",
            "/transients-view",
        ]
        assert exported == []

    def test_the_roots_reach_the_homepage_route(self, served, exported, panel_extension, tmp_path):
        input_root_folder = tmp_path / "data"
        output_directory = tmp_path / "derivatives"
        input_root_folder.mkdir()
        output_directory.mkdir()

        main(argv=["--input-root", str(input_root_folder), "--output-root", str(output_directory)])
        template = served["routes"]["/"]()

        # The visible "Selected files" pane, not just the parameter: assigning value alone
        # leaves the browser showing nothing until it re-lists its directory.
        assert template._widgets["input_root_selector"].value == [str(input_root_folder)]
        assert list(template._widgets["input_root_selector"]._selector.value) == [str(input_root_folder)]
        assert template._widgets["output_root_selector"].value == [str(output_directory)]
        assert list(template._widgets["output_root_selector"]._selector.value) == [str(output_directory)]

    def test_an_input_root_folder_that_does_not_exist_is_left_unselected(
        self, served, exported, panel_extension, tmp_path
    ):
        main(argv=["--input-root", str(tmp_path / "missing")])
        template = served["routes"]["/"]()

        assert template._widgets["input_root_selector"].value == []
        assert list(template._widgets["input_root_selector"]._selector.value) == []

    def test_export_logs_exports_without_starting_a_server(self, served, exported):
        main(argv=["--export-logs"])

        assert exported == [True]
        assert served == {}

    def test_version_prints_the_installed_version_without_starting_a_server(self, served, exported, capsys):
        with pytest.raises(SystemExit) as exit_info:
            main(argv=["--version"])

        assert exit_info.value.code == 0
        assert capsys.readouterr().out.strip() == f"GuPPy {version('guppy-neuro')}"
        assert served == {}
        assert exported == []

    def test_unrecognized_argument_exits(self, served, exported):
        with pytest.raises(SystemExit):
            main(argv=["--not-a-flag"])

        assert served == {}
        assert exported == []

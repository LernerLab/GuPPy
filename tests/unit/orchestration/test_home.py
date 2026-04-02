import panel as pn
import pytest

from guppy.orchestration.home import build_homepage


@pytest.fixture
def homepage(panel_extension):
    """Build a fresh homepage template for each test."""
    return build_homepage()


def test_returns_bootstrap_template(homepage):
    assert isinstance(homepage, pn.template.BootstrapTemplate)


def test_hooks_contains_onclick_process(homepage):
    assert "onclickProcess" in homepage._hooks
    assert callable(homepage._hooks["onclickProcess"])


def test_hooks_contains_get_input_parameters(homepage):
    assert "getInputParameters" in homepage._hooks
    assert callable(homepage._hooks["getInputParameters"])


def test_widgets_contains_files_1(homepage):
    assert "files_1" in homepage._widgets
    assert hasattr(homepage._widgets["files_1"], "value")


def test_onclick_process_calls_save_parameters(homepage, tmp_path, monkeypatch):
    folder = tmp_path / "session1"
    folder.mkdir()
    homepage._widgets["files_1"].value = [str(folder)]

    calls = []
    monkeypatch.setattr(
        "guppy.orchestration.home.save_parameters",
        lambda inputParameters: calls.append(inputParameters),
    )
    homepage._hooks["onclickProcess"]()
    assert len(calls) == 1
    assert isinstance(calls[0], dict)


def test_get_input_parameters_returns_dict(homepage, tmp_path):
    folder = tmp_path / "session1"
    folder.mkdir()
    homepage._widgets["files_1"].value = [str(folder)]

    result = homepage._hooks["getInputParameters"]()
    assert isinstance(result, dict)

import os

import panel as pn
import pytest


@pytest.fixture(scope="session")
def panel_extension():
    """Call pn.extension() exactly once for the entire test session.

    Panel requires this before any widget instantiation.
    """
    pn.extension()


@pytest.fixture(scope="session")
def frontend_base_dir(tmp_path_factory):
    """Create a real temp directory and point GUPPY_BASE_DIR at it.

    Ensures FileSelector resolves to a real path and prevents Tk dialogs.
    Restores the original value on teardown.
    """
    base_dir = tmp_path_factory.mktemp("frontend_base")
    original = os.environ.get("GUPPY_BASE_DIR")
    os.environ["GUPPY_BASE_DIR"] = str(base_dir)
    yield base_dir
    if original is None:
        del os.environ["GUPPY_BASE_DIR"]
    else:
        os.environ["GUPPY_BASE_DIR"] = original


@pytest.fixture
def parameter_form(panel_extension, frontend_base_dir, tmp_path):
    """Build a BootstrapTemplate + ParameterForm backed by a real temp directory.

    Sets files_1.value to a list of session paths under tmp_path so that
    getInputParameters() can be called without raising.
    """
    from guppy.frontend.input_parameters import ParameterForm

    session_dir = tmp_path / "session1"
    session_dir.mkdir()
    template = pn.template.BootstrapTemplate(title="Test")
    form = ParameterForm(template=template)
    form.files_1.value = [str(session_dir)]
    return form


@pytest.fixture
def sidebar(panel_extension):
    """Build a BootstrapTemplate + Sidebar."""
    from guppy.frontend.sidebar import Sidebar

    template = pn.template.BootstrapTemplate(title="Test")
    return Sidebar(template=template)

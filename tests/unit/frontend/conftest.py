import panel as pn
import pytest


@pytest.fixture(scope="session")
def panel_extension():
    """Call pn.extension() exactly once for the entire test session.

    Panel requires this before any widget instantiation.
    """
    pn.extension()

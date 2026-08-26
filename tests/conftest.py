import multiprocessing
import os
from pathlib import Path

import holoviews as hv
import panel as pn
import pytest

# Use "spawn" start method for all multiprocessing in tests. "fork" (the Linux
# default) can deadlock when forking a multi-threaded pytest host, and can stall
# coverage measurement waiting on child-process signals. "spawn" creates a clean
# interpreter for each worker, which is safe in all environments. Windows always
# uses "spawn" so force=True is a no-op there; macOS/Linux benefit from it.
multiprocessing.set_start_method("spawn", force=True)

PYPROJECT_PATH = Path(__file__).parent.parent / "pyproject.toml"


def pytest_configure(config: pytest.Config) -> None:
    """Enable coverage measurement inside subprocesses that tests spawn.

    coverage's startup hook is installed as a ``.pth`` file, but it is inert unless
    ``COVERAGE_PROCESS_START`` names a config file. pytest-cov does not set it, so a plain
    ``python -c`` child (see ``tests/integration/test_integration_ndx_events_import_state.py``)
    is measured as if it never ran. Children inherit this environment, so setting it here is
    enough — no test needs to pass ``env=`` explicitly.

    Only set when ``--cov`` was requested: otherwise every subprocess in the suite would start
    coverage and leave a stray ``.coverage.*`` data file behind on an ordinary test run.
    """
    if getattr(config.option, "cov_source", None):
        os.environ.setdefault("COVERAGE_PROCESS_START", str(PYPROJECT_PATH))


@pytest.fixture(scope="session")
def panel_extension() -> None:
    """Load the Panel and Holoviews rendering extensions exactly once for the session.

    Panel requires ``pn.extension()`` before any widget instantiation. Holoviews needs a
    plotting backend registered (``hv.extension("bokeh")``) before any ``opts.NdOverlay`` call
    the dashboard/plotter widgets make; registering it here — rather than relying on it being
    loaded implicitly by some earlier test — keeps frontend tests independent of run order.
    """
    pn.extension()
    hv.extension("bokeh")

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


@pytest.fixture(scope="session", autouse=True)
def _stores_cache_home_patch(tmp_path_factory: pytest.TempPathFactory):
    """Redirect the Step-1 store-labels cache (``~/.storesList.json``) away from the real home.

    Session-scoped so it is live before the session-scoped ``step1_output_*`` integration
    fixtures drive the store-labeling save, which writes the cache via
    ``store_label_cache_path()``.
    """
    home = tmp_path_factory.mktemp("stores_cache_home")
    patcher = pytest.MonkeyPatch()
    patcher.setattr("guppy.orchestration.store_labeling.store_label_cache_path", lambda: home / ".storesList.json")
    yield
    patcher.undo()


@pytest.fixture(autouse=True)
def isolated_stores_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Give each test its own store-labels cache directory so state never leaks across tests."""
    monkeypatch.setattr(
        "guppy.orchestration.store_labeling.store_label_cache_path", lambda: tmp_path / ".storesList.json"
    )
    return tmp_path


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

import json
import multiprocessing
import os
import sys
from pathlib import Path

import panel as pn
import pytest

# Set GUPPY_BASE_DIR before any guppy source modules are imported. This is
# GuPPy's headless flag: orchestration code checks it to skip figure creation.
# The per-test step functions will override this with the actual base_dir.
os.environ.setdefault("GUPPY_BASE_DIR", "1")

# Use "spawn" start method for all multiprocessing in tests. "fork" (the Linux
# default) can deadlock when forking a multi-threaded pytest host, and can stall
# coverage measurement waiting on child-process signals. "spawn" creates a clean
# interpreter for each worker, which is safe in all environments. Windows always
# uses "spawn" so force=True is a no-op there; macOS/Linux benefit from it.
multiprocessing.set_start_method("spawn", force=True)

# Ensure the 'src' directory and 'tests/' itself are on sys.path so that
# test files can do `from conftest import ...` regardless of their nesting depth.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TESTS_ROOT = os.path.dirname(__file__)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
for path in (SRC_PATH, TESTS_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

# Static data directories — import these in test modules instead of redefining locally
STUBBED_TESTING_DATA = Path(PROJECT_ROOT) / "stubbed_testing_data"
TESTING_DATA = Path(PROJECT_ROOT) / "testing_data"


def event_ts_offset_for(base_dir: str | Path) -> float:
    """Return the ``timeForLightsTurnOn`` used in a pipeline run under ``base_dir``.

    This is the constant by which current event timestamps and PSTH event-time labels
    shifted relative to the (older, lights-on-basis) reference outputs (issue #355).
    Pass it as ``event_ts_offset`` to :func:`compare_output_folders` so the consistency
    suite tolerates exactly that known shift while comparing all other data exactly.
    """
    matches = sorted(Path(base_dir).rglob("GuPPyParamtersUsed.json"))
    assert matches, f"No GuPPyParamtersUsed.json found under {base_dir}"
    with open(matches[0]) as params_file:
        return json.load(params_file)["timeForLightsTurnOn"]


@pytest.fixture(scope="session")
def panel_extension() -> None:
    """Call pn.extension() exactly once for the entire test session.

    Panel requires this before any widget instantiation.
    """
    pn.extension()

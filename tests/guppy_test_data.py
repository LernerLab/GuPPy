"""Shared, import-safe test-data locations and helpers.

This module holds constants and helpers that test modules need at *collection* time
(e.g. inside ``@pytest.mark.parametrize`` decorators and module-level case lists), so it must
stay a plain importable module with no pytest or heavy imports. It lives under a unique name
(rather than in ``conftest.py``) so ``from guppy_test_data import ...`` resolves the same way from
every invocation directory, instead of colliding with the several ``conftest.py`` files in the tree.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Static data directories — import these in test modules instead of redefining locally
STUBBED_TESTING_DATA = PROJECT_ROOT / "stubbed_testing_data"
TESTING_DATA = PROJECT_ROOT / "testing_data"


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

"""Shared, import-safe test-data locations and helpers.

This module holds constants and helpers that test modules need at *collection* time
(e.g. inside ``@pytest.mark.parametrize`` decorators and module-level case lists), so it must
stay a plain importable module with no pytest or heavy imports. It lives under a unique name
(rather than in ``conftest.py``) so ``from guppy_test_data import ...`` resolves the same way from
every invocation directory, instead of colliding with the several ``conftest.py`` files in the tree.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import h5py

if TYPE_CHECKING:
    import holoviews as hv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Static data directories — import these in test modules instead of redefining locally
STUBBED_TESTING_DATA = PROJECT_ROOT / "stubbed_testing_data"
TESTING_DATA = PROJECT_ROOT / "testing_data"


def event_ts_offset_for(base_dir: str | Path) -> float:
    """Return the absolute lights-on instant of a pipeline run under ``base_dir``.

    This is the constant by which current event timestamps and PSTH event-time labels
    shifted relative to the (older, lights-on-basis) reference outputs (issue #355).
    Pass it as ``event_ts_offset`` to :func:`compare_output_folders` so the consistency
    suite tolerates exactly that known shift while comparing all other data exactly.

    The instant is ``recordingStart + timeForLightsTurnOn``: the run's warm-up is measured
    from the recording's own clock, which does not necessarily start at 0.
    """
    matches = sorted(Path(base_dir).rglob("GuPPyParamtersUsed.json"))
    assert matches, f"No GuPPyParamtersUsed.json found under {base_dir}"
    with open(matches[0]) as params_file:
        time_for_lights_turn_on = json.load(params_file)["timeForLightsTurnOn"]

    return recording_start_for(base_dir) + time_for_lights_turn_on


def recording_start_for(base_dir: str | Path) -> float:
    """Return the recording start of a pipeline run under ``base_dir``.

    This is the constant by which current continuous timestamps shifted relative to
    reference outputs generated when the extractor re-zeroed its clock (issue #407).
    Pass it as ``continuous_ts_offset`` to :func:`compare_output_folders`.
    """
    time_correction_files = sorted(Path(base_dir).rglob("timeCorrection_*.hdf5"))
    assert time_correction_files, f"No timeCorrection_*.hdf5 found under {base_dir}"
    with h5py.File(time_correction_files[0], "r") as time_correction_file:
        return float(time_correction_file["recordingStart"][0])


def resolve_plot(plot: "hv.DynamicMap") -> "hv.Element":
    """Materialize the element a trace-plot builder's ``DynamicMap`` currently displays.

    The visualization builders wrap their curves in :func:`~guppy.visualization.shading.shade_trace`,
    so they return a range-linked ``DynamicMap`` rather than a bare element. Indexing it with
    the empty key resolves it against the full data, which is what assertions inspect.
    """
    return plot[()]

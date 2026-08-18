# Design decisions

Architecture explains where things live. This page explains why some of the less obvious choices
were made the way they were, and where the deliberate seams are if you need to add a new step, a
new result view, or a new validation check.

## HDF5 as the inter-step medium

Each pipeline step reads its input from files the previous step wrote to disk, rather than passing
data through memory. The practical payoff is that a step is independently re-runnable: its entire
input is the on-disk output of the step before it, so you can close the GUI after Step 2 and reopen
it days later to run Step 3 — there is no in-memory session to lose.

[`utils/_hdf5_io.py`](https://github.com/LernerLab/GuPPy/blob/main/src/guppy/utils/_hdf5_io.py)
holds the two functions every step uses to talk to these files: `read_hdf5` and `write_hdf5`. Array
datasets are created with `maxshape=(None,)` and `chunks=True`, so a store's dataset can be resized
and overwritten in place on a later run rather than requiring the whole file to be rewritten. No
HDF5 attributes are used anywhere — every piece of metadata, down to a store's own name, is a
dataset like any other.

One naming detail worth knowing up front: Step 2's raw HDF5 output is keyed by store *id* (the name
the acquisition system used), while Step 3's processed output is keyed by store *label* (the name
you assigned in Label Stores). See [Output data model](../reference/outputs.md) for the full
on-disk layout.

## Why a background thread, not the Panel IOLoop

Each pipeline step's worker runs on a `threading.Thread`, started from
`_run_worker_with_progress` in
[`orchestration/home.py`](https://github.com/LernerLab/GuPPy/blob/main/src/guppy/orchestration/home.py).
The alternative — calling the worker synchronously from the button's click handler — would block
the Panel server's IOLoop for the whole step. Bokeh's browser tab depends on that IOLoop answering
websocket heartbeats; block it and the tab drops its connection and never recovers, even after the
step finishes.

Instead, the handler starts the worker thread and returns immediately. A `pn.state.add_periodic_callback`
polls the step's progress every 100ms from the IOLoop and updates the sidebar's progress bar, so the
UI stays responsive for the whole duration of a step.

## Why `spawn`, not `fork`, for multiprocessing

Every pipeline step that parallelizes work across CPU cores — reading raw data, computing PSTHs,
finding transients — uses `multiprocessing.get_context("spawn")` rather than the platform default.
[`orchestration/read_raw_data.py`](https://github.com/LernerLab/GuPPy/blob/main/src/guppy/orchestration/read_raw_data.py)
spells out why: the worker runs on a background thread of the Panel server process, which also
carries the Tornado IOLoop threads and a bound Bokeh socket. Forking that process would copy only
the calling thread into the child, so any lock another thread happened to be holding at that instant
— a logging lock, an HDF5 file lock — would stay locked forever in the fork, with no thread left to
release it. `spawn` starts a fresh interpreter instead, sidestepping the problem entirely. The same
pattern recurs in `orchestration/psth.py`, `orchestration/transients.py`, and
`analysis/transients.py`.

`numberOfCores` from `inputParameters` controls the pool size: `0` means "use every core"
(`mp.cpu_count()`); a value above `cpu_count()` is clamped to `cpu_count() - 1` with a logged
warning; and `<= 1` skips the pool entirely, running the tasks serially in the parent process
instead.

## Progress reporting is pull-based

[`utils/progress.py`](https://github.com/LernerLab/GuPPy/blob/main/src/guppy/utils/progress.py) is
modeled on the standard library's `logging` module. A `StepProgress` instance is bound to a
`ContextVar` for the duration of one step run — `home.py` is the only thing that ever performs that
binding. Deep analysis code reports progress by calling the module-level `progress.start()`,
`progress.advance()`, `progress.track()`, and `progress.fail()`, with no progress object threaded
through any function signature. When nothing is bound, those calls are a silent no-op.

That no-op behavior is what lets the headless testing API call the same worker functions with no
GUI attached at all — see [Testing](../contributing/testing.md) for how the test suite exercises
the pipeline this way.

Completion is defined as "the worker thread finished," never "the progress counter reached its
declared total." A step whose total undercounts its own work would otherwise report success while
its output is still partway written, so the poller in `home.py` only reports success once
`thread.is_alive()` is `False`.

## Two server-lifetime strategies, both intentional

`StepView`, in
[`orchestration/step_view.py`](https://github.com/LernerLab/GuPPy/blob/main/src/guppy/orchestration/step_view.py),
never tears anything down. A step's result view (preprocess, transients) is a route on the one
long-running main Panel server, addressed by a `uuid4().hex` token stashed in a `pending` dict when
the view is opened, and cleaned up via `pn.state.on_session_destroyed` once the browser tab closes.
Because the server backing the route is never restarted, the result view never shows Bokeh's
"server connection lost" banner.

`store_labeling.py`'s page-serving function takes the opposite approach: it finds a free port with
`scanPortsAndFind` and calls `template.show(port=...)`, spinning up a fresh, separate, short-lived
server dedicated to that one session folder's Label Stores page.

These are two different, deliberate strategies for two different situations, not one superseding
the other. If you add a new result view, pick whichever fits the view you're building rather than
copying whichever pattern you happened to read first.

## The `inputParameters` dict contract

Every orchestration worker takes a single flat `dict`, produced by
`ParameterForm.getInputParameters()` in
[`frontend/input_parameters.py`](https://github.com/LernerLab/GuPPy/blob/main/src/guppy/frontend/input_parameters.py).
Its keys mix camelCase (`noChannels`, `nSecPrev`) and snake_case (`session_folders`,
`zscore_method`) with no consistent rule between them. This is a known wart, not a convention — new
keys should not imitate it.

There are three overlapping but different subsets of this dict in play at once:

- What the form actually produces from `getInputParameters()`.
- What `orchestration/save_parameters.py` snapshots into `GuPPyParamtersUsed.json` — a hardcoded
  list of keys that deliberately excludes things like `session_folders`, `numberOfCores`, and
  `mode`, because those describe *how this run was invoked*, not the analysis configuration that
  produced the output.
- What the headless testing API layer injects on top before calling the real worker — keys like
  `store_id_to_store_label`, `run_name`, and the NPM decomposition parameters, which only exist in
  a headless context and are never part of the GUI form.

NPM's decomposition parameters (`npm_split_events`, `npm_time_unit`, `npm_timestamp_column_name`)
are a special case: they are written to a `.npm_params.json` sidecar file next to `storesList.csv`
(see `NPM_PARAMS_FILENAME` in
[`utils/utils.py`](https://github.com/LernerLab/GuPPy/blob/main/src/guppy/utils/utils.py)) instead
of living in `GuPPyParamtersUsed.json`. Step 2 has to reproduce Step 1's in-memory decomposition
identically, so these parameters are pinned to the run folder they were chosen for, rather than
living in the form state that a user could change before running the next step.

## The headless testing API's design principle

`step1()` and its siblings in
[`testing/api.py`](https://github.com/LernerLab/GuPPy/blob/main/src/guppy/testing/api.py) drive the
real production call chain rather than reimplementing it: set `GUPPY_BASE_DIR`, call
`build_homepage()`, set `template._widgets["files_1"].value` and call
`template._hooks["getInputParameters"]()` exactly as a click on the folder selector and a call to
the form would, inject the handful of headless-only keys, then call the real orchestration worker
(`orchestrate_store_labeling_page`, and its counterparts for the other steps). This is the whole
reason `home.py` bothers exposing `_hooks` and `_widgets` at all — a test exercises the identical
code path a user's click does, rather than a parallel path that could drift from it.

## Validation layering, the parts architecture.md doesn't say

[`utils/validation.py`](https://github.com/LernerLab/GuPPy/blob/main/src/guppy/utils/validation.py)'s
module docstring spells out three conventions worth knowing before adding a check:

- Its helpers raise `ValueError` specifically, never a bare `Exception`. `home.py` catches exactly
  that type from parameter parsing and step orchestration and surfaces it to the user via
  `pn.state.notifications.error(str(e), duration=0)`. Raising anything else means the failure looks
  like a genuine bug instead of a fixable input problem.
- A check is promoted into this shared module only once it is actually reused across call sites —
  not the first time it comes up. One-off checks stay inline where they are used.
- Every message follows the same template: name the offending value, state the rule, tell the user
  the valid range or the fix. `validate_window_bounds` produces messages like
  `"windowStart=-1 is before the signal start 0s; signal timespan is [0, 10]s — choose values
  within this range."`

## Run folders as the unit of output

A single session folder can hold several run folders — `<session>_output_<run_name>` directories
from separate Label Stores runs — and most steps operate on whichever subset `selected_runs`
identifies. `discover_run_folders` and `select_run_folders` in
[`utils/utils.py`](https://github.com/LernerLab/GuPPy/blob/main/src/guppy/utils/utils.py) do the
listing and filtering: `discover_run_folders` returns every run folder under a session, and
`select_run_folders` narrows that to the run names a step was actually asked to touch, raising a
`ValueError` that lists the available run names if one requested is missing.

One easy-to-overlook piece of global state sits outside any session or run folder entirely:
`~/.storesList.json`, a cache in the user's home directory that maps every store id ever labeled to
the label it was given, used to pre-fill the Label Stores dropdowns on future sessions. It persists
across sessions and projects, and is not part of any run's output.

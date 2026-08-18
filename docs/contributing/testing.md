# Testing

GuPPy's test suite lives under `tests/`, split into four subdirectories that trade off speed
against how close they get to a real recording. Knowing which one a change belongs in — and which
command to run while iterating on it — saves you from waiting on the slow ones.

## Test layout

| Directory | What it exercises | Data it uses |
|---|---|---|
| `tests/unit/` | Individual functions and classes, no I/O | none |
| `tests/integration/` | End-to-end step1–step5 pipeline runs | `stubbed_testing_data/` |
| `tests/consistency/` | Regression checks against v1.3.0 reference outputs | `testing_data/` |
| `tests/UI/` | The Panel web UI, driven by Playwright | `stubbed_testing_data/` |

`tests/unit/` mirrors the packages under `src/guppy/`, with one subpackage per test subpackage:
`analysis/`, `extractors/`, `frontend/`, `orchestration/`, `testing/`, `utils/`, and
`visualization/`. When you add a function to `src/guppy/analysis/`, its test belongs in
`tests/unit/analysis/`, and so on.

## Running a single test

```bash
pytest tests/unit/analysis/test_z_score.py -v
```

The same pattern works for a single test function with `::test_name` appended, or for an entire
directory in place of a file.

## Markers

Four of the five registered markers gate tests that need something beyond a plain `pytest`
invocation:

- **`full_data`** — needs the full `testing_data/` download, not present in a normal checkout.
  Deselect with `-m "not full_data"`.
- **`parallel`** — verifies GuPPy's multiprocessing behavior. Run in its own invocation so it does
  not race the rest of the suite.
- **`ui`** — needs a live Panel server and a browser (`playwright install chromium`). Deselect with
  `-m "not ui"`.
- **`dandi_live`** — streams from the real DANDI Archive over the network. Opt in explicitly with
  `-m dandi_live`; never run it as part of a broader selection.
- **`progress_bar`** — exercises the progress-bar file-locking loop. Currently applied to no tests,
  and skipped on Windows in CI when it is.

Markers are declared in
[`pyproject.toml`](https://github.com/LernerLab/GuPPy/blob/main/pyproject.toml) under
`[tool.pytest.ini_options]`.

## Test data

`stubbed_testing_data/` and `testing_data/` sound similar but serve different purposes:

- **`stubbed_testing_data/`** is small (tens of MB) and committed to the repository via Git LFS. A
  normal LFS-aware clone pulls it down automatically; otherwise run `git lfs install` followed by
  `git lfs pull`. Each session is documented in
  [`stubbed_testing_data/README.md`](https://github.com/LernerLab/GuPPy/blob/main/stubbed_testing_data/README.md).
  This is what `tests/integration/` and most of `tests/unit/extractors/` read from.
- **`testing_data/`** is large (single-digit GB), gitignored, and pulled from a shared Google Drive
  folder via rclone in CI. Most contributors will not have it locally. A `full_data` test does not
  skip when the directory is missing — it hard-fails on an assertion — so deselect `full_data`
  explicitly rather than relying on a skip.

## A command for day-to-day use

```bash
pytest tests -m "not full_data and not dandi_live and not ui" -v
```

This runs everything a typical contributor can run without the full data download or a browser.
UI tests need `playwright install chromium` first, and CI runs them in their own invocation so the
live Panel server's global state never collides with the headless `build_homepage()` calls the rest
of the suite makes.

## Continuous integration

[`pr-tests.yml`](https://github.com/LernerLab/GuPPy/blob/main/.github/workflows/pr-tests.yml) runs
on every pull request with `full_data` tests skipped. A required
`detect-changelog-updates` job means any PR touching `src/`, `tests/`, `pyproject.toml`, or
`.github/` must also update `CHANGELOG.md`, or CI fails.

[`dailies.yml`](https://github.com/LernerLab/GuPPy/blob/main/.github/workflows/dailies.yml) runs
the full OS-by-Python-version matrix overnight, including `full_data` tests. `dandi_live` tests are
excluded from every pytest invocation in both workflows, so they never run automatically — only a
contributor running `-m dandi_live` locally exercises them.

## The headless testing pattern

Most integration and orchestration tests never touch a browser. They set the `GUPPY_BASE_DIR`
environment variable to bypass the Tk folder dialogs, call `build_homepage()` to assemble the Panel
template, and then drive the pipeline through `step1()` through `step5()` from
[`guppy.testing.api`](https://github.com/LernerLab/GuPPy/blob/main/src/guppy/testing/api.py). This
mirrors the production call chain, so a test exercises the same code path a click does.

Extractor tests follow their own shared-mixin pattern; see
[Adding a new acquisition format](new_recording_format.md) for that walkthrough.

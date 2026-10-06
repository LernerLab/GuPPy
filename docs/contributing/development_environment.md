# Development environment

This page covers the mechanics of working on GuPPy itself: setting up a conda environment,
installing the dependency groups, and the style checks a pull request has to pass.

## Setting up a conda environment

GuPPy is developed inside a conda virtual environment named `guppy_env`. There is no
`environment.yml` in the repository; creating the environment is a documented convention, not a
file you run:

```bash
conda create -n guppy_env python=3.12
conda activate guppy_env
```

With the environment active, clone the repository and install it in editable mode:

```bash
git clone https://github.com/LernerLab/GuPPy.git
cd GuPPy
pip install -e .
```

`pip install -e .` installs GuPPy in place from `src/`, per the `where = ["src"]` setting under
`[tool.setuptools.packages.find]` in `pyproject.toml`. Edits to files under `src/guppy/` take
effect immediately, with no reinstall.

## Installing dependency groups

GuPPy declares its dependency groups under `[dependency-groups]` in `pyproject.toml`. These are
[PEP 735](https://peps.python.org/pep-0735/) dependency groups, not
`[project.optional-dependencies]` extras, so they are installed with `pip install --group
<name>`, never with an extras suffix like `pip install -e .[dev]`.

Three groups are defined:

- `dev` — `pre-commit`, for running the style hooks described below, and `towncrier`, for
  assembling the changelog at release time.
- `test` — `pytest`, `pytest-cov`, `pytest-xdist`, `pytest-playwright`, `playwright`, and
  `ndx-events==0.2.2` (pinned to the version the mock NWB test files were written with).
- `docs` — `sphinx`, `pydata-sphinx-theme`, `myst-parser`, `sphinx-autodoc-typehints`, for
  building this documentation site.

Install whichever groups you need, for example:

```bash
pip install --group dev
pip install --group test
pip install --group docs
```

## Code style

Style checks run through [pre-commit](https://pre-commit.com), configured in
`.pre-commit-config.yaml`:

- **Black** formats all Python at a 120-character line length (`[tool.black]` in
  `pyproject.toml`). `docs/` is excluded from Black's own formatting.
- **Ruff** lints with `select = ["F401", "I", "D101", "D102", "D103", "UP006", "UP007", "ANN"]` —
  unused imports, all isort rules, missing docstrings on public classes/methods/functions,
  non-PEP-585 and non-PEP-604 annotation style, and all type-hint rules. `E501` (line-too-long) is
  not selected, so Ruff's own default line length is inert; Black is what enforces line length.
  Import sorting follows `relative-imports-order = "closest-to-furthest"` with
  `known-first-party = ["guppy", "guppy_test_data"]`, and docstrings are checked against the
  `numpy` pydocstyle convention.
- **codespell** catches misspellings. It is biased toward American English, which is one of the
  reasons this documentation uses American spelling throughout.
- `check-yaml`, `end-of-file-fixer`, and `trailing-whitespace` (from the upstream
  `pre-commit-hooks` repo) apply repo-wide, including to `docs/`.

Run the hooks only on the files you actually changed:

```bash
pre-commit run --files path/to/changed_file.py path/to/other_changed_file.py
```

:::{note}
Do not run `pre-commit run --all-files`. The repository carries pre-existing violations outside
any given working set — trailing whitespace and missing final newlines across
`docs/_static/images/**/*.svg`, plus several hundred Ruff errors under `docs/scripts/` — and
`--all-files` rewrites all of them, burying your intended change under dozens of off-target
modified files.
:::

### Style contract for your first PR

A few conventions apply across the codebase, beyond what the automated hooks catch:

- All new functions are keyword-only (a bare `*` in the signature).
- Docstrings follow the numpydoc style.
- Avoid broad `try`/`except`. Prefer explicit `if`/`else`, and let real bugs fail loudly instead
  of being caught and hidden.

## Changelog entries

Every pull request that changes `src/`, `tests/`, `pyproject.toml`, or `.github/` adds a changelog
entry. Entries are not written into `CHANGELOG.md` directly: each one is its own file under
`changelog_entries/`, named `<PR number>.<type>.md`, so pull requests open at the same time never
edit the same lines. The type picks the section the entry lands in:

- `feature` — Features
- `bugfix` — Fixes
- `improvement` — Improvements
- `deprecation` — Deprecations and Removals

The file holds a sentence or two of Markdown saying what changed, from a user's point of view:

```bash
$ cat changelog_entries/503.improvement.md
The cross-correlation lag axis is now named and documented in seconds, the unit it has always held, rather than milliseconds.
```

The `[PR #503](...)` link is generated from the file name, so leave it out of the text. Because
the name carries the pull request's number, open the pull request first and add the entry once
the number is known. A pull request with a second entry of the same type names it
`<PR number>.<type>.1.md`.

[towncrier](https://towncrier.readthedocs.io/) assembles the entries into a new section at the
top of `CHANGELOG.md` when a release is made; see [Making a release](making_a_release.md).
`towncrier build --draft` prints the section the entries would produce without touching any files.

## Building the docs locally

Install the `docs` dependency group, then build with Sphinx directly — there is no `Makefile`
under `docs/`:

```bash
pip install --group docs
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser to check your changes.

The published site at [guppy.readthedocs.io](https://guppy.readthedocs.io/) is built from
`.readthedocs.yaml`, which does not set `fail_on_warning`. A broken relative link or malformed
directive will not fail the Read the Docs build, so a clean exit code from `sphinx-build` is not
enough — read the warnings it prints before opening a pull request.

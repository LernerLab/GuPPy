# v2.0.0-alpha9 (Upcoming)

## Features
- Added support for pynwb 4.0, including the new core `EventsTable` event type (NWB Schema 2.10.0); each `EventsTable` becomes a store, split into one store per unique value of its optional text `annotation` column. Dropped support for the `ndx-events` 0.4 extension, which is unreadable under pynwb 4.0. [PR #367](https://github.com/LernerLab/GuPPy/pull/367)

## Fixes
- The visualization dashboard's plot Save buttons are now "Save As…" browser downloads that let you choose the filename and location, and they render the current view (respecting live zoom/pan and typed axis limits) instead of a stale earlier render; saving no longer freezes the plot's controls until a page refresh, and render failures now surface as notifications instead of failing silently. [PR #375](https://github.com/LernerLab/GuPPy/pull/375)
- The Heat Map's first and last trial rows no longer render at half height: the Trials (Y) axis now always spans the full cell edges on every render, and its manual axis-limit boxes (which could clip the edge rows) were removed since the axis only encodes trial number. [PR #374](https://github.com/LernerLab/GuPPy/pull/374)

## Improvements
- Renamed vague variable names (`arr`, `d`, `ts`, `op`, `cols`, suffixed `*_arr`, etc.) throughout `src/guppy/` to descriptive, context-appropriate names; behavior-preserving (consistency suite unchanged). [PR #378](https://github.com/LernerLab/GuPPy/pull/378)
- Deduplicated copy-pasted code: shared timestamp-realignment kernels for artifact-removal and multi-session combining, a shared group-averaging preamble, a single pipeline-step launch helper in the homepage, and shared ndx-fiber-photometry boilerplate across the mock-NWB generators. [PR #377](https://github.com/LernerLab/GuPPy/pull/377)
- Removed commented-out dead code throughout `src/guppy/` and clarified the remaining comments. [PR #376](https://github.com/LernerLab/GuPPy/pull/376)

## Deprecations and Removals
- Standardized the store/session/run vocabulary across the codebase, GUI, and persisted contracts, and rebranded Step 1 "Save Storenames" to "Label Stores" (see the new [Glossary](https://guppy.readthedocs.io/en/latest/reference/glossary.html)). Breaking, hard cutover with no migration: `GuPPyParamtersUsed.json` keys `folderNames`→`session_folders`, `folderNamesForAvg`→`group_session_folders`, `runName`→`run_name`, `runNamePolicy`→`run_name_policy`, `selectedOutputs`→`selected_runs`, `groupSelectedOutputs`→`group_selected_runs`, and the headless `storenames_map` parameter→`store_id_to_store_label`; `storesList.csv` rows are now `store_id` (row 0) and `store_label` (row 1). Old session output folders are not migrated — re-run Step 1 to regenerate them. [PR #NNN](https://github.com/LernerLab/GuPPy/pull/NNN)

# v2.0.0-alpha8 (July 7th, 2026)

## Features
- Brought the visualization dashboard's Heat Map tab up to parity with the PSTH line plots: numeric X (Time) and Y (Trials) axis-limit boxes that snap to zoom/pan, editable colour-scale (clim) limits that recolour the datashaded data (not just the colorbar), and an independent "Hide minor tick marks" toggle. [PR #372](https://github.com/LernerLab/GuPPy/pull/372)

# v2.0.0-alpha7 (July 7th, 2026)

## Features
- Added a "Hide minor tick marks" toggle to the visualization dashboard's PSTH tab that removes the small ticks between axis numbers on the three line plots for a cleaner look (ticks shown by default). [PR #370](https://github.com/LernerLab/GuPPy/pull/370)
- Reorganized the visualization dashboard's PSTH tab into per-plot cards with numeric axis-range inputs (that snap to zoom), color pickers, a comparison-plot palette selector, and per-plot save buttons. [PR #365](https://github.com/LernerLab/GuPPy/pull/365)
- Added an optional "Import Custom Events" GUI step for pasting external behavioral timestamps (copied from a spreadsheet column), written as GuPPy-compatible single-column CSVs that surface as stores in the Storenames GUI; advanced users can hand-build the same CSV format, documented in a new how-to guide. [PR #362](https://github.com/LernerLab/GuPPy/pull/362)
- Added Iteratively Re-Weighted Least Squares (IRWLS) as the control-channel fitting method and made it the new default (robust to outliers; ordinary least-squares `OLS` fitting remains selectable via the new `control_fit_method` parameter). [PR #359](https://github.com/LernerLab/GuPPy/pull/359)
- Each pipeline step now writes `GuPPyParamtersUsed.json` into its output directory automatically, and selecting an existing output run reloads its saved parameters into the form so the snapshot always matches what was executed and resuming a run no longer overwrites its parameters. Removed the manual "Save Input Parameters" button and renumbered the sidebar steps 1–5. Resolves [#301](https://github.com/LernerLab/GuPPy/issues/301). [PR #353](https://github.com/LernerLab/GuPPy/pull/353)
- Added docstring checks to pre-commit. [PR #311](https://github.com/LernerLab/GuPPy/pull/311)
- Added numpydoc-style docstrings to all public functions and classes in the extractor layer (`src/guppy/extractors/`). [PR #312](https://github.com/LernerLab/GuPPy/pull/312)
- Added numpydoc-style docstrings to all public functions and classes in the analysis layer (`src/guppy/analysis/`). [PR #313](https://github.com/LernerLab/GuPPy/pull/313)
- Added numpydoc-style docstrings to all public functions and classes in the orchestration layer (`src/guppy/orchestration/`). [PR #314](https://github.com/LernerLab/GuPPy/pull/314)
- Added numpydoc-style docstrings to all public functions and classes in the frontend layer (`src/guppy/frontend/`). [PR #316](https://github.com/LernerLab/GuPPy/pull/316)
- Added numpydoc-style docstrings to all public functions in the utils layer (`src/guppy/utils/`). [PR #317](https://github.com/LernerLab/GuPPy/pull/317)
- Added numpydoc-style docstrings to all public functions in the testing layer (`src/guppy/testing/`). [PR #318](https://github.com/LernerLab/GuPPy/pull/318)
- Added parameterized output directories: step 2 accepts a user-supplied run name, steps 1 and 3–6 honour a per-session run-name filter, and `GuPPyParamtersUsed.json` is written into the selected output directories so multiple parameter sets can coexist in one session. [PR #325](https://github.com/LernerLab/GuPPy/pull/325)
- Added type hint checks to pre-commit. [PR #346](https://github.com/LernerLab/GuPPy/pull/346)
- Added type hints to all functions in the frontend layer (`src/guppy/frontend/`). [PR #351](https://github.com/LernerLab/GuPPy/pull/351)
- Added type hints to all functions in the orchestration layer (`src/guppy/orchestration/`). [PR #350](https://github.com/LernerLab/GuPPy/pull/350)
- Added type hints to all functions in the analysis layer (`src/guppy/analysis/`). [PR #349](https://github.com/LernerLab/GuPPy/pull/349)
- Added type hints to all functions in the extractors layer (`src/guppy/extractors/`). [PR #348](https://github.com/LernerLab/GuPPy/pull/348)
- Added type hints to all functions in the utils, visualization, testing, and root layers. [PR #347](https://github.com/LernerLab/GuPPy/pull/347)

## Fixes
- Group averaging now only requires the selected sessions to share the same fiber (control/signal) storenames rather than an identical full storename set, so sessions recorded from the same region under different behavioral conditions (e.g. Novel Object vs Novel Female) can be averaged together for cross-condition group figures. [PR #369](https://github.com/LernerLab/GuPPy/pull/369)
- Fixed the visualization dashboard rendering blank (only the title bar, no plots or controls) when an event's group average had a single contributing session: the single-trial heatmap drew a raw QuadMesh across the full time axis, overflowing Bokeh's client-side renderer. Single-trial heatmaps now use the same datashaded path as multi-trial ones. [PR #369](https://github.com/LernerLab/GuPPy/pull/369)
- Unified the pipeline step numbering on the canonical Storenames = Step 1 scheme across the testing API, tests, error messages, comments, and docs, so error messages that tell the user to re-run a step now match the GUI sidebar labels. [PR #361](https://github.com/LernerLab/GuPPy/pull/361)
- Stored event timestamps now share the recording-start time basis with the continuous `timestampNew` stream instead of being re-zeroed to `timeForLightsTurnOn`, so all series can be co-registered without per-stream offset bookkeeping (PSTH results are unchanged). Resolves [#355](https://github.com/LernerLab/GuPPy/issues/355). [PR #356](https://github.com/LernerLab/GuPPy/pull/356)
- Fixed bug with step five, which was causing the baseline uncorrected HDF5 file to not exist. [PR #241](https://github.com/LernerLab/GuPPy/pull/241)

## Improvements
- Hoisted step-3 multiprocessing pool out of the per-session loop and batched reads per `(session, extractor)` pair: ~3.3× faster DANDI streaming and ~2.2× faster local NWB on representative sessions.
- Expanded the first tutorial with embedded screenshots and a step-by-step walkthrough of the Storenames and Visualization GUIs, corrected button names and HDF5 output descriptions, and added `docs/take_screenshots.py` to regenerate the tutorial screenshots from the stubbed CSV sample data. [PR #303](https://github.com/LernerLab/GuPPy/pull/303)
- Saved GuPPy version and expanded the parameter set written to `GuPPyParamtersUsed.json` (adds `artifactsRemovalMethod`, `computeCorr`, `plot_zScore_dff`, `visualize_zscore_or_dff`, `averageForGroup`). [PR #328](https://github.com/LernerLab/GuPPy/pull/328)
- Renamed the per-event dict variable `S` (in `tdt_recording_extractor.py` and `doric_recording_extractor.py`) to `event_dict`, the helper-local `new_S` to `split_event_dict`, and the storenames-config dict `d` (in `storenames.py`, `_fetchValues`/`_save`, and the `StorenamesSelector.{get,set}_literal_input_2` parameter) to `storenames_config`, addressing part of [#187](https://github.com/LernerLab/GuPPy/issues/187). [PR #304](https://github.com/LernerLab/GuPPy/pull/304)
- Deduplicated the two copy-pasted `write_hdf5` implementations (extractor side and analysis side) into a single canonical helper at `guppy.utils._hdf5_io`, fixing a latent bug on the analysis-side writer that silently dropped scalar overwrites when the key already existed. Also extracted the duplicated `_default_root_path` helper into `guppy.frontend.frontend_utils.default_root_path` so the `GUPPY_BASE_DIR` precedence rule lives in one place. Addresses part of [#174](https://github.com/LernerLab/GuPPy/issues/174). [PR #305](https://github.com/LernerLab/GuPPy/pull/305)
- Added a Read the Docs documentation badge to `README.md` and a `Documentation` project URL in `pyproject.toml` pointing at https://guppy.readthedocs.io/, so the documentation is discoverable directly from the GitHub landing page and the PyPI listing. [PR #306](https://github.com/LernerLab/GuPPy/pull/306)
- Added cross-correlation explanation page to the documentation site, with six generated SVG figures and a self-contained PEP 723 script (`docs/scripts/cross_correlation_explainer.py`) that regenerates them in place. [PR #307](https://github.com/LernerLab/GuPPy/pull/307)
- Added an explanation page on z-score normalization (standard, baseline, and modified variants) at `docs/explanation/zscore.md`, with four generated SVG figures and a self-contained PEP 723 script that regenerates them. Enabled MyST `dollarmath` for LaTeX equation rendering and bumped the Read the Docs build to Python 3.13 / ubuntu-24.04 with the `docs` dependency group. [PR #308](https://github.com/LernerLab/GuPPy/pull/308)
- Added a PSTH explanation page (origin, the construction operation, drift correction, peak vs AUC summary statistics, and event rejection) at `docs/explanation/psth.md`, with four generated SVG figures and a self-contained PEP 723 script (`docs/scripts/psth_explainer.py`) that regenerates them in place. [PR #315](https://github.com/LernerLab/GuPPy/pull/315)
- Added a fiber photometry explanation page at `docs/explanation/fiber_photometry.md`, with one hand-illustrated technique schematic (PNG) and two generated SVG figures (population summing, and a space-vs-time landscape placing photometry against electrophysiology, two-photon imaging, and fMRI). The matplotlib figures are produced by a self-contained PEP 723 script at `docs/scripts/fiber_photometry_explainer.py`. [PR #319](https://github.com/LernerLab/GuPPy/pull/319)
- Added an explanation page on the isosbestic correction at `docs/explanation/isosbestic_correction.md`, covering the two-state GCaMP framework, GuPPy's linear-fit-and-subtract procedure, what the corrected trace does and does not remove, and why the synthetic-exponential fallback is not equivalent. Eight generated SVG figures and a self-contained PEP 723 script (`docs/scripts/isosbestic_explainer.py`) that regenerates them in place. [PR #324](https://github.com/LernerLab/GuPPy/pull/324)
- Added an input parameter reference page at `docs/reference/parameters.md`, documenting every GUI parameter (description, type, default, accepted range) organised to mirror the homepage cards, with an alphabetical index mapping internal `inputParameters` keys (as written in `GuPPyParametersUsed.json` and the headless API) back to their GUI parameters. Enabled `myst_heading_anchors = 3` in `conf.py` so intra-page section links resolve. [PR #341](https://github.com/LernerLab/GuPPy/pull/341)
- Added a transient detection explanation page (motivation, basic detector, drift handling via per-chunk MAD, the two-stage outlier-trim scheme, summary statistics, and limitations) at `docs/explanation/transient_detection.md`, with six generated SVG figures and a self-contained PEP 723 script (`docs/scripts/transient_detection_explainer.py`) that regenerates them in place. [PR #332](https://github.com/LernerLab/GuPPy/pull/332)
- Added an explanation page on artifacts in fiber photometry at `docs/explanation/artifacts.md`, covering the catalogue of common artifacts grouped by recording-chain stage, four property axes (time structure, wavelength dependence, behaviour coupling, frequency content) used to characterise them, a decision tree routing artifacts to correction methods, and upstream avoidance through indicator and protocol choices. Two generated SVG figures and a self-contained PEP 723 script (`docs/scripts/artifacts_explainer.py`) that regenerates them in place.
- Updated stubbed testing data README.md with complete descriptions of each store name. [PR #343](https://github.com/LernerLab/GuPPy/pull/343)
- `NpmRecordingExtractor` now demultiplexes interleaved channels and splits events entirely in memory rather than writing intermediate CSVs into the source data folder; the per-file decomposition parameters are persisted to the output directory so step 3 can reproduce them. Addresses part of [#329](https://github.com/LernerLab/GuPPy/issues/329). [PR #352](https://github.com/LernerLab/GuPPy/pull/352)
- Moved TDT epoc split-event determination from step 3 to step 2, so `storesList.csv` is fully settled at discovery time; `read()` no longer mutates `storesList.csv` or leaves a `.cache_storesList.csv` behind, and split sub-events are now labeled by the user in the storenames step. Addresses part of [#329](https://github.com/LernerLab/GuPPy/issues/329). [PR #352](https://github.com/LernerLab/GuPPy/pull/352)

## Deprecations and Removals
- Removed the manual "Save Input Parameters" button (Step 1); each pipeline step now writes the parameter snapshot automatically. [PR #353](https://github.com/LernerLab/GuPPy/pull/353)


# v2.0.0-alpha6 (April 29th, 2026)

## Features
- Updated license from GPL to BSD-3-Clause [PR #309](https://github.com/LernerLab/GuPPy/pull/309)


# v2.0.0-alpha5 (April 28th, 2026)

## Fixes
- Cross-correlation now raises a descriptive `ValueError` (instead of silently skipping) when `compute_cross_correlation=True` but fewer than two distinct signal regions are present; the error message is surfaced as a persistent notification in the Panel UI so users do not need to inspect the terminal. [PR #284](https://github.com/LernerLab/GuPPy/pull/284)
- Fixed stale output data when overwriting storenames in step 2: the output directory is now fully cleared before writing the new `storesList.csv`, removing any leftover HDF5 files and other pipeline artefacts. [PR #281](https://github.com/LernerLab/GuPPy/pull/281)
- Replaced the uninformative `"Error in naming convention of files or Error in storesList file"` exception with an actionable message that reports the mismatching pair-name suffixes, the directory searched, and a suggestion to re-run step 2. [PR #280](https://github.com/LernerLab/GuPPy/pull/280)
- Replaced the generic error for invalid `baselineWindowStart`/`baselineWindowEnd` values with layered upfront validation in the baseline z-score path: rejects non-numeric/`NaN` inputs, enforces `start < end`, and checks each bound against the signal timespan — reporting which limit was violated and the valid range. The input parameters tooltip now documents the expected units, ordering, and bounds. [PR #283](https://github.com/LernerLab/GuPPy/pull/283)
- Switched the remaining numeric input-parameter widgets (`timeForLightsTurnOn`, `numberOfCores`, `moving_wd`, `highAmpFilt`, `transientsThresh`, `moving_avg_filter`, `no_channels_np`, `nSecPrev`, `nSecPost`, `timeInterval`, `bin_psth_trials`, `baselineCorrectionStart`, `baselineCorrectionEnd`) from `LiteralInput(type=int)` to `IntInput`, so non-numeric input is rejected at the browser level instead of being silently reverted to the previous valid value. [PR #297](https://github.com/LernerLab/GuPPy/pull/297)
- Step 6 now validates the visualization metric selection (`z-score or ΔF/F? (for visualization)`) against the step-5 PSTH outputs on disk at the start of `visualizeResults`. When the requested metric was not computed in step 5, a `ValueError` is raised that names the missing metric, lists the affected session output directories, and tells the user to either change the visualization selection or re-run step 5 with the relevant option enabled. [PR #288](https://github.com/LernerLab/GuPPy/pull/288)
- Doric extractor now validates signal/control channels early with actionable error messages: rejects empty, non-finite (NaN/inf), and constant (zero-variance) channels — common in unused `AIn-X - Dem (AOut-Y)` demodulation channels and LED-drive `AOut` outputs. Also raises a descriptive `ValueError` when a requested channel name is missing from a Doric CSV / V1 / V6 file, listing the available channels, and filters trailing all-NaN columns (e.g. `Unnamed: 7` from Doric CSVs with trailing commas) during event discovery so they no longer surface as selectable events in Step 2. The mixed-modality `read_raw_data` orchestrator likewise lists available events when a requested event is not found in any extractor. [PR #290](https://github.com/LernerLab/GuPPy/pull/290)
- Added group-analysis validation in steps 5 and 6 with actionable error messages for mismatched/non-overlapping storenames and missing average outputs, replacing the prior `IndexError` and silent fall-through behaviors. [PR #293](https://github.com/LernerLab/GuPPy/pull/293)
- Sidebar button click handlers now surface input-parameter validation errors (e.g. "No folder is selected for analysis") as a persistent Panel notification instead of dying silently in a worker thread traceback. [PR #296](https://github.com/LernerLab/GuPPy/pull/296)
- Fixed TDT split-event extraction collapsing float-valued event codes (e.g. `0.1, 0.2, 0.4, 0.8, 10.0`) to integers, which caused duplicate `storesList.csv` rows, silent overwrites of per-code HDF5 files, and a downstream `KeyError` in step-6 visualization; sub-event suffixes now preserve unique floats as filesystem-safe `0p1`, `0p2`, … strings. [PR #294](https://github.com/LernerLab/GuPPy/pull/294)
- Fixed `detect_acquisition_formats` skipping the intermediate `event*.csv` files that `NpmRecordingExtractor` materializes when `npm_split_events=True`, which left `CsvRecordingExtractor` undispatched and broke step-3 reads of NPM split-event TTLs. Single-column timestamp CSVs are now uniformly reported as `csv` regardless of whether NPM data is present. [PR #298](https://github.com/LernerLab/GuPPy/pull/298)

## Improvements
- Added input validation in step 2 to reject duplicate store names and mismatched signal/control region pairs, with descriptive error messages naming the offending entries. [PR #275](https://github.com/LernerLab/GuPPy/pull/275)
- Consolidated input-validation logic from the [#138](https://github.com/LernerLab/GuPPy/issues/138) sub-PRs into `src/guppy/utils/validation.py` and moved peak-window and PSTH baseline-correction validation upfront in step 5 so errors surface before any HDF5 IO. [PR #299](https://github.com/LernerLab/GuPPy/pull/299)
- Audited every user-facing error message across `src/guppy/`: stripped ANSI escape codes, converted input-validation `assert`s and generic `raise Exception(...)` calls to `ValueError` / `FileNotFoundError`, and rewrote vague strings (naming-convention mismatches, CSV column counts, TDT/Doric/NWB extractor errors, typos) to name the offending value, state the rule, and give the fix. [PR #299](https://github.com/LernerLab/GuPPy/pull/299)


# v2.0.0-alpha4 (April 15th, 2026)

## Features
- Added read support for NWB files with dedicated recording extractor. [PR #261](https://github.com/LernerLab/GuPPy/pull/261)
- Added --start-path option to guppy launch command [PR #265](https://github.com/LernerLab/GuPPy/pull/265)
- Added a dedicated DANDI NWB streaming extractor, a prototype streaming script, and orchestration-layer support for running step 2 and step 3 against `dandi://` URIs. [PR #266](https://github.com/LernerLab/GuPPy/pull/266)
- Added support for streaming NWB files from DANDI, complete with front-end file selector. [PR #267](https://github.com/LernerLab/GuPPy/pull/267)

## Fixes
- Fixed pickling issue for long storenames in `read_and_save_all_events`. [PR #261](https://github.com/LernerLab/GuPPy/pull/261)

## Improvements
- Added documentation site with Sphinx, pydata-sphinx-theme, and MyST-Parser. Includes Diataxis structure and a first tutorial covering the end-to-end GUI workflow with stubbed CSV test data. [PR #264](https://github.com/LernerLab/GuPPy/pull/264)
- Improved test suite coverage to greater than or equal to 85% on CodeCov. [PR #260](https://github.com/LernerLab/GuPPy/pull/260)


# v2.0.0-alpha3 (April 1st, 2026)

## Fixes
- Fixed npm_recording_extractor.py bug that was caused by mixing standard event CSV type CSVs with NPM data. [PR #256](https://github.com/LernerLab/GuPPy/pull/256)

## Improvements
- Expanded test suite with unit tests for frontend components: [PR #250](https://github.com/LernerLab/GuPPy/pull/250)
- Re-balanced test suite to conform to standard testing pyramid: [PR #255](https://github.com/LernerLab/GuPPy/pull/255)


# v2.0.0-alpha2 (March 31st, 2026)

## Fixes
- Fixed plot saving logic and added selenium as a dependency. [PR #252](https://github.com/LernerLab/GuPPy/pull/252)


# v2.0.0-alpha1 (March 30th, 2026)

## Features

- Modernized python packaging and distribution: [PR #129](https://github.com/LernerLab/GuPPy/pull/129)
- Added support for Python 3.10-3.13: [PR #129](https://github.com/LernerLab/GuPPy/pull/129)
- Added pytest-based headless test suite for pipeline steps 1–5 with CI workflows: [PR #153](https://github.com/LernerLab/GuPPy/pull/153)
- Added daily tests to the automatic CI/CD pipeline: [PR #234](https://github.com/LernerLab/GuPPy/pull/234)

## Fixes

- Fixed bug with group analysis by updating pandas syntax: [PR #192](https://github.com/LernerLab/GuPPy/pull/192)

## Deprecations and Removals

- Dropped support for Python 3.6: [PR #129](https://github.com/LernerLab/GuPPy/pull/129)
- Restructured directory layout for improved organization: [PR #129](https://github.com/LernerLab/GuPPy/pull/129)
- Converted savingInputParameters.ipynb to saving_input_parameters.py: [PR #129](https://github.com/LernerLab/GuPPy/pull/129)

## Improvements

- Replaced scattered print statements with centralized structured logging: [PR #160](https://github.com/LernerLab/GuPPy/pull/160)
- Added pre-commit hooks for automated code formatting and linting (Black + Ruff): [PR #161](https://github.com/LernerLab/GuPPy/pull/161)
- Expanded test suite with an additional example session across steps 2–5: [PR #179](https://github.com/LernerLab/GuPPy/pull/179)
- Introduced `BaseRecordingExtractor` and format-specific subclasses for TDT, Doric, NPM, and CSV data ingestion: [PR #171](https://github.com/LernerLab/GuPPy/pull/171)
- Refactored monolithic analysis code into modular components under `src/guppy/analysis/`: [PR #190](https://github.com/LernerLab/GuPPy/pull/190)
- Refactored frontend code into modular components under `src/guppy/frontend/`: [PR #191](https://github.com/LernerLab/GuPPy/pull/191)
- Added code coverage reporting via Codecov to CI workflows: [PR #194](https://github.com/LernerLab/GuPPy/pull/194)
- Added GitHub Actions workflow to automatically detect changes to the source code and require updating the changelog for any PR that modifies code: [PR #233](https://github.com/LernerLab/GuPPy/pull/233)
- Restored automatic data modality detection and mixed-modality TTL/signal support with a modular, separation-of-concerns architecture: [PR #226](https://github.com/LernerLab/GuPPy/pull/226)
- Expanded test suite with consistency tests that compare results to GuPPy-v1.3.0: [PR #207](https://github.com/LernerLab/GuPPy/pull/207)
- Expanded test suite with unit tests for recording extractor classes: [PR #240](https://github.com/LernerLab/GuPPy/pull/240)
- Migrated testing datasets from Google Drive to GitHub LFS with comprehensive documentation and CI/CD integration: [PR #242](https://github.com/LernerLab/GuPPy/pull/242)
- Added GitHub Actions cache for stubbed testing data to avoid exhausting Git LFS bandwidth limits: [PR #245](https://github.com/LernerLab/GuPPy/pull/245)
- Expanded test suite with unit tests for analysis functions: [PR #247](https://github.com/LernerLab/GuPPy/pull/247)
- Expanded test suite with unit tests for orchestration functions: [PR #249](https://github.com/LernerLab/GuPPy/pull/249)
- Expanded test suite with unit tests for utility functions: [PR #248](https://github.com/LernerLab/GuPPy/pull/248)

# GuPPy-v1.3.0 (August 12th, 2025)

- Added support for NPM TTL files with multiple format versions
- Added support for multiple NPM files and CSV TTL files simultaneously
- Added binning by trials feature for data organization
- Extended peak AUC analysis with additional window options
- Enhanced cross-correlation module with artifact removal options
- Optional filtering - can disable signal filtering when needed
- Improved storenames GUI for better user experience
- Automatic saving of input parameters for group analysis
- Enhanced visualization GUI with improved Y-axis limits
- Fixed Windows and macOS compatibility issues
- Improved Doric file format support
- Added directory checking for output folders
- Fixed various bugs in group analysis and PSTH computation
- Resolved port number errors and improved error handling

# GuPPy-v1.2.0 (November 11th, 2021)

- Support for Doric system file (.csv and .doric)
- storenames GUI changed, designed it in a way which is less error prone
- Saving of input parameters is not required for doing the analysis
- Visualization GUI changed
- user-defined for number of cores used
- added cross-correalation computation
- two user-defined parameters for transients detection
- artifacts removal can be done with two different methods
- compute negative peaks along with positive peaks in a user-defined window

# GuPPy-v1.1.4 (October 28th, 2021)

- Support for Neurophotometrics data
- Option for binning of PSTH trials
- Option to carry out analysis without using isosbestic control channel
- Plot to see control fitted channel to signal channel
- Selection and deletion of chunks with specific keys in artifacts removal
- Option to change moving average filter window
- Option to compute variations of z-score based on different computation method.
- Faster computation speed for PSTH computation step

# GuPPy-v1.1.2 (August 4th, 2021)

- Minor Bug Fixes
- multiple windows for peak and AUC computation
- bug fix for searching a file name irrespective of lower-case of upper-case

# GuPPy-v1.1.1 (July 6th, 2021)

It is the GuPPy's first release for people to use and give us feedbacks on it

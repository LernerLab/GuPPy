# v2.0.0-alpha7 (Upcoming)

## Features
- Added docstring checks to pre-commit. [PR #311](https://github.com/LernerLab/GuPPy/pull/311)
- Added numpydoc-style docstrings to all public functions and classes in the extractor layer (`src/guppy/extractors/`). [PR #312](https://github.com/LernerLab/GuPPy/pull/312)
- Added numpydoc-style docstrings to all public functions and classes in the analysis layer (`src/guppy/analysis/`). [PR #313](https://github.com/LernerLab/GuPPy/pull/313)
- Added numpydoc-style docstrings to all public functions and classes in the orchestration layer (`src/guppy/orchestration/`). [PR #314](https://github.com/LernerLab/GuPPy/pull/314)
- Added numpydoc-style docstrings to all public functions and classes in the frontend layer (`src/guppy/frontend/`). [PR #316](https://github.com/LernerLab/GuPPy/pull/316)
- Added numpydoc-style docstrings to all public functions in the utils layer (`src/guppy/utils/`). [PR #317](https://github.com/LernerLab/GuPPy/pull/317)
- Added numpydoc-style docstrings to all public functions in the testing layer (`src/guppy/testing/`). [PR #318](https://github.com/LernerLab/GuPPy/pull/318)

## Fixes

## Improvements
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

## Deprecations and Removals


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

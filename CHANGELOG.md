# v2.0.0 (Upcoming)

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
- Expanded test suite with unit tests for orchestration functions: [PR #248](https://github.com/LernerLab/GuPPy/pull/248)

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

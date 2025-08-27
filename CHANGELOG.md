# v2.0.0 (Upcoming)

## Features

- Modernized python packaging and distribution: [PR #129](https://github.com/LernerLab/GuPPy/pull/129)

## Fixes

## Deprecations and Removals

## Improvements

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
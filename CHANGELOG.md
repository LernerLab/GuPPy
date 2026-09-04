# v2.0.0-beta3 (Upcoming)

## Features
- Artifact periods can now be marked by dragging horizontally across a trace on the Select Artifact Windows page, which shows one selectable trace at a time and toggles between **Mark artifacts** and **Navigate**; a **Copy windows from run** control loads the periods already saved for another run of the same session. [PR #472](https://github.com/LernerLab/GuPPy/pull/472)

## Fixes
- An event store whose timestamps come from a different clock than the signal is now rejected with a message naming the mismatch — for Neurophotometrics sessions, the timestamp column the events actually ride — instead of failing with an unreadable array error during PSTH computation. [PR #489](https://github.com/LernerLab/GuPPy/pull/489)
- A bound typed outside the recording on the Select Artifact Windows page is now pulled in to the nearest edge, with a warning naming the bound and the recording edge it moved to, instead of failing the first Save and then silently succeeding against a rewritten value. [PR #472](https://github.com/LernerLab/GuPPy/pull/472)
- An artifact period that reaches the start or the end of the trace now trims that end of the recording outright, so a single session can lose more of its opening than **Eliminate first few seconds** takes from the whole batch. [PR #472](https://github.com/LernerLab/GuPPy/pull/472)
- Fixed the README's documentation links, which all carried an `/en/latest/` path prefix that 404s on the single-version Read the Docs project. [PR #469](https://github.com/LernerLab/GuPPy/pull/469)

## Improvements
- The rest of the analysis layer and GuPPy's validation, custom-event and NWB helpers now build paths with `pathlib.Path`. [PR #484](https://github.com/LernerLab/GuPPy/pull/484)
- GuPPy's core HDF5 and results I/O now builds paths with `pathlib.Path` rather than `os.path`, and `ruff`'s `PTH` rules are enabled with the not-yet-converted modules listed explicitly so the migration is tracked rather than open-ended. [PR #483](https://github.com/LernerLab/GuPPy/pull/483)
- Every `zip()` over parallel sequences now declares `strict=`, so a pair that has silently truncated to the shorter operand raises instead. [PR #482](https://github.com/LernerLab/GuPPy/pull/482)
- The six `write_*` helpers in `analysis/standard_io.py` no longer share a mutable list as their `index`/`columns` default, and each visualization dashboard now keeps its own render cache instead of one shared across every open dashboard. `ruff`'s `B006`, `B008`, `RUF012` and `DTZ` rules are enabled to keep both classes of shared state out. [PR #482](https://github.com/LernerLab/GuPPy/pull/482)
- Logging calls now hand their arguments to the logger instead of formatting the message up front, so a filtered-out record costs nothing to build. Pre-f-string syntax (`.format()`, `typing.Callable`/`Sequence`/`Iterable`, `lru_cache(maxsize=None)`) is swept, and `ruff`'s `UP` and `G` rule sets are enabled to keep it that way. [PR #481](https://github.com/LernerLab/GuPPy/pull/481)
- Reading and writing `storesList.csv` now goes through a single `read_stores_list()`/`write_stores_list()` pair instead of a `np.genfromtxt(...).reshape(2, -1)` incantation copy-pasted across 22 call sites. [PR #480](https://github.com/LernerLab/GuPPy/pull/480)
- Which existing run Steps 2–5 read can now be chosen by name: a **Run name(s) for all sessions** picker above the Output Folder Selection browser selects that run in every selected session at once, and directories ticked in the browser are left alone. Changing the session selection no longer discards the run choices already made for the other sessions. [PR #474](https://github.com/LernerLab/GuPPy/pull/474)
- Reading TDT tanks is much faster on a network share: each continuous store's `.tev` data is now fetched in large sequential chunks instead of one small read per data block, so Read Raw Data no longer pays a network round trip for every block. [PR #473](https://github.com/LernerLab/GuPPy/pull/473)
- GuPPy now reports its own version: `guppy --version` prints it, the user interface header shows it, and the [installation page](https://guppy.readthedocs.io/en/latest/installation.html) covers how to check it and how to upgrade an existing install. [PR #471](https://github.com/LernerLab/GuPPy/pull/471)
- The documentation now explains why `.h5` and `.hdf5` outputs need different readers, rather than only warning that they do: `.h5` files are written by pandas in its fixed format, so reading one with h5py shows the table decomposed into its storage blocks instead of its columns. [PR #470](https://github.com/LernerLab/GuPPy/pull/470)

## Deprecations and Removals

# v2.0.0-beta2 (September 3rd, 2026)

## Features
- Added a **Compute PSTH Significance?** parameter: a bootstrap confidence interval at every timepoint marks which stretches of the PSTH window differ from baseline, and which differ between a named pair of events. Runs in Step 4 over a session's trials and in Group Analysis over a group's session averages, writing `psth_significance_output/` and a new **Significance** tab in Step 5. [PR #453](https://github.com/LernerLab/GuPPy/pull/453)
- Added a **Use Transients as Events?** parameter for spontaneous activity, where there is no external event to align to: the transients detected in each recording site now become that recording site's event timestamps, so the PSTH, peak and AUC are computed against them directly instead of requiring an artificial TTL file to be exported and re-imported by hand. [PR #424](https://github.com/LernerLab/GuPPy/pull/424)
- Added tonic/basal fluorescence analysis for pharmacological experiments as a new optional **Tonic Analysis** step: name epoch windows (e.g. baseline vs. post-injection) per recording site on the preprocessed traces, and saving averages the z-score and ΔF/F over each window into `tonic_<recording_site>.h5`. The visualization's Tonic tab charts each epoch's change from a selectable baseline epoch as bars, alongside the windows shaded on the traces and a table of the absolute means. PSTH analysis is unaffected. [PR #397](https://github.com/LernerLab/GuPPy/pull/397)
- Added a **Compute Binned Metrics?** parameter: the whole session can now be divided into fixed-width time bins, each reporting its mean z-score, mean ΔF/F and transient count, for correlating the signal against a behavioral measure scored on its own fixed schedule instead of against discrete events. Results are written per recording site and shown on a new **Binned** tab in Step 5. [PR #422](https://github.com/LernerLab/GuPPy/pull/422)
- Added a **behavioral covariate** store type: a continuous variable measured outside the rig — a hand-scored akinesia severity, say — can now be brought in as an ordinary GuPPy CSV, labeled in Step 1, and correlated against the per-bin photometry metrics. Pearson r and Spearman rho are reported per recording site as descriptive statistics, without a p-value, because the two series are autocorrelated across bins in a way the usual significance tests do not allow for. Results are shown on a new **Covariates** tab in Step 5, where the covariate's own trace and its per-bin means stack under the photometry trace and metric bins they were correlated against, on one linked time axis, above a bin-by-bin scatter with its least-squares line. [PR #425](https://github.com/LernerLab/GuPPy/pull/425)
- Reworked group analysis into two optional steps: **Label Groups** defines a named group of output runs in its own page, and **Group Analysis** averages into the groups picked in the new **Group Output Folder Selection** card — the same selection Step 5 visualizes, so several groups can coexist and be opened alongside individual sessions. The **Average Group?** and **Visualize Average Results?** parameters are removed. [PR #446](https://github.com/LernerLab/GuPPy/pull/446)

## Fixes
- Sessions no longer have to sit side by side to be analyzed together: the file browser now accepts session folders from different parent directories, so a run can mix sessions kept in separate data roots. [PR #451](https://github.com/LernerLab/GuPPy/pull/451)
- Fixed a deadlock that could hang Read Raw Data, PSTH computation or transient detection indefinitely: the worker pools were torn down with `terminate()`, which waits for each worker to die after signalling it, so a worker slow to exit left the run stuck with no error and no progress. The pools are now closed and joined, letting each worker finish on its own. [PR #455](https://github.com/LernerLab/GuPPy/pull/455)

## Improvements
- NWB export now writes the tonic epoch means, binned metrics, behavioral covariates and spontaneous-mode events that earlier releases silently left out, and warns when a selected run holds PSTH significance results, which are not exported yet. [PR #457](https://github.com/LernerLab/GuPPy/pull/457)
- Made `count_samples()` a required method of the recording extractor contract, so a new acquisition format that omits it fails immediately instead of quietly mis-sizing the Read Raw Data progress bar. [PR #450](https://github.com/LernerLab/GuPPy/pull/450)
- Refreshed the GuPPy branding: a new logo across the README and the documentation site, a fish-only mark in the docs header that stays legible in both light and dark mode, and a favicon for the documentation site, which previously had none. [PR #445](https://github.com/LernerLab/GuPPy/pull/445)
- Reworked the documentation home page into a card-grid launch point over the Diátaxis sections, moved the installation walkthrough from the README onto its own [installation page](https://guppy.readthedocs.io/en/latest/installation.html), and trimmed the README to a quick start plus pointers to the docs. [PR #444](https://github.com/LernerLab/GuPPy/pull/444)
- Added a [how-to guide for group analysis](https://guppy.readthedocs.io/en/latest/how-to/group-analysis.html): running Step 4's cross-session averaging and Step 5's averaged-results visualization. [PR #435](https://github.com/LernerLab/GuPPy/pull/435)
- Removed the `progress_bar` pytest marker, which was applied to no tests and made every Windows CI run deselect nothing. [PR #449](https://github.com/LernerLab/GuPPy/pull/449)
- Hardened the test workflow against the failures that had been turning the daily run red: tests are bounded by a 300-second per-test timeout so a hang reports a traceback instead of stalling the job, the daily runs get a 60-minute budget to fit the full-data consistency suite, coverage is collected only on the one job that uploads it, and a port-scanning test that failed at random is now deterministic. [PR #455](https://github.com/LernerLab/GuPPy/pull/455)

## Deprecations and Removals
- Removed headless mode: the `GUPPY_BASE_DIR` environment variable and the `is_headless()` helper are gone, with the file pickers' starting directory now coming from `guppy --start-path`. The scripted branches of Label Stores, Label Groups and Input Metadata are removed too — `guppy.testing.api` now drives the real Panel pages headlessly — and Step 1's create-new mode no longer silently overwrites an existing run folder. The `base_dir` argument of `guppy.testing.api.label_groups` is removed, and `api.step1` gains an `isosbestic_control` keyword. [PR #452](https://github.com/LernerLab/GuPPy/pull/452)

# v2.0.0-beta1 (August 20th, 2026)

## Features
- Added export to NWB, with a metadata input form for supplying subject/session details. Every format GuPPy reads is supported: a session recorded in TDT, Doric, Neurophotometrics or GuPPy's own CSV layout is bundled with its GuPPy outputs, and a session read from an NWB file — locally or streamed from DANDI — has its outputs added to the file it came from, so it needs no metadata form at all. A session's format is detected from its folder rather than assumed, and events imported from custom CSVs are read alongside whatever the rig recorded. The metadata form takes a session start time, marked as required for every format but TDT, whose tank header is the only raw one that reliably records it. Sessions the export cannot take are refused upfront with a message naming why: sessions whose traces come from more than one acquisition system, and sessions analyzed with **Combine Data?** enabled. NWB export also aborts upfront with a clear message when a selected session had its artifacts removed with the `concatenate` method, which re-times samples and breaks alignment to the acquisition clock. Documented in a how-to covering both export routes, an explanation page on what an exported file holds, and reference entries for the files the two steps write; sessions carrying data GuPPy does not read are pointed at NeuroConv, which can combine the GuPPy outputs with other acquisition streams in a single file. [PR #357](https://github.com/LernerLab/GuPPy/pull/357)
- Added a **Photobleaching Detrend?** parameter: the isosbestic control fit can now include an exponential decay term for the photobleaching the control channel does not see, removing the slow drift that otherwise survives into the ΔF/F on multi-hour recordings. [PR #416](https://github.com/LernerLab/GuPPy/pull/416)
- Added an **AUC Units** parameter: the peak/AUC areas can now be reported in z-score (or ΔF/F) × seconds, the unit commonly reported in the literature, instead of the sampling-rate-dependent one-sample spacing that remains the default. [PR #415](https://github.com/LernerLab/GuPPy/pull/415)
- Artifact removal is now two optional steps of its own — **Select Artifact Windows** and **Remove Artifacts** — where you mark the periods containing artifacts and apply them in one pass. [PR #413](https://github.com/LernerLab/GuPPy/pull/413)
- Added a baseline-epoch mode for isosbestic control fitting: fit coefficients can now be estimated from a user-specified pre-injection window and applied across the whole recording, so a step-change (e.g. a drug injection) no longer corrupts the control fit. [PR #392](https://github.com/LernerLab/GuPPy/pull/392)
- Added support for pynwb 4.0, including the new core `EventsTable` event type (NWB Schema 2.10.0); each `EventsTable` becomes a store, split into one store per unique value of its optional text `annotation` column. Dropped support for the `ndx-events` 0.4 extension, which is unreadable under pynwb 4.0. [PR #367](https://github.com/LernerLab/GuPPy/pull/367)
- Soft-deprecated the `concatenate` artifact-removal method: `replace with NaN` is now the default because `concatenate` re-times the kept samples onto a fresh timeline, breaking alignment to the acquisition clock. [PR #396](https://github.com/LernerLab/GuPPy/pull/396)

## Fixes
- The OLS isosbestic control fit is back on the better-conditioned `polyfit` solve it used before the photobleaching detrend work, which had incidentally routed it through an unscaled least-squares solve. [PR #423](https://github.com/LernerLab/GuPPy/pull/423)
- The Neurophotometrics timestamp unit recorded in `.npm_params.json` now always matches the one applied: it is asked once for the whole session instead of per file (a question that could be skipped entirely, leaving a header-less session recorded as seconds while milliseconds were applied) and is persisted resolved. Files written by earlier alpha versions are rejected with a message to re-run Step 1. [PR #412](https://github.com/LernerLab/GuPPy/pull/412)
- The Step 3 preprocessing and artifact-removal plots now render on full-length recordings instead of showing empty axes: every sample of every trace was sent to the browser and drawn point by point, so a one-hour session pushed 296 MB over the websocket and left the tab frozen for around 155 seconds before anything appeared, and each recording-site switch or window edit repeated the cost. Traces are now drawn by per-pixel sample density, which keeps the full data server-side and re-aggregates it on zoom, so no sample is dropped and a one-sample artifact spike stays visible at any zoom level. The transients peak view is drawn the same way. [PR #410](https://github.com/LernerLab/GuPPy/pull/410)
- Neurophotometrics timestamps are now on the absolute recording clock, matching the TDT, Doric and NWB readers: the NPM reader no longer re-zeros its continuous channels to their own first sample, nor its event streams to the first channel's first sample, so both keep the acquisition clock. Conversion to seconds is unchanged. [PR #409](https://github.com/LernerLab/GuPPy/pull/409)
- `timeForLightsTurnOn` ("Eliminate first few seconds") now consistently means seconds of warm-up measured from the recording's own start, instead of being read as an absolute time for every format except TDT; the PSTH is anchored on the same instant, so recordings whose acquisition clock does not start at 0 (e.g. an NWB file with a non-zero `starting_time`) no longer index events past the end of the signal and produce empty or misaligned trials. [PR #408](https://github.com/LernerLab/GuPPy/pull/408)
- NWB files using `ndx-events` 0.2 `LabeledEvents` are now readable in environments where the `ndx-events` package is installed: label discovery accepts either the hand-written class's `labels` attribute or the cached-spec class's `data__labels`, instead of assuming the latter and failing with an `AttributeError`. [PR #406](https://github.com/LernerLab/GuPPy/pull/406)
- Step 3 with artifact removal enabled no longer reports success partway through: the progress bar under-counted its work (the artifact-removal pass was not budgeted), so the run was treated as finished after z-scoring and the preprocessing view opened against half-written output. [PR #405](https://github.com/LernerLab/GuPPy/pull/405)
- Doric CSV timestamps are now on the absolute recording clock, matching the `.doric` V1/V6 paths: the CSV reader no longer re-zeros `Time(s)` to start at 0, so both the continuous signal/control streams and the TTL event onsets keep the acquisition clock. [PR #399](https://github.com/LernerLab/GuPPy/pull/399)
- Doric TTL/digital-event channels no longer drop the final event when a recording stops mid-pulse: onset detection now flags every observed low→high transition directly, instead of scanning for gaps between low samples (which could only see a pulse bracketed by a low sample on both sides, so a pulse still high at the last sample was silently dropped). [PR #395](https://github.com/LernerLab/GuPPy/pull/395)
- Removing a large artifact that leaves a short surviving segment now raises a clear error naming the segment, its sample count, and the moving-average filter requirement, instead of the opaque scipy "input vector x must be greater than padlen" message. [PR #392](https://github.com/LernerLab/GuPPy/pull/392)
- Time-based PSTH trial binning ("Time (min)") now assigns each event to the correct time bin: events and bin edges are both on the recording-start basis, so events near a bin boundary (within `timeForLightsTurnOn` seconds) are no longer misbinned. [PR #393](https://github.com/LernerLab/GuPPy/pull/393)
- Cross-correlation between two brain regions no longer crashes when independent artifact removal leaves the regions with a different set of surviving trials: trials are now paired by matching event timestamps (only the events both regions kept are correlated) instead of being lined up by position, so a per-region trial mismatch produces a valid result instead of a `ValueError` that forced an application restart. [PR #388](https://github.com/LernerLab/GuPPy/pull/388)
- CSV and NPM inputs now read successfully regardless of the case of their column names: detection already matched column names case-insensitively, but the read step indexed them by exact lower case, so a valid file with `Timestamps`/`Data`/`Sampling_Rate` (CSV) or `Flags`/`LedState` (NPM) headers passed detection and then crashed with a `KeyError`. [PR #386](https://github.com/LernerLab/GuPPy/pull/386)
- Numeric analysis parameters are now range-checked at config time with an informative notification instead of failing late: `numberOfCores` must be between 1 and the host's core count; `filter_window` and `timeForLightsTurnOn` must be non-negative; `moving_window`, `highAmpFilt`, and `transientsThresh` must be positive; and `nSecPrev` must be strictly less than `nSecPost`. [PR #387](https://github.com/LernerLab/GuPPy/pull/387)
- Signal-control pairing is now captured as explicit data instead of being reverse-engineered from the label string. In Step 1 (Label Stores) each pair's region name is entered once — on the signal — and each control picks which signal it belongs to, so region names may now contain underscores (e.g. `left_hemisphere`) and a control can no longer be silently mismatched to the wrong region. Internally the region is recovered by stripping the known `signal_`/`control_` prefix and pairing is done explicitly, replacing the fragile last-underscore split and alphabetical sort. [PR #385](https://github.com/LernerLab/GuPPy/pull/385)
- Doric signal/control channels with a small fraction of NaN/inf samples (dropped samples, demodulator warmup) are no longer rejected outright: the reader now drops non-finite samples jointly across all signal/control channels (matching the CSV path), keeping control and signal aligned. Channels that are entirely non-finite, empty, or zero-variance still fail fast. [PR #384](https://github.com/LernerLab/GuPPy/pull/384)
- The visualization dashboard's plot Save buttons are now "Save As…" browser downloads that let you choose the filename and location, and they render the current view (respecting live zoom/pan and typed axis limits) instead of a stale earlier render; saving no longer freezes the plot's controls until a page refresh, and render failures now surface as notifications instead of failing silently. [PR #375](https://github.com/LernerLab/GuPPy/pull/375)
- The Heat Map's first and last trial rows no longer render at half height: the Trials (Y) axis now always spans the full cell edges on every render, and its manual axis-limit boxes (which could clip the edge rows) were removed since the axis only encodes trial number. [PR #374](https://github.com/LernerLab/GuPPy/pull/374)

## Improvements
- Added a [Comparing Two Parameter Sets](https://guppy.readthedocs.io/en/latest/tutorials/compare_parameters.html) tutorial: how to give each parameter set its own named run so a second analysis sits beside the first instead of overwriting it, worked through two moving-average filter windows. [PR #438](https://github.com/LernerLab/GuPPy/pull/438)
- Added an [Analyze data streamed from the DANDI Archive](https://guppy.readthedocs.io/en/latest/how-to/analyze-dandi-data.html) how-to guide: the API key prerequisite, selecting assets from a public dandiset, mapping NWB store ids onto signal/control labels via the `FiberPhotometryTable`, and what the pipeline writes locally. [PR #437](https://github.com/LernerLab/GuPPy/pull/437)
- Added a [how-to guide for combining a session split across two data files](https://guppy.readthedocs.io/en/latest/how-to/combine-data.html): setting up matching run names and session-folder naming for **Combine Data?**. [PR #436](https://github.com/LernerLab/GuPPy/pull/436)
- Added a [Testing](https://guppy.readthedocs.io/en/latest/contributing/testing.html) page to the Contributor's Guide: the `tests/unit`/`integration`/`consistency`/`UI` layout, the registered pytest markers, the `stubbed_testing_data/` vs. `testing_data/` distinction, and the headless `GUPPY_BASE_DIR` testing pattern. [PR #432](https://github.com/LernerLab/GuPPy/pull/432)
- Added an [Adding a new acquisition format](https://guppy.readthedocs.io/en/latest/contributing/new_recording_format.html) contributor's guide page: the extractor contract, the end-to-end registration checklist, and what the test suite requires. Also corrects two stale statements it superseded, in [Architecture](https://guppy.readthedocs.io/en/latest/contributing/architecture.html) and [Output data model](https://guppy.readthedocs.io/en/latest/reference/outputs.html). [PR #434](https://github.com/LernerLab/GuPPy/pull/434)
- Added a [Contributor's Guide](https://guppy.readthedocs.io/en/latest/contributing/index.html) section to the documentation, opening with an [Architecture](https://guppy.readthedocs.io/en/latest/contributing/architecture.html) page that maps the seven packages under `src/guppy/`, their entry points, and which orchestration module backs each pipeline step. Replaces the orphaned `docs/architecture.md`, which described the v1-to-v2 refactor and was unreachable from the site nav. [PR #428](https://github.com/LernerLab/GuPPy/pull/428)
- Replaced twelve hand-written copies of the "is this store a photometry channel" test with a single named `is_channel_label` predicate. Three of the copies, in the Doric reader, were case-sensitive, so a store label hand-edited to `Signal_DMS` was read there as a TTL instead of as a channel; all twelve now match case-insensitively, as the rest of the pipeline already did. [PR #427](https://github.com/LernerLab/GuPPy/pull/427)
- Added a [Custom Plots from GuPPy Outputs](https://guppy.readthedocs.io/en/latest/tutorials/custom_plots.html) tutorial: how to open a run folder's HDF5 files in Python and build your own figures from the traces, the PSTH table and the peak/AUC table. [PR #421](https://github.com/LernerLab/GuPPy/pull/421)
- Added an [Output data model](https://guppy.readthedocs.io/en/latest/reference/outputs.html) reference page documenting every file GuPPy writes: the run-folder naming scheme, the `.hdf5` and `.h5` formats, and the datasets or table columns inside each output from Step 1 through the group average. [PR #418](https://github.com/LernerLab/GuPPy/pull/418)
- The Step 3 preprocessing and artifact-removal reviews now stack all five traces for a recording site — control, signal, signal with fitted control, z-score and ΔF/F — on one shared time axis under a single site selector, so zooming any of them zooms them all. The `plot_zScore_dff` parameter, which only gated whether the z-score/ΔF/F traces were shown, has been removed. [PR #414](https://github.com/LernerLab/GuPPy/pull/414)
- The Step 3 preprocessing review no longer shades saved artifact windows over its traces, which implied a removal that has not happened yet — marking and removal are both later steps. Shading remains on the Select Artifact Windows page, where it tracks the periods you are marking. [PR #414](https://github.com/LernerLab/GuPPy/pull/414)
- Trace plots in the preprocessing, artifact-window, artifact-removal and transient-peak views now stretch to the width of the browser tab instead of rendering at a fixed 750 px. [PR #414](https://github.com/LernerLab/GuPPy/pull/414)
- Artifact periods are now marked with form controls — one row of two numeric bounds and a delete button per period, nudgeable with the arrow keys — and an "apply to all recording sites" button copies a period across sites. [PR #413](https://github.com/LernerLab/GuPPy/pull/413)
- The progress bar no longer communicates through `~/pbSteps.txt` and `~/pbError.txt`: now that the pipeline steps run in the main process, they report progress directly over an in-memory channel, which also removes the extra Step 2 poller thread and stops the analysis code from writing to the home directory. [PR #405](https://github.com/LernerLab/GuPPy/pull/405)
- Pipeline Steps 2-4 now run inside the main application process instead of being launched as `python -m` subprocesses: their log output finally reaches `guppy.log` (the child processes never configured logging, so every step's records were silently discarded) and quitting the app no longer leaves orphaned worker processes behind. [PR #404](https://github.com/LernerLab/GuPPy/pull/404)
- Replaced the last matplotlib/TkAgg pop-up windows — the Step 3 (preprocessing & artifact-removal) and Step 4 (transient-peak) plots — with in-browser Panel pages served by the main app; artifact boundaries are now marked by typing good-chunk windows in an editable per-recording-site table (shaded live on the traces) instead of clicking on a native figure. [PR #401](https://github.com/LernerLab/GuPPy/pull/401)
- Removed Tkinter from the codebase: the Neurophotometrics Step 1 prompts (split-events and timestamp column/unit), the only remaining Tk dialogs, are now rendered on the Label Stores page as Panel widgets with a "Confirm NPM configuration" button. [PR #400](https://github.com/LernerLab/GuPPy/pull/400)
- Made the daily CI test suite robust and directory-independent: isolated the live-server UI tests into their own serial run so they no longer race the headless pipeline tests through Panel's global state, registered the Holoviews bokeh backend deterministically, rewrote the progress-poller test to synchronize on output instead of wall-clock sleeps, and restructured shared test fixtures so any `tests/` subdirectory passes standalone. [PR #391](https://github.com/LernerLab/GuPPy/pull/391)
- Renamed the internal "region" concept to "recording_site" across code, error messages, GUI copy, tests, and docs, and added a `recording_site` glossary term to clarify that it means one fiber's signal-plus-optional-isosbestic-control pair (not the anatomical brain region). Naming-only; no on-disk format change (`storesList.csv`, `store_label` values like `signal_DMS`, and HDF5 dataset keys are byte-identical). [PR #390](https://github.com/LernerLab/GuPPy/pull/390)
- Removed global-variable state where avoidable: the Label Stores GUI's store selection now lives on the `StoreLabelingSelector` object instead of a module global shared through button callbacks, and the duplicated headless-mode check is centralized in a single `is_headless()` helper. [PR #380](https://github.com/LernerLab/GuPPy/pull/380)
- Renamed vague variable names (`arr`, `d`, `ts`, `op`, `cols`, suffixed `*_arr`, etc.) throughout `src/guppy/` to descriptive, context-appropriate names; behavior-preserving (consistency suite unchanged). [PR #378](https://github.com/LernerLab/GuPPy/pull/378)
- Deduplicated copy-pasted code: shared timestamp-realignment kernels for artifact-removal and multi-session combining, a shared group-averaging preamble, a single pipeline-step launch helper in the homepage, and shared ndx-fiber-photometry boilerplate across the mock-NWB generators. [PR #377](https://github.com/LernerLab/GuPPy/pull/377)
- Removed commented-out dead code throughout `src/guppy/` and clarified the remaining comments. [PR #376](https://github.com/LernerLab/GuPPy/pull/376)
- Improved plot rendering by setting line_width to 1. [PR #430](https://github.com/LernerLab/GuPPy/pull/430)
- Added a [Development Environment](https://guppy.readthedocs.io/en/latest/contributing/development_environment.html) page to the Contributor's Guide: conda setup, the `dev`/`test`/`docs` dependency groups, running GuPPy from source, headless mode, the pre-commit style stack, and building the docs locally. [PR #431](https://github.com/LernerLab/GuPPy/pull/431)

## Deprecations and Removals
- Removed the `removeArtifacts?` and `removeArtifacts method` controls from the Input Parameters form; the method is now chosen on the Select Artifact Windows page. Both keys remain in `GuPPyParamtersUsed.json`. [PR #413](https://github.com/LernerLab/GuPPy/pull/413)
- Standardized the store/session/run vocabulary across the codebase, GUI, and persisted contracts, and rebranded Step 1 "Save Storenames" to "Label Stores" (see the new [Glossary](https://guppy.readthedocs.io/en/latest/reference/glossary.html)). Breaking, hard cutover with no migration: `GuPPyParamtersUsed.json` keys `folderNames`→`session_folders`, `folderNamesForAvg`→`group_session_folders`, `runName`→`run_name`, `runNamePolicy`→`run_name_policy`, `selectedOutputs`→`selected_runs`, `groupSelectedOutputs`→`group_selected_runs`, and the headless `storenames_map` parameter→`store_id_to_store_label`; `storesList.csv` rows are now `store_id` (row 0) and `store_label` (row 1). Old session output folders are not migrated — re-run Step 1 to regenerate them. [PR #379](https://github.com/LernerLab/GuPPy/pull/379)

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

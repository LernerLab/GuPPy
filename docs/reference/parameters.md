# Input parameter reference

Every parameter the GuPPy GUI exposes, organized to match what you see on screen. The page mirrors the five cards on the homepage (**Input Folder Selection**, **Output Folder Selection**, **Individual Analysis**, **Group Analysis**, **Visualization Parameters**) and the visual sub-groupings inside each card. Each row gives the parameter as it appears in the GUI, a one-line description of what it does, the data type, the default value, and the accepted values or range. Prose paragraphs underneath cover the parameters that need more than a single line.

This is a reference, not a tutorial. If you are running GuPPy for the first time, start with [Your First Analysis](../tutorials/first_analysis.md). The defaults documented here are tuned for a typical 1 kHz dual-wavelength CSV recording and work for the tutorial sample data without modification.

The pipeline-step numbering used in this page matches the steps in [Your First Analysis](../tutorials/first_analysis.md): Step 2 (Load the raw data), Step 3 (Preprocess the signal), Step 4 (Compute the PSTH), Step 5 (Visualize the results).

---

## Input Folder Selection

The first card on the homepage, open by default. Selects the session data the pipeline reads.

*Used by: Step 2 (Load the raw data).*

| Parameter | Description | Type | Default | Options / range |
|-----------|-------------|------|---------|-----------------|
| Data Source | Local-folder mode vs DANDI streaming. | radio | `local` | `local`, `dandi` |
| (file browser) | Session folders to analyze. | list of paths | empty | absolute paths to session directories |
| (DANDI selector) | DANDI assets to materialize as sessions. | dict | `None` (local mode) | per-session mapping of `dandi://` URIs |

**Data Source** picks between selecting local session folders from the file browser (the common case) and streaming NWB sessions directly from DANDI. The browser is hidden when `dandi` is selected and the DANDI selector takes its place.

**File browser** holds the list of session folder paths the pipeline will analyze. Multiple folders are allowed for batch runs; all of them must share a common parent directory or the pipeline raises a validation error before any work starts. The pipeline records the common parent automatically and uses it to anchor output locations; this is not a configurable knob.

**DANDI selector** is populated only in `dandi` mode. Each selected DANDI asset URI is materialized into a session directory under a user-chosen output root, and the pipeline records the URI that backed each session.

---

## Output Folder Selection

The second card on the homepage, collapsed by default. Selects which existing per-session output run the later steps read and write.

*Used by: Steps 2–5 (every step that operates on an existing output run: Load the raw data, Preprocess the signal, Compute the PSTH, Visualize the results).*

| Parameter | Description | Type | Default | Options / range |
|-----------|-------------|------|---------|-----------------|
| (existing-runs browser) | Existing `*_output_*` run directories the later steps act on. | list of paths | empty | one or more `*_output_*` directories, at least one per selected session |

**Existing-runs browser** lists the `*_output_*` directories that already exist for the selected sessions and lets you pick which run each later step acts on. A run directory is created when you configure channels in the Storenames GUI (Step 1); every step from loading the raw data onward then reads and writes the run you select here. Select at least one run per session that has output directories on disk, or the step raises a descriptive error before any work starts. This is a UI selector, not a saved analysis parameter, so it has no internal name in the index below.

---

## Individual Analysis

The largest card on the homepage and the only one open by default. The left column holds a flat list of widgets covering preprocessing, transient detection, output metric selection, and artifact removal. The right column holds four labeled widget boxes for z-score, PSTH, baseline correction, and peak / AUC parameters.

### Compute and batching

*Used by: multiple steps.*

| Parameter | Description | Type | Default | Options / range |
|-----------|-------------|------|---------|-----------------|
| # of cores | How many CPU workers run the pipeline's per-channel steps in parallel. | int | `2` | positive integer, kept below the host's core count |
| Combine Data? | Concatenate two split files into one trace. | bool | `False` | `True`, `False` |

**# of cores** controls the parallelism used during raw-data reading, preprocessing, and PSTH computation, where the pipeline can process channels independently and in parallel. Setting it equal to the number of physical cores on the machine is usually fine; setting it higher than the number of cores does not help.

**Combine Data?** is for the unusual case where one recording session was split across two data files (for example a system that wrote separate files for two halves of a recording). When `True`, the pipeline concatenates the matching channels across both files into a single trace before preprocessing.

### Signal preprocessing

*Used by: Step 3 (Preprocess the signal).*

| Parameter | Description | Type | Default | Options / range |
|-----------|-------------|------|---------|-----------------|
| Isosbestic Control Channel? | Use the 405 nm channel to remove motion artifacts. | bool | `True` | `True`, `False` |
| Eliminate first few seconds | Drop the LED-warmup transient at the start. | int | `1` | non-negative seconds |
| Window for Moving Average filter | Width of the smoothing kernel. | int | `100` | positive integer, in samples (not seconds) |

**Isosbestic Control Channel?** declares whether the recording includes a 405 nm control channel. When `True`, preprocessing fits the control trace to the signal trace by linear regression and subtracts the fitted control to remove motion artifacts and photobleaching that affect both wavelengths equally. When `False`, GuPPy synthesizes a stand-in control channel by fitting an exponential decay curve (`a + b·exp(-x/c)`) to the signal itself and uses it in place of the missing 405 nm channel, so the same regression-and-subtract step still runs. Because a synthetic control carries no motion information, this mode removes the photobleaching trend but not motion artifacts. See the [isosbestic correction explainer](../explanation/isosbestic_correction.md) for the underlying biology and math.

**Eliminate first few seconds** drops this many seconds from the start of every recording. The first second or two of fiber-photometry data is usually contaminated by the bright transient when the LED first turns on; this parameter exists to discard that. Default `1` is conservative.

**Window for Moving Average filter** is the width of the moving-average smoothing kernel applied to both control and signal traces during preprocessing, expressed in **samples**, not seconds. The default `100` is appropriate for recordings sampled around 1 kHz; lower it proportionally for slower acquisition rates (for example use `10` for a 100 Hz recording).

### Output metric selection

*Used by: Step 3 (Preprocess) writes the metrics; Step 4 (Compute the PSTH) and the transient detector read them.*

| Parameter | Description | Type | Default | Options / range |
|-----------|-------------|------|---------|-----------------|
| z_score and/or ΔF/F? (psth) | Metric Step 4 aligns events on. | str | `z_score` | `z_score`, `dff`, `Both` |
| z_score and/or ΔF/F? (transients) | Metric the transient detector operates on. | str | `z_score` | `z_score`, `dff`, `Both` |
| z-score plot and/or ΔF/F plot? | Plot pop-up at the end of Step 3. | str | `None` | `z_score`, `dff`, `Both`, `None` |

**z_score and/or ΔF/F? (psth)** chooses which metric Step 4 uses to align events. Selecting `Both` writes two complete sets of PSTH outputs, one per metric. See the [z-score normalization explainer](../explanation/zscore.md) for what `z_score` is and how it differs from `dff`.

**z_score and/or ΔF/F? (transients)** chooses which metric the transient detector operates on. Same `Both` semantics.

**z-score plot and/or ΔF/F plot?** controls the matplotlib plot that pops up at the end of Step 3. `None` skips the plot; the other options open one window for the chosen metric (or two windows for `Both`).

### Transient detection

*Used by: Step 3 (Preprocess) runs the transient detector on the corrected signal.*

| Parameter | Description | Type | Default | Options / range |
|-----------|-------------|------|---------|-----------------|
| Moving Window for transients detection (s) | Rolling window for the detector. | int | `15` | positive seconds |
| HAFT | Drop excursions above this multiple of MAD before detection. | int | `2` | positive integer |
| TD Thresh | Detection threshold, in multiples of MAD above the median. | int | `3` | positive integer |

**Moving Window for transients detection (s)** is the rolling window used by the detector, in seconds.

**HAFT** (High Amplitude Filtering Threshold) filters out events whose amplitude exceeds this multiple of the trace MAD above the median. The intent is to drop unrealistically large excursions before transient detection, since those are typically motion or recording artifacts that survived preprocessing.

**TD Thresh** (Transients Detection threshold) is the detection threshold proper: local maxima exceeding this multiple of MAD above the median (computed after the high-amplitude filter) are flagged as transients.

### Format-specific (Neurophotometrics)

*Used by: Step 2 (Load the raw data) when the recording is an NPM CSV without `Flags` or `LedState` columns.*

| Parameter | Description | Type | Default | Options / range |
|-----------|-------------|------|---------|-----------------|
| Number of channels (Neurophotometrics only) | Manual fallback for stripped NPM CSVs. | int | `2` | positive integer |

**Number of channels (Neurophotometrics only)** is read only when the data is a Neurophotometrics recording whose CSV does not include the `Flags` or `LedState` column. Modern NPM recordings include those columns and GuPPy can infer channel structure automatically; this parameter is the manual fallback for older or stripped-down NPM CSVs. Other format-specific behavior (TDT epoc handling, Doric channel selection, NWB recording-extractor selection) is handled by the recording extractors at read time and does not require user-set parameters.

### Artifact removal

*Used by: Step 3 (Preprocess) runs the interactive removal flow when enabled.*

| Parameter | Description | Type | Default | Options / range |
|-----------|-------------|------|---------|-----------------|
| removeArtifacts? | Enable the interactive removal flow. | bool | `False` | `True`, `False` |
| removeArtifacts method | How dropped chunks are handled. | str | `concatenate` | `concatenate`, `replace with NaN` |

**removeArtifacts?** enables the manual artifact-removal step in Step 3: when `True`, GuPPy presents an interactive plot during preprocessing and lets you select bad chunks to drop. When `False`, no chunks are removed.

**removeArtifacts method** chooses how dropped chunks are handled. `concatenate` removes the bad sections and stitches the surviving good sections together (so the resulting trace is shorter than the input). `replace with NaN` keeps the trace at its original length but masks the dropped samples with NaN, which downstream code treats as missing.

### Z-score Parameters

*Used by: Step 3 (Preprocess) writes the z-score files.*

| Parameter | Description | Type | Default | Options / range |
|-----------|-------------|------|---------|-----------------|
| z-score computation Method | Normalisation formula. | str | `standard z-score` | `standard z-score`, `baseline z-score`, `modified z-score` |
| Baseline Window Start Time (s) | Start of the baseline window for `baseline z-score`. | int | `0` | seconds, must be `< Baseline Window End Time` and within the signal's recorded timespan |
| Baseline Window End Time (s) | End of the baseline window for `baseline z-score`. | int | `0` | seconds, must be `> Baseline Window Start Time` and within the signal's recorded timespan |

**z-score computation Method** picks the normalization formula. `standard z-score` uses the mean and standard deviation across the entire trace. `baseline z-score` uses the mean and standard deviation of a user-specified window (the baseline window parameters below). `modified z-score` uses the median and median absolute deviation, which is robust to outliers and to long tonic shifts. See the [z-score normalization explainer](../explanation/zscore.md) for the formulas and trade-offs.

**Baseline Window Start Time (s)** and **Baseline Window End Time (s)** define the baseline window in seconds. Both default to `0`, which is the sentinel meaning "no window set"; you only need non-zero values when the z-score method is `baseline z-score`. The validator enforces start < end, both finite numbers, and both within the signal's actual timespan, surfacing a descriptive error if any of those conditions fail.

### PSTH Parameters

*Used by: Step 4 (Compute the PSTH).*

See the [PSTH explainer](../explanation/psth.md) for what these parameters configure (the peri-event window, event-timestamp deduplication, binning across events) and the reasoning behind the default values.

| Parameter | Description | Type | Default | Options / range |
|-----------|-------------|------|---------|-----------------|
| Seconds before 0 | Pre-event window edge. | int | `-10` | typically negative; defines the pre-event window |
| Seconds after 0 | Post-event window edge. | int | `20` | typically positive; defines the post-event window |
| Compute Cross-correlation | Cross-correlate PSTHs across regions. | bool | `False` | `True`, `False`. Requires at least two distinct signal regions; raises `ValueError` otherwise. |
| Time Interval (s) | Minimum spacing for accepted event timestamps. | int | `2` | seconds; bursts of event timestamps closer than this are discarded as duplicates |
| Bin PSTH trials | Binning unit (time vs count). | str | `Time (min)` | `Time (min)`, `# of trials` |
| Time(min) / # of trials for binning | Bin size; `0` disables binning. | int | `0` | `0` disables binning; positive values use the unit selected above |

**Seconds before 0** and **Seconds after 0** define the peri-event window. Defaults give a 30-second window from 10 s before to 20 s after each event timestamp.

**Compute Cross-correlation** turns on cross-correlation between PSTHs of two distinct signal regions, useful for detecting coordinated activity between brain areas. The pipeline raises a descriptive `ValueError` when this is `True` but only one signal region is configured. See the [cross-correlation explainer](../explanation/cross_correlation.md) for interpretation guidance.

**Time Interval (s)** suppresses bursts of event timestamps. If two event timestamps in the input are closer than this number of seconds, the second one is dropped before PSTH alignment, preventing double-counted overlapping windows.

**Bin PSTH trials** and **Time(min) / # of trials for binning** together control binning of the resulting PSTH. With `Bin PSTH trials = "Time (min)"` and the bin size set to `5`, the PSTH is averaged into 5-minute bins along the trial axis; with `# of trials` and bin size `10`, bins of 10 trials each. Setting the bin size to `0` disables binning entirely.

### Baseline Parameters

*Used by: Step 4 (Compute the PSTH) when correcting per-event offsets.*

| Parameter | Description | Type | Default | Options / range |
|-----------|-------------|------|---------|-----------------|
| Baseline Correction Start time | Start of the per-event baseline subtraction window. | int | `-5` | seconds, within `[Seconds before 0, Seconds after 0]` |
| Baseline Correction End time | End of the per-event baseline subtraction window. | int | `0` | seconds, within `[Seconds before 0, Seconds after 0]` and `> Baseline Correction Start time` |

These two parameters define a baseline window inside the PSTH window. The mean of each event-aligned trace within this baseline is subtracted from that trace before averaging, removing per-event offsets so that all event-aligned traces are centered on the same baseline.

Set both to `0` to disable baseline correction. If the first event timestamp in the recording is closer to the start of the trace than `Baseline Correction Start time - Seconds before 0` seconds, that event is rejected because its baseline window would fall outside the recording.

### Peak and AUC Parameters

*Used by: Step 4 (Compute the PSTH) computes peak amplitude and area under the curve for each window.*

| Parameter | Description | Type | Default | Options / range |
|-----------|-------------|------|---------|-----------------|
| Peak Start time | Start times for the peak/AUC windows. | list of int | `[-5, 0, 5]` (rows 1-3 of the table; rows 4-10 are NaN) | one or more start times in seconds, within `[Seconds before 0, Seconds after 0]` |
| Peak End time | End times paired with the starts. | list of int | `[0, 3, 10]` (rows 1-3 of the table; rows 4-10 are NaN) | one or more end times in seconds, paired with starts |

The peak / AUC widget is a small table with rows of (start, end) pairs. Each row defines a window inside the PSTH within which GuPPy computes the peak amplitude and area under the curve of the trial-mean trace. Multiple rows let you measure the same PSTH across multiple windows in a single run (for example, an early `[-5, 0]` baseline window, an immediate post-event `[0, 3]` window, and a later `[5, 10]` window). The tabulator widget accepts up to ten rows; rows whose start or end value is NaN are ignored.

---

## Group Analysis

Collapsed by default on the homepage. Configures cross-session averaging.

*Used by: Step 4 (Compute the PSTH) writes the averages when enabled.*

| Parameter | Description | Type | Default | Options / range |
|-----------|-------------|------|---------|-----------------|
| (file browser) | Session folders to include in the cross-session average. | list of paths | empty | absolute paths to session directories |
| Average Group? | Write averaged outputs to `average/`. | bool | `False` | `True`, `False` |

**File browser** is the list of session folders to include in the cross-session average. Distinct from the Individual-Analysis browser so you can run individual analyses and group analyses against different folder sets in the same configuration.

**Average Group?** must be `True` for Step 4 to write averaged outputs into the `average/` directory. If `False`, PSTH outputs are per-session only.

---

## Visualization Parameters

Collapsed by default on the homepage. Configures Step 5.

*Used by: Step 5 (Visualize the results).*

| Parameter | Description | Type | Default | Options / range |
|-----------|-------------|------|---------|-----------------|
| z-score or ΔF/F? (for visualization) | Which metric the Visualization GUI plots. | str | `z_score` | `z_score`, `dff` |
| Visualize Average Results? | Show cross-session averages instead of per-session. | bool | `False` | `True`, `False` |

**z-score or ΔF/F? (for visualization)** picks which metric the Visualization GUI plots. Must match a metric that Step 3 actually wrote: if you ran preprocessing with the PSTH metric set to `z_score` and try to visualize `dff`, GuPPy raises a descriptive error pointing at the missing files.

**Visualize Average Results?** decides whether the Visualization GUI shows individual-session results or the cross-session averages produced by Step 4 with `Average Group?` set to `True`. Single-session analyses should leave this `False`.

---

## Internal name index

This index is for readers who arrive with an internal parameter name in hand and need to find the corresponding GUI parameter. That happens in four situations:

- **Reproducing or auditing a past analysis** by reading the `GuPPyParamtersUsed.json` snapshot that GuPPy writes after every run; the JSON is keyed by internal names. Selecting a finished output run in the Individual-Analysis output picker also reloads this snapshot back into the form, so you can resume a run without the defaults silently overwriting the parameters the earlier steps used.
- **Writing a headless or scripted analysis** against the API in `src/guppy/testing/api.py`, which takes a dict keyed by these names.
- **Debugging a validator or pipeline error**, since error messages cite the internal name (for example `baselineWindowEnd=120 exceeds signal duration 90.5s`).
- **Reading or contributing to the source code**, where parameter accesses go through the internal names.

The table is sorted alphabetically by internal name. Each row links to the section above where the parameter is documented in full. Internal-only keys that have no GUI counterpart (`abspath`) are listed too.

| Internal name | Parameter | Section |
|---------------|-----------|---------|
| `abspath` | (auto-derived; not user-set) | [Input Folder Selection](#input-folder-selection) |
| `artifactsRemovalMethod` | removeArtifacts method | [Artifact removal](#artifact-removal) |
| `averageForGroup` | Average Group? | [Group Analysis](#group-analysis) |
| `baselineCorrectionEnd` | Baseline Correction End time | [Baseline Parameters](#baseline-parameters) |
| `baselineCorrectionStart` | Baseline Correction Start time | [Baseline Parameters](#baseline-parameters) |
| `baselineWindowEnd` | Baseline Window End Time (s) | [Z-score Parameters](#z-score-parameters) |
| `baselineWindowStart` | Baseline Window Start Time (s) | [Z-score Parameters](#z-score-parameters) |
| `bin_psth_trials` | Time(min) / # of trials for binning | [PSTH Parameters](#psth-parameters) |
| `combine_data` | Combine Data? | [Compute and batching](#compute-and-batching) |
| `computeCorr` | Compute Cross-correlation | [PSTH Parameters](#psth-parameters) |
| `dandi_uri_map` | (DANDI selector) | [Input Folder Selection](#input-folder-selection) |
| `filter_window` | Window for Moving Average filter | [Signal preprocessing](#signal-preprocessing) |
| `folderNames` | (file browser, Input Folder Selection) | [Input Folder Selection](#input-folder-selection) |
| `folderNamesForAvg` | (file browser, Group Analysis) | [Group Analysis](#group-analysis) |
| `highAmpFilt` | HAFT | [Transient detection](#transient-detection) |
| `isosbestic_control` | Isosbestic Control Channel? | [Signal preprocessing](#signal-preprocessing) |
| `mode` | Data Source | [Input Folder Selection](#input-folder-selection) |
| `moving_window` | Moving Window for transients detection (s) | [Transient detection](#transient-detection) |
| `nSecPost` | Seconds after 0 | [PSTH Parameters](#psth-parameters) |
| `nSecPrev` | Seconds before 0 | [PSTH Parameters](#psth-parameters) |
| `noChannels` | Number of channels (Neurophotometrics only) | [Format-specific (Neurophotometrics)](#format-specific-neurophotometrics) |
| `numberOfCores` | # of cores | [Compute and batching](#compute-and-batching) |
| `peak_endPoint` | Peak End time | [Peak and AUC Parameters](#peak-and-auc-parameters) |
| `peak_startPoint` | Peak Start time | [Peak and AUC Parameters](#peak-and-auc-parameters) |
| `plot_zScore_dff` | z-score plot and/or ΔF/F plot? | [Output metric selection](#output-metric-selection) |
| `removeArtifacts` | removeArtifacts? | [Artifact removal](#artifact-removal) |
| `selectForComputePsth` | z_score and/or ΔF/F? (psth) | [Output metric selection](#output-metric-selection) |
| `selectForTransientsComputation` | z_score and/or ΔF/F? (transients) | [Output metric selection](#output-metric-selection) |
| `timeForLightsTurnOn` | Eliminate first few seconds | [Signal preprocessing](#signal-preprocessing) |
| `timeInterval` | Time Interval (s) | [PSTH Parameters](#psth-parameters) |
| `transientsThresh` | TD Thresh | [Transient detection](#transient-detection) |
| `use_time_or_trials` | Bin PSTH trials | [PSTH Parameters](#psth-parameters) |
| `visualize_zscore_or_dff` | z-score or ΔF/F? (for visualization) | [Visualization Parameters](#visualization-parameters) |
| `visualizeAverageResults` | Visualize Average Results? | [Visualization Parameters](#visualization-parameters) |
| `zscore_method` | z-score computation Method | [Z-score Parameters](#z-score-parameters) |

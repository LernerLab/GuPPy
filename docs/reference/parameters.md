# Input parameter reference

Every parameter the GuPPy GUI exposes, organized to match what you see on screen. The page mirrors the four cards on the homepage (**Input Folder Selection**, **Output Folder Selection**, **Parameter Selection**, **Group Output Folder Selection**) and the titled sections inside each card. The GUI answers what a single parameter does, through the **?** beside each control; this page is where they are documented together, in relation to each other and in more detail. Each row gives the parameter as it appears in the GUI, a one-line description of what it does, the data type, the default value, and the accepted values or range. Prose paragraphs underneath cover the parameters that need more than a single line. If this is your first time using GuPPy, follow the [Your First Analysis](../tutorials/first_analysis.md) tutorial instead.

The pipeline-step numbering used in this page matches the steps in [Your First Analysis](../tutorials/first_analysis.md): Step 2 (Load the raw data), Step 3 (Preprocess the signal), Step 4 (Compute the PSTH), Step 5 (Visualize the results).

---

## Input Folder Selection

The first card on the homepage, open by default. Selects the session data the pipeline reads.

*Used by: Step 2 (Load the raw data); **Combine Data?** is also read by Steps 3-7.*

| Parameter | Description | Type | Default | Options / range |
|-----------|-------------|------|---------|-----------------|
| Data Source | Local-folder mode vs DANDI streaming. | radio | `local` | `local`, `dandi` |
| (file browser) | Session folders to analyze. | list of paths | empty | absolute paths to session directories |
| (DANDI selector) | DANDI assets to materialize as sessions. | dict | `None` (local mode) | per-session mapping of `dandi://` URIs |
| Combine Data? | Concatenate two split files into one trace. | bool | `False` | `True`, `False` |

**Data Source** picks between selecting local session folders from the file browser (the common case) and streaming NWB sessions directly from DANDI. The browser is hidden when `dandi` is selected and the DANDI selector takes its place. See [Analyze data streamed from the DANDI Archive](../how-to/analyze-dandi-data.md) for the DANDI workflow.

**File browser** holds the list of session folder paths the pipeline will analyze. Multiple folders are allowed for batch runs, and they do not have to sit side by side: sessions kept in different folders can be analyzed together in a single run. Each session's results are written inside that session's own folder. The pipeline records the directory that contains all of the selected sessions automatically; this is not a configurable knob.

**Combine Data?** is for the unusual case where one recording session was split across two data files (for example a system that wrote separate files for two halves of a recording). When `True`, the pipeline concatenates the matching channels across both files into a single trace before preprocessing.

**DANDI selector** is populated only in `dandi` mode. Each selected DANDI asset URI is materialized into a session directory under a user-chosen output root, and the pipeline records the URI that backed each session.

---

## Output Folder Selection

The second card on the homepage, collapsed by default. Says where GuPPy writes its run folders, and selects which existing per-session output run the later steps read and write.

*Used by: Step 1 (which creates the run folder) and Steps 2–5 (every step that operates on an existing output run: Load the raw data, Preprocess the signal, Compute the PSTH, Visualize the results).*

| Parameter | Description | Type | Default | Options / range |
|-----------|-------------|------|---------|-----------------|
| Output Location | Whether run folders are collected in one base directory or written inside each session folder. | choice | separate output directory | `separate output directory`, `inside each session folder` |
| (output base directory browser) | The directory the run folders are written into. | path | empty | any directory that is not itself a selected session |
| Run name(s) for all sessions | Run names to select across every selected session at once. | list of run names | empty | run names found in any selected session |
| (existing-runs browser) | Existing `*_output_*` run directories the later steps act on. | list of paths | empty | one or more `*_output_*` directories, at least one per selected session |

**Output Location** decides where every run folder goes. On *separate output directory* — the default — no analysis output is written into your session folders, which keeps raw data immutable and lets it live on a read-only volume, be archived, or be checksummed as a unit. Leave the browser under it empty and each session's runs go into a `guppy_output` directory beside that session; because that is worked out one session at a time, a session's runs stay put however you change the selection between steps. Pick a directory in the browser instead and every selected session's runs go there together. Since a run folder is named `<session folder name>_output_<run name>`, sessions writing into the same base directory need distinct folder names, and GuPPy refuses the run rather than letting two sessions write over each other.

*inside each session folder* restores the pre-2.0.0-beta4 layout, where each run folder is created inside the session folder it was analyzed from. Analyses made with an earlier version of GuPPy are only reachable under this setting.

**Existing-runs browser** lists the `*_output_*` directories that already exist for the selected sessions and lets you pick which run each later step acts on. A run directory is created when you configure channels in the Label Stores GUI (Step 1); every step from loading the raw data onward then reads and writes the run you select here.

**Run name(s) for all sessions** reaches those same directories by name instead of by browsing to them, so one choice covers a whole batch. Step 1 names each run: the run directory `sample_data_csv_1_output_1` has the run name `1`. Naming a run selects it in every selected session that has one by that name, and removing the name deselects exactly those — directories you ticked in the browser yourself are left alone either way. The picker offers every run name found in *any* selected session, so a name only some of them have still works; the sessions without it are yours to fill in from the browser.

Changing which sessions are selected does not discard these choices: sessions that stay selected keep the runs you picked for them, and a session you add picks up the run names currently named above.

Steps 2-4 need at least one run per session that has output directories on disk, or they raise a descriptive error before any work starts. Step 5 is the exception: it can visualize groups on their own, so it accepts an empty selection here as long as a group is selected. This is a UI selector, not a saved analysis parameter, so it has no internal name in the index below. To analyze one session under two different parameter sets and compare the results, see [Comparing Two Parameter Sets](../tutorials/compare_parameters.md).

---

## Parameter Selection

The largest card on the homepage, collapsed by default (only Input Folder Selection is open on launch). It holds one column of titled sections, each named for the operation its parameters configure and each stating the pipeline steps that read it, ordered by the step that consumes them. The card is not specific to a single analysis level: several of its sections are read by the Group Analysis step as well.

### Parallel Execution

*Used by: Steps 2 and 4, and the Group Analysis step.*

| Parameter | Description | Type | Default | Options / range |
|-----------|-------------|------|---------|-----------------|
| # of cores | How many CPU workers run the pipeline's per-channel steps in parallel. | int | `2` | positive integer, kept below the host's core count |

**# of cores** controls the parallelism used during raw-data reading, preprocessing, and PSTH computation, where the pipeline can process channels independently and in parallel. Setting it equal to the number of physical cores on the machine is usually fine; setting it higher than the number of cores does not help.

### Control Channel Fitting

*Used by: Step 3 (Preprocess the signal).*

| Parameter | Description | Type | Default | Options / range |
|-----------|-------------|------|---------|-----------------|
| Isosbestic Control Channel? | Use the isosbestic control channel to remove motion artifacts. | bool | `True` | `True`, `False` |
| Control Channel Fitting Method | How the control channel is fit to the signal before subtraction. | str | `IRWLS` | `IRWLS`, `OLS` |
| Control Fit Window | Whether the control-to-signal fit is estimated over the full trace or a baseline epoch. | str | `full trace` | `full trace`, `baseline epoch` |
| Control Fit Window Start Time (s) | Start of the baseline epoch used to estimate the fit. | int | `0` | seconds, must be `< Control Fit Window End Time` and within the signal's recorded timespan |
| Control Fit Window End Time (s) | End of the baseline epoch used to estimate the fit. | int | `0` | seconds, must be `> Control Fit Window Start Time` and within the signal's recorded timespan |
| Photobleaching Detrend? | Add an exponential decay term to the control fit. | bool | `False` | `True`, `False` |

**Isosbestic Control Channel?** declares whether the recording includes an isosbestic control channel. When `True`, preprocessing fits the isosbestic control channel to the signal trace and subtracts the fitted control to remove motion artifacts and photobleaching that affect both wavelengths equally. When `False`, GuPPy synthesizes a stand-in control channel by fitting an exponential decay curve (`a + b·exp(-x/c)`) to the signal itself, then runs the same fit-and-subtract step using this synthetic trace as the control channel that gets fitted and subtracted. Because a synthetic control carries no motion information, this mode removes the photobleaching trend but not motion artifacts. See the [isosbestic correction explainer](../explanation/isosbestic_correction.md) for the underlying biology and math.

**Control Channel Fitting Method** chooses how the control channel is rescaled onto the signal before subtraction. `IRWLS` (the default) uses Iteratively Re-Weighted Least Squares with a Tukey bisquare weighting, a robust regression that down-weights outlier samples (transients, brief wavelength-dependent artifacts) so they do not distort the fit. It is equivalent to ordinary least squares on clean data and more reliable when outliers are present, so it is almost always equal to or better than a plain least-squares fit. `OLS` selects ordinary least-squares regression instead. See the [isosbestic correction explainer](../explanation/isosbestic_correction.md) for details.

**Control Fit Window** chooses which part of the recording the control-to-signal fit is estimated from. `full trace` (the default) estimates the fit coefficients over the whole recording, matching prior behavior. `baseline epoch` estimates the coefficients from only the window set by **Control Fit Window Start Time (s)** and **Control Fit Window End Time (s)**, then applies those fixed coefficients across the entire recording. Use it when a sustained step-change in the signal — such as a drug injection — would otherwise distort a full-trace fit: fitting on the clean pre-injection window keeps the coefficients stable while the measured control channel continues to correct motion and photobleaching after the injection. This mode requires an isosbestic control channel (**Isosbestic Control Channel?** set to `True`). Both time bounds are in seconds; the validator enforces start < end and that both fall within the signal's recorded timespan, and it reports an error if the window contains no data after artifact removal.

**Photobleaching Detrend?** extends the control fit with an exponential decay term, for the photobleaching the isosbestic control channel does not see. Fitting and subtracting the control cancels the bleaching the two wavelengths share, but the indicator bleaches by its own kinetics as well, and no rescaling of the control can remove that part — on long recordings it survives into the corrected ΔF/F as a slow drift, which confounds any comparison between an early part of the session and a late one. When `True`, the fitted baseline becomes `slope·control + intercept + b·exp(-x/c)` instead of `slope·control + intercept`, and ΔF/F is computed against that. The decay term is part of the fit, so it appears in `cntrl_sig_fit_<recording site>` and in the preprocessing review page. Its time constant is held within the length of the recording, since a decay slower than the recording cannot be measured from it. This parameter requires an isosbestic control channel (**Isosbestic Control Channel?** set to `True`), and requires **Control Channel Fitting Method** to be `OLS` — the decay term makes the fit nonlinear, and the nonlinear fit has no robust variant.

### Signal Filtering

*Used by: Step 3 (Preprocess the signal); the moving-average window is also read by the Group Analysis step when it computes PSTH significance.*

| Parameter | Description | Type | Default | Options / range |
|-----------|-------------|------|---------|-----------------|
| Eliminate first few seconds | Drop the LED-warmup transient at the start. | int | `1` | non-negative seconds |
| Window for Moving Average filter | Width of the smoothing kernel. | int | `100` | positive integer, in samples (not seconds) |

**Eliminate first few seconds** drops this many seconds from the start of every recording. The first second or two of fiber-photometry data is usually contaminated by the bright transient when the LED first turns on; this parameter exists to discard that. Default `1` is conservative. It applies the same trim to every session in the batch; to cut deeper into a single recording, mark its opening as an artifact period instead — see [Remove artifacts from a recording](../how-to/artifact-removal.md#trimming-extra-time-from-the-start).

**Window for Moving Average filter** is the width of the moving-average smoothing kernel applied to both control and signal traces during preprocessing, expressed in **samples**, not seconds. The default `100` is appropriate for recordings sampled around 1 kHz; lower it proportionally for slower acquisition rates (for example use `10` for a 100 Hz recording).

### Z-score Normalization

*Used by: Step 3 (Preprocess) writes the z-score files.*

| Parameter | Description | Type | Default | Options / range |
|-----------|-------------|------|---------|-----------------|
| z-score computation Method | Normalisation formula. | str | `standard z-score` | `standard z-score`, `baseline z-score`, `modified z-score` |
| Baseline Window Start Time (s) | Start of the baseline window for `baseline z-score`. | int | `0` | seconds, must be `< Baseline Window End Time` and within the signal's recorded timespan |
| Baseline Window End Time (s) | End of the baseline window for `baseline z-score`. | int | `0` | seconds, must be `> Baseline Window Start Time` and within the signal's recorded timespan |

**z-score computation Method** picks the normalization formula. `standard z-score` uses the mean and standard deviation across the entire trace. `baseline z-score` uses the mean and standard deviation of a user-specified window (the baseline window parameters below). `modified z-score` uses the median and median absolute deviation, which is robust to outliers and to long tonic shifts. See the [z-score normalization explainer](../explanation/zscore.md) for the formulas and trade-offs.

**Baseline Window Start Time (s)** and **Baseline Window End Time (s)** define the baseline window in seconds. Both default to `0`, which is the sentinel meaning "no window set"; you only need non-zero values when the z-score method is `baseline z-score`. The validator enforces start < end, both finite numbers, and both within the signal's actual timespan, surfacing a descriptive error if any of those conditions fail.

### PSTH Computation

*Used by: Step 4 (Compute the PSTH); the metric selector is also read by the Group Analysis step.*

See the [PSTH explainer](../explanation/psth.md) for what these parameters configure (the peri-event window, event-timestamp deduplication, binning across events) and the reasoning behind the default values.

| Parameter | Description | Type | Default | Options / range |
|-----------|-------------|------|---------|-----------------|
| z_score and/or ΔF/F? (psth) | Metric Step 4 aligns events on. | str | `z_score` | `z_score`, `dff`, `Both` |
| Seconds before 0 | Pre-event window edge. | int | `-10` | typically negative; defines the pre-event window |
| Seconds after 0 | Post-event window edge. | int | `20` | typically positive; defines the post-event window |
| Compute Cross-correlation | Cross-correlate PSTHs across recording sites. | bool | `False` | `True`, `False`. Requires at least two distinct signal recording sites; raises `ValueError` otherwise. |
| Time Interval (s) | Minimum spacing for accepted event timestamps. | int | `2` | seconds; bursts of event timestamps closer than this are discarded as duplicates |
| Bin PSTH trials | Binning unit (time vs count). | str | `Time (min)` | `Time (min)`, `# of trials` |
| Time(min) / # of trials for binning | Bin size; `0` disables binning. | int | `0` | `0` disables binning; positive values use the unit selected above |
| Baseline Correction Start time | Start of the per-event baseline subtraction window. | int | `-5` | seconds, within `[Seconds before 0, Seconds after 0]` |
| Baseline Correction End time | End of the per-event baseline subtraction window. | int | `0` | seconds, within `[Seconds before 0, Seconds after 0]` and `> Baseline Correction Start time` |

**z_score and/or ΔF/F? (psth)** chooses which metric Step 4 uses to align events. Selecting `Both` writes two complete sets of PSTH outputs, one per metric. See the [z-score normalization explainer](../explanation/zscore.md) for what `z_score` is and how it differs from `dff`.

**Seconds before 0** and **Seconds after 0** define the peri-event window. Defaults give a 30-second window from 10 s before to 20 s after each event timestamp.

**Compute Cross-correlation** turns on cross-correlation between PSTHs of two distinct signal recording sites, useful for detecting coordinated activity between brain areas. The pipeline raises a descriptive `ValueError` when this is `True` but only one signal recording site is configured. See the [cross-correlation explainer](../explanation/cross_correlation.md) for interpretation guidance.

**Time Interval (s)** suppresses bursts of event timestamps. If two event timestamps in the input are closer than this number of seconds, the second one is dropped before PSTH alignment, preventing double-counted overlapping windows.

**Bin PSTH trials** and **Time(min) / # of trials for binning** together control binning of the resulting PSTH. With `Bin PSTH trials = "Time (min)"` and the bin size set to `5`, the PSTH is averaged into 5-minute bins along the trial axis; with `# of trials` and bin size `10`, bins of 10 trials each. Setting the bin size to `0` disables binning entirely.

**Baseline Correction Start time** and **Baseline Correction End time** define a baseline window inside the PSTH window. The mean of each event-aligned trace within this baseline is subtracted from that trace before averaging, removing per-event offsets so that all event-aligned traces are centered on the same baseline.

Set both to `0` to disable baseline correction. If the first event timestamp in the recording is closer to the start of the trace than `Baseline Correction Start time - Seconds before 0` seconds, that event is rejected because its baseline window would fall outside the recording.

### Peak and AUC Measurement

*Used by: Step 4 (Compute the PSTH) computes peak amplitude and area under the curve for each window.*

| Parameter | Description | Type | Default | Options / range |
|-----------|-------------|------|---------|-----------------|
| Peak Start time | Start times for the peak/AUC windows. | list of int | `[-5, 0, 5]` (rows 1-3 of the table; rows 4-10 are NaN) | one or more start times in seconds, within `[Seconds before 0, Seconds after 0]` |
| Peak End time | End times paired with the starts. | list of int | `[0, 3, 10]` (rows 1-3 of the table; rows 4-10 are NaN) | one or more end times in seconds, paired with starts |
| AUC Units | Time unit the area under the curve is integrated against. | str | `samples` | `samples`, `seconds` |

The peak / AUC widget is a small table with rows of (start, end) pairs. Each row defines a window inside the PSTH within which GuPPy computes the peak amplitude and area under the curve of the trial-mean trace. Multiple rows let you measure the same PSTH across multiple windows in a single run (for example, an early `[-5, 0]` baseline window, an immediate post-event `[0, 3]` window, and a later `[5, 10]` window). The tabulator widget accepts up to ten rows; rows whose start or end value is NaN are ignored.

**AUC Units** controls the spacing used to integrate each window. `seconds` reports the area in z-score × seconds (or ΔF/F × seconds), the unit commonly reported in the literature. `samples` integrates with one-sample spacing instead, so the same response reads larger the faster it was sampled; a 1017 Hz recording gives a value roughly 1000× that of a 1 Hz recording. The choice applies to every `area_*` column in the `peak_AUC_*` outputs and is recorded in `GuPPyParamtersUsed.json`.

---

### Transient Detection

*Used by: Step 4 (Compute the PSTH) runs the transient detector on the corrected signal; Step 5 and the Group Analysis step read the metric and the transients-as-events switch.*

| Parameter | Description | Type | Default | Options / range |
|-----------|-------------|------|---------|-----------------|
| z_score and/or ΔF/F? (transients) | Metric the transient detector operates on. | str | `z_score` | `z_score`, `dff`, `Both` |
| Use Transients as Events? | Use each recording site's detected transients as its event timestamps. | bool | `False` | `True`, `False` |
| Moving Window for transients detection (s) | Rolling window for the detector. | int | `15` | positive seconds |
| HAFT | Drop excursions above this multiple of MAD before detection. | int | `2` | positive integer |
| TD Thresh | Detection threshold, in multiples of MAD above the median. | int | `3` | positive integer |

**z_score and/or ΔF/F? (transients)** chooses which metric the transient detector operates on. Same `Both` semantics.

**Moving Window for transients detection (s)** is the rolling window used by the detector, in seconds.

**HAFT** (High Amplitude Filtering Threshold) filters out events whose amplitude exceeds this multiple of the trace MAD above the median. The intent is to drop unrealistically large excursions before transient detection, since those are typically motion or recording artifacts that survived preprocessing.

**TD Thresh** (Transients Detection threshold) is the detection threshold proper: local maxima exceeding this multiple of MAD above the median (computed after the high-amplitude filter) are flagged as transients.

**Use Transients as Events?** is for spontaneous activity, where there is no external event to align to. With it on, the transients detected in each recording site become that recording site's event timestamps, and the PSTH, peak and AUC are computed against them exactly as they would be against a TTL train — no manual export and re-import of an artificial TTL file. One event is produced per metric the detector runs on, named `transients_z_score` and/or `transients_dff` depending on **z_score and/or ΔF/F? (transients)**. The peak and AUC windows are then measured relative to each transient's peak, and **Time Interval (s)** in [PSTH Computation](#psth-computation) de-bursts the transient train the same way it de-bursts a TTL train. Cross-correlation is skipped for these events, since each recording site has its own transient times and the two sites therefore share no trials.

### Metric Binning

*Used by: Step 4 (Compute the PSTH), after transient detection.*

| Parameter | Description | Type | Default | Options / range |
|-----------|-------------|------|---------|-----------------|
| Compute Binned Metrics? | Reduce the whole session to one row per fixed time bin. | bool | `False` | `True` / `False` |
| Bin Width (s) | Width of those bins. | int | `120` | positive seconds |

**Compute Binned Metrics?** divides the whole recording into equal time bins and reports, for each bin and each recording site, the mean z-score, the mean ΔF/F and the number of transients detected in it. Use it when you want to relate the signal to something measured on its own fixed schedule across the session — a behavior score rated every couple of minutes, say — rather than to discrete events. The results are written as `binned_metrics_<site>.csv` and `.h5`, and shown on the **Binned** tab in Step 5. Off by default; see the [output data model](outputs.md#step-4-compute-the-psth) for the exact table.

**Bin Width (s)** is how wide each bin is. Bins start at the first corrected timestamp; the last bin is kept even when the session does not divide evenly, so it can be shorter than the rest. Whole seconds only.

If the thing measured on its own schedule was recorded as data rather than watched by eye, label it as a **behavioral covariate** in Step 1 and GuPPy will bin it onto the same bins and correlate it against every per-bin metric. That needs no additional parameter — labeling the store is what turns it on — but it does require **Compute Binned Metrics?** to be enabled. See [Correlate a behavioral covariate](../how-to/correlate-behavioral-covariates.md).

### Significance Testing

*Used by: Step 4 (Compute the PSTH), and Group Analysis.*

| Parameter | Description | Type | Default | Options / range |
|-----------|-------------|------|---------|-----------------|
| Compute PSTH Significance? | Bootstrap which stretches of the PSTH window are significant. | bool | `False` | `True` / `False` |
| Significance Level (alpha) | Two-sided threshold the confidence interval is computed at. | float | `0.05` | strictly between 0 and 1 |
| Bootstrap Resamples | How many resamples each interval is built from. | int | `1000` | positive |
| Event A / Event B | Pairs of events to compare against each other. | table | one blank row | event labels from Step 1 |

**Compute PSTH Significance?** turns on bootstrap significance testing. With it on, every event is tested against zero — "is there a response at all, and when?" — for each recording site and metric. The results are written into `psth_significance_output/` and shown on the **Significance** tab in Step 5. Off by default, since the test adds several seconds per comparison. See the [explainer](../explanation/psth_significance.md) for what the test does and how to read it.

**Significance Level (alpha)** is the two-sided threshold: `0.05` gives a 95% interval, `0.01` a 99% one. A stricter alpha widens the interval, so fewer stretches clear zero. The value is recorded in each result file alongside the significance flags, since the flags mean nothing without it.

**Bootstrap Resamples** is how many times the trials are resampled to build each interval. More resamples means less run-to-run variation and a proportionally longer run; the default of 1000 is the usual choice for confidence intervals. Note that resolving a two-sided alpha needs at least `2 / alpha` resamples — 40 at `0.05`, 200 at `0.01` — below which the interval comes back narrower than the alpha you asked for, and GuPPy logs a warning saying so.

**Event A / Event B** is the table of pairs to compare against each other, using the event labels assigned in Step 1 (Label Stores). It starts with one blank row; **+ Add comparison** appends another, and the ✕ beside a row removes it. There is no limit on the number of pairs. Testing against zero needs no configuration because there is only one sensible version of it; which two events are worth contrasting is a scientific judgement, so you name those. Each pair is compared within every recording site and metric present. Leave the table blank to run only the tests against zero.

Comparisons run **inside a single output folder**. Run the step on a session run folder and its trials are resampled; run Group Analysis on a group folder and that group's session averages are, which is the correct unit for a group-level claim. Comparing two different folders — one session against another, or one group against another — is not yet supported.

Requires a non-zero **Window for Moving Average filter**, since the minimum duration of a significant stretch is derived from it.

### Artifact Removal

Artifact removal is not configured from this form. It is handled by two optional steps that run after Step 3 — **Select Artifact Windows** and **Remove Artifacts** — and the removal method is chosen on the Select Artifact Windows page. See [Remove artifacts from a recording](../how-to/artifact-removal.md).

Both settings still appear in `GuPPyParamtersUsed.json` as a record of what was applied to each run:

| Internal name | Meaning | Written by |
|---------------|---------|------------|
| `removeArtifacts` | Whether artifacts were removed from this run. | `False` after Step 3; `True` after Remove Artifacts. |
| `artifactsRemovalMethod` | How the marked periods were applied. | Select Artifact Windows records the method chosen on the page. |

`replace with NaN` (the default) keeps the trace at its original length and masks the marked samples with NaN, which downstream code treats as missing. `concatenate` drops the marked sections and stitches the surviving ones together, so the resulting trace is shorter than the input; it re-times the kept samples onto a new timeline, is unsupported by NWB export, and cannot be combined with cross-correlation.

## Group Output Folder Selection

Collapsed by default on the homepage. Picks which defined groups the pipeline works with.

*Used by: the Group Analysis step, and Step 5 (Visualize the results).*

| Parameter | Description | Type | Default | Options / range |
|-----------|-------------|------|---------|-----------------|
| (file browser) | Group output directories to work with. | list of paths | empty | absolute paths to `<name>_group` directories |

**File browser** is the group counterpart of [Output Folder Selection](#output-folder-selection), rooted at your home directory. The same selection serves both averaging and visualization, so you choose it once: the Group Analysis step averages into the selected groups, and Step 5 opens them alongside any selected session runs. Groups are created in the Label Groups GUI, whose controls are covered in [Average results across sessions](../how-to/group-analysis.md). This is a UI selector, not a saved analysis parameter, so it has no internal name in the index below.

Groups are visualized by selecting them here; there is no separate mode to switch on.

---

## Internal name index

This index is for readers who arrive with an internal parameter name in hand and need to find the corresponding GUI parameter. That happens in four situations:

- **Reproducing or auditing a past analysis** by reading the `GuPPyParamtersUsed.json` snapshot that GuPPy writes after every run; the JSON is keyed by internal names. Selecting a finished output run in the Output Folder Selection card also reloads this snapshot back into the form, so you can resume a run without the defaults silently overwriting the parameters the earlier steps used.
- **Writing a headless or scripted analysis** against the API in `src/guppy/testing/api.py`, which takes a dict keyed by these names.
- **Debugging a validator or pipeline error**, since error messages cite the internal name (for example `baselineWindowEnd=120 exceeds signal duration 90.5s`).
- **Reading or contributing to the source code**, where parameter accesses go through the internal names.

The table is sorted alphabetically by internal name. Each row links to the section above where the parameter is documented in full. Internal-only keys that have no GUI counterpart (`abspath`) are listed too.

| Internal name | Parameter | Section |
|---------------|-----------|---------|
| `abspath` | (auto-derived; not user-set) | [Input Folder Selection](#input-folder-selection) |
| `artifactsRemovalMethod` | (recorded provenance; set on the Select Artifact Windows page) | [Artifact Removal](#artifact-removal) |
| `auc_units` | AUC Units | [Peak and AUC Measurement](#peak-and-auc-measurement) |
| `baselineCorrectionEnd` | Baseline Correction End time | [PSTH Computation](#psth-computation) |
| `baselineCorrectionStart` | Baseline Correction Start time | [PSTH Computation](#psth-computation) |
| `baselineWindowEnd` | Baseline Window End Time (s) | [Z-score Normalization](#z-score-normalization) |
| `baselineWindowStart` | Baseline Window Start Time (s) | [Z-score Normalization](#z-score-normalization) |
| `bin_psth_trials` | Time(min) / # of trials for binning | [PSTH Computation](#psth-computation) |
| `binnedMetricsWidth` | Bin Width (s) | [Metric Binning](#metric-binning) |
| `combine_data` | Combine Data? | [Input Folder Selection](#input-folder-selection) |
| `computeBinnedMetrics` | Compute Binned Metrics? | [Metric Binning](#metric-binning) |
| `computeCorr` | Compute Cross-correlation | [PSTH Computation](#psth-computation) |
| `computePsthSignificance` | Compute PSTH Significance? | [Significance Testing](#significance-testing) |
| `control_fit_method` | Control Channel Fitting Method | [Control Channel Fitting](#control-channel-fitting) |
| `controlFitWindowEnd` | Control Fit Window End Time (s) | [Control Channel Fitting](#control-channel-fitting) |
| `controlFitWindowMode` | Control Fit Window | [Control Channel Fitting](#control-channel-fitting) |
| `controlFitWindowStart` | Control Fit Window Start Time (s) | [Control Channel Fitting](#control-channel-fitting) |
| `dandi_uri_map` | (DANDI selector) | [Input Folder Selection](#input-folder-selection) |
| `filter_window` | Window for Moving Average filter | [Signal Filtering](#signal-filtering) |
| `session_folders` | (file browser, Input Folder Selection) | [Input Folder Selection](#input-folder-selection) |
| `highAmpFilt` | HAFT | [Transient Detection](#transient-detection) |
| `isosbestic_control` | Isosbestic Control Channel? | [Control Channel Fitting](#control-channel-fitting) |
| `mode` | Data Source | [Input Folder Selection](#input-folder-selection) |
| `moving_window` | Moving Window for transients detection (s) | [Transient Detection](#transient-detection) |
| `nSecPost` | Seconds after 0 | [PSTH Computation](#psth-computation) |
| `nSecPrev` | Seconds before 0 | [PSTH Computation](#psth-computation) |
| `numberOfCores` | # of cores | [Parallel Execution](#parallel-execution) |
| `peak_endPoint` | Peak End time | [Peak and AUC Measurement](#peak-and-auc-measurement) |
| `peak_startPoint` | Peak Start time | [Peak and AUC Measurement](#peak-and-auc-measurement) |
| `psthComparisonsA` | Event A (comparison table) | [Significance Testing](#significance-testing) |
| `psthComparisonsB` | Event B (comparison table) | [Significance Testing](#significance-testing) |
| `psthBootstrapResamples` | Bootstrap Resamples | [Significance Testing](#significance-testing) |
| `psthSignificanceAlpha` | Significance Level (alpha) | [Significance Testing](#significance-testing) |
| `photobleaching_detrend` | Photobleaching Detrend? | [Control Channel Fitting](#control-channel-fitting) |
| `removeArtifacts` | (recorded provenance; not user-set) | [Artifact Removal](#artifact-removal) |
| `selectForComputePsth` | z_score and/or ΔF/F? (psth) | [PSTH Computation](#psth-computation) |
| `selectForTransientsComputation` | z_score and/or ΔF/F? (transients) | [Transient Detection](#transient-detection) |
| `timeForLightsTurnOn` | Eliminate first few seconds | [Signal Filtering](#signal-filtering) |
| `timeInterval` | Time Interval (s) | [PSTH Computation](#psth-computation) |
| `transientsThresh` | TD Thresh | [Transient Detection](#transient-detection) |
| `use_time_or_trials` | Bin PSTH trials | [PSTH Computation](#psth-computation) |
| `useTransientsAsEvents` | Use Transients as Events? | [Transient Detection](#transient-detection) |
| `zscore_method` | z-score computation Method | [Z-score Normalization](#z-score-normalization) |

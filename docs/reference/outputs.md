# Output data model

Every file GuPPy writes to disk: where it lands, what its name means, and what is inside it. `store_id`, `store_label`, `recording_site`, `session_folder`, `run` and `run_folder` are used here as the [Glossary](glossary.md) defines them.

---

## Where outputs go

Every output of a run lives in a single directory, the run folder, created as a subdirectory of the session folder:

```
<session_folder>/<session_name>_output_<run_name>/
```

`<session_name>` is the session folder's own name, so a session at `/data/Photo_63_207` gets run folders at `/data/Photo_63_207/Photo_63_207_output_1`. `<run_name>` is either the name you type in the Label Stores GUI or, when you leave it to GuPPy, the lowest integer for which no such directory exists yet — `_output_1` on the first run, `_output_2` on the second. Re-running Step 1 over an existing run folder with the overwrite option deletes its entire contents first. Because each run folder is self-contained, one session can hold several runs analyzed under different parameters — see [Comparing Two Parameter Sets](../tutorials/compare_parameters.md).

Three further directories can appear:

| Directory | Location | Written by |
|-----------|----------|------------|
| `cross_correlation_output/` | inside a run folder | Step 4, when **Compute Cross-correlation** is enabled |
| `<name>_group/` | in the destination directory you pick | the Group Analysis step |
| `saved_plots/` | inside a run folder | Step 5 — created but left empty, see [Step 5](#step-5-visualize-the-results) |

A group directory is not inside any session folder. It is created in the destination directory chosen in the Group Analysis card.

---

## File formats

GuPPy's two HDF5 extensions are **not** interchangeable: `.hdf5` and `.h5` are written by different libraries and have to be read by different ones. [Custom Plots from GuPPy Outputs](../tutorials/custom_plots.md) works through loading each of them in Python.

### `.hdf5`

Raw HDF5 written with h5py, following GuPPy's convention of one file per store with several named datasets inside it. Array datasets are one-dimensional and chunked; scalars (a store's name, its sample count) are zero-dimensional datasets. **No HDF5 attributes are set anywhere** — `sampling_rate` and the other scalars are datasets, not attributes.

### `.h5`

A pandas DataFrame written with `DataFrame.to_hdf`, holding exactly one DataFrame under the key `df`, and read back through pandas rather than h5py. These are the PSTH, peak/AUC, transient-summary, binned-metrics, covariate, cross-correlation and tonic tables.

### `.csv`

Flat text. Inside a run folder: the store mappings (`storesList.csv`, `combine_storesList.csv`), tables that also exist as an `.h5` (peak/AUC, transient frequency and amplitude, binned metrics, binned covariates and covariate correlations), and the two tables written as CSV only — `transientsOccurrences_<metric>.csv` and `tonic_epochs_<site>.csv`. The channel exports written outside the run folder are CSV too.

### `.npy`

A raw NumPy array, used only for the artifact keep-windows in `coordsForPreProcessing_<site>.npy`.

### `.json`

Provenance and configuration: `GuPPyParamtersUsed.json` and `.npm_params.json`.

### `.yaml`

Plain text, written only by Step 6: the NWB metadata overlay `nwb_metadata.yaml`. It is the authoritative artifact behind the metadata form, and is meant to be read, edited and copied between sessions by hand.

### `.nwb`

HDF5 written through pynwb, holding the whole session — raw traces, events and every GuPPy product — as typed NWB objects rather than named arrays. Written only by Step 7. The file carries the `ndx-fiber-photometry`, `ndx-ophys-devices` and `ndx-guppy` schema extensions inside it, so it stays readable without them installed. See [Fiber photometry data in NWB](../explanation/nwb.md).

---

## Filename vocabulary

The placeholders used in the filename patterns:

| Placeholder | Meaning | Example |
|-------------|---------|---------|
| `<store_id>` | The store name as it appears in the acquisition system's own files | `Dv1A`, `405R`, `Sample_Signal_Channel` |
| `<store_label>` | The analytic label you assigned in Step 1 | `signal_DMS`, `control_DMS`, `port_entries` |
| `<site>` | A recording site — the suffix shared by a `signal_`/`control_` label pair | `DMS`, `region` |
| `<metric>` | A preprocessed trace name, always `z_score_<site>` or `dff_<site>` | `z_score_DMS` |
| `<metric-prefix>` | The metric without its recording site | `z_score`, `dff` |
| `<event>` | The `store_label` of an event store | `port_entries`, `RewardPort` |

Because `<metric>` already ends in the recording site, filenames that combine an event with a metric carry the site twice: `port_entries_DMS_z_score_DMS.h5` is the PSTH of the event `port_entries` at site `DMS`, computed on the `z_score_DMS` trace.

Backslashes and forward slashes in a store or event name are replaced with underscores before the name reaches a filename, so a TDT store named `PAB/` produces files beginning `PAB_`.

---

## Step 1: Label your channels

| File | Contents |
|------|----------|
| `storesList.csv` | The store id → store label mapping for the run |
| `.npm_params.json` | Neurophotometrics decomposition settings (NPM sessions only) |

**`storesList.csv`** is a headerless CSV of exactly two rows and one column per store. Row 0 holds the `store_id` values, row 1 the `store_label` values you assigned, column by column. Every later step reads it, and a run folder without it is not a valid run.

```
Dv1A,Dv2A,PrtN
control_DMS,signal_DMS,port_entries
```

**`.npm_params.json`** is written only for Neurophotometrics sessions and holds the settings Step 2 uses to demultiplex the interleaved channels: `npm_split_events`, `npm_time_unit` (always stored resolved, never null) and `npm_timestamp_column_name`.

---

## Step 2: Load the raw data

| File | Contents |
|------|----------|
| `<store_id>.hdf5` | One file per store, holding that store's raw samples and timestamps |

These files are named after the **store id**, not the store label — `Dv1A.hdf5`, not `signal_DMS.hdf5`. Step 3 writes a second, separate set of files named after the store labels; see [Two sets of per-store HDF5 files](#two-sets-of-per-store-hdf5-files).

Which datasets a file carries depends on the acquisition format and on whether the store is a continuous trace or an event stream. Event stores have timestamps but no `data`.

| Dataset | Shape | Contents | Written by |
|---------|-------|----------|------------|
| `timestamps` | 1-D float | Sample or event times on the acquisition clock | every format |
| `data` | 1-D float | The samples (continuous stores only) | every format |
| `sampling_rate` | 0-D float (TDT, NWB); 1-D float, length 1 (CSV, Doric, NPM) | Samples per second (continuous stores only) | every format |
| `store_id` | 0-D string | The store's own name | TDT, NWB |
| `npoints` | 0-D int | Samples per acquisition block | TDT, NWB |
| `channels` | 1-D int | Per-block channel number | TDT |

For TDT stores, `data` is longer than `timestamps`: the acquisition writes samples in blocks of `npoints` and timestamps one per block, so the two arrays only line up after Step 3 has expanded the timebase. For every other format they are the same length.

TDT epoc stores whose strobe values encode several distinct behaviors are split at discovery time into one store per strobe value, named by appending the formatted value to the store id — `PAB/` with strobe `16` becomes `PAB16.hdf5`. Each split store gets its own file, with `timestamps` filtered to that strobe value.

---

## Step 3: Preprocess the signal

| File | Contents |
|------|----------|
| `timeCorrection_<site>.hdf5` | The corrected timebase for one recording site |
| `<store_label>.hdf5` | The trimmed and filtered samples for one labeled store |
| `<event>_<site>.hdf5` | Event timestamps mapped onto the corrected timebase |
| `z_score_<site>.hdf5` | The z-scored trace |
| `dff_<site>.hdf5` | The ΔF/F trace |
| `cntrl_sig_fit_<site>.hdf5` | The control channel after fitting it to the signal |
| `cntrl<i>.hdf5` | Placeholder store, non-isosbestic recordings only |
| `combine_storesList.csv` | Merged store mapping, **Combine Data?** runs only |

**`timeCorrection_<site>.hdf5`** carries five datasets: `timestampNew` (the corrected timestamp array, the timebase every other trace at this site is on), `correctionIndex` (the index array applied to the raw timestamps to produce it), `sampling_rate` (length-1 array), `timeRecStart` (the raw first timestamp, before correction) and `recordingStart` (the instant the corrected timebase treats as the recording start — `0.0` for TDT, the raw first timestamp for every other format).

**`<store_label>.hdf5`** holds a single dataset `data`: the store's samples after the warm-up seconds have been trimmed and the moving-average filter applied. One file per labeled store, so `signal_DMS.hdf5` and `control_DMS.hdf5`.

**`<event>_<site>.hdf5`** holds a single dataset `ts`: the event's timestamps on the corrected timebase. There is one file per (event, recording site) pair, each carrying that site's own corrected timestamps — an event store labeled `port_entries` in a two-site recording produces both `port_entries_DMS.hdf5` and `port_entries_DLS.hdf5`. Step 4 rewrites `ts` in place with the subset of timestamps the PSTH actually used.

**`z_score_<site>.hdf5`, `dff_<site>.hdf5` and `cntrl_sig_fit_<site>.hdf5`** each hold a single dataset `data`. `cntrl_sig_fit_<site>.hdf5` is the trace the preprocessing and artifact-removal views overlay on the signal.

**`cntrl<i>.hdf5` and the synthesized control** appear when **Isosbestic Control Channel?** is off. GuPPy first copies the signal store's file to `cntrl<i>.hdf5` as a placeholder and appends a `cntrl<i>` → `control_<site>` column to `storesList.csv`, then writes the synthetic exponential-decay control it fits to the signal into `control_<site>.hdf5` under the key `data`. In this mode `control_<site>.hdf5` is a GuPPy-generated trace, not a recorded one.

**`combine_storesList.csv`** appears only when **Combine Data?** is on. It has the same two-row shape as `storesList.csv` and holds the union of the store mappings across all sessions in the combined group. The combined traces themselves are written into the first run folder of the group, overwriting its `timeCorrection_<site>.hdf5`, `<store_label>.hdf5` and `<event>_<site>.hdf5` files.

### Two sets of per-store HDF5 files

A run folder ends up with two overlapping sets of HDF5 files. Step 2 writes one per **store id** (`Dv1A.hdf5`) holding the raw samples straight from the acquisition files. Step 3 writes one per **store label** (`signal_DMS.hdf5`) holding the same channel after trimming and filtering. Nothing renames or replaces the Step 2 files; both sets sit side by side, and `storesList.csv` is what connects them.

---

## Artifact windows and removal

*Written by: Select Artifact Windows and Remove Artifacts (both optional, between Steps 3 and 4).*

| File | Contents |
|------|----------|
| `coordsForPreProcessing_<site>.npy` | The periods of the recording to **keep** |

**`coordsForPreProcessing_<site>.npy`** is a float array of shape `(2M, 2)`. Column 0 holds the keep-window bounds interleaved as `[start_0, end_0, start_1, end_1, …]`; column 1 is an unused placeholder of zeros. The file records what survives, not the artifact periods you marked in the GUI. A recording site with no marked artifacts gets no file at all, and the pipeline then treats the whole recording as one keep window.

Remove Artifacts does not create new files. It rewrites the Step 3 outputs in place: `data` in each `<store_label>.hdf5`, and `ts` in each `<event>_<site>.hdf5`. The `concatenate` method additionally rewrites `timestampNew` and sets `recordingStart` to `0.0` in `timeCorrection_<site>.hdf5`. See [Remove artifacts from a recording](../how-to/artifact-removal.md).

---

## Tonic analysis

*Written by: Tonic Analysis (optional, between Remove Artifacts and Step 4).*

| File | Contents |
|------|----------|
| `tonic_epochs_<site>.csv` | The epoch windows defined for one recording site |
| `tonic_<site>.h5` | Each epoch's mean z-score and mean ΔF/F |

Both files are written per recording site, and only for sites that have at least one epoch window. Saving with a site's windows cleared deletes that site's pair.

**`tonic_epochs_<site>.csv`** has a header row and one row per epoch, with columns `label`, `start` and `end`. `start` and `end` are in seconds of absolute session time, on the same timebase as `timestampNew`. There is no index column.

**`tonic_<site>.h5`** holds a DataFrame under the key `df`, indexed by the epoch labels from the CSV (index name `epoch`), with columns `mean_zscore` and `mean_dff` — the mean of `z_score_<site>.hdf5` and of `dff_<site>.hdf5` over that epoch's window. Windows are clamped to the recording's timespan and NaN samples are excluded from the mean.

The means are computed at save time from the traces then on disk, and the differences between epochs shown in the visualization are derived at view time from a selectable baseline epoch — only the absolute means are stored. See [Measure tonic signal levels across a session](../how-to/tonic-analysis.md).

---

## Step 4: Compute the PSTH

| File | Contents |
|------|----------|
| `<event>_<site>_<metric>.h5` | The PSTH: one column per trial, plus summary columns |
| `<event>_<site>_baselineUncorrected_<metric>.h5` | The same PSTH before baseline correction |
| `peak_AUC_<event>_<site>_<metric>.h5` and `.csv` | Peak amplitude and area under the curve |
| `freqAndAmp_<metric>.h5` and `.csv` | Transient frequency and mean amplitude, one row |
| `transientsOccurrences_<metric>.csv` | One row per detected transient |
| `transient_outputs_<metric>.hdf5` | The trace and peak indices behind the transient results |
| `transients_<metric>.hdf5` | The transient times as an event train, **Use Transients as Events?** runs only |
| `binned_metrics_<site>.h5` and `.csv` | One row per fixed-width time bin, when **Compute Binned Metrics?** is enabled |
| `binned_covariates_<site>.h5` and `.csv` | One row per time bin, holding the behavioral covariate means |
| `covariate_correlations_<site>.h5` and `.csv` | One row per metric–covariate pair |
| `cross_correlation_output/corr_<event>_<metric-prefix>_<siteA>_<siteB>.h5` | Cross-correlation between two recording sites |

**The PSTH files** hold a `float32` DataFrame under the key `df`, with one row per time point in the peri-event window. The columns, in order: one column per trial, labeled with that trial's event timestamp as a string; then, when trial binning is enabled, a `bin_(<label>)` and `bin_err_(<label>)` pair per bin; then `timestamps` (the peri-event time axis, running from `-nSecPrev` to `+nSecPost`); then `mean` and `err`, the mean and standard error across the single-trial columns only. The `_baselineUncorrected_` file is the same table before the baseline correction was subtracted.

**The peak/AUC files** hold a DataFrame with one row per trial, per bin, and one for `mean` — matching the PSTH columns — and one column per configured peak window: `peak_pos_<N>`, `peak_neg_<N>` and `area_<N>` for the Nth window. The row index is the session folder name joined to the column label, so a trial row reads `<session_name>_<event timestamp>`. A window that falls outside the PSTH yields `peak_<N>` and `area_<N>` set to `NaN`; when no peak windows are configured at all, the columns are a single `peak` and `area` pair, both `NaN`. Whether `area_<N>` is in metric-seconds or in metric-samples depends on the **AUC Units** parameter. The `.h5` and `.csv` hold the same table.

**`freqAndAmp_<metric>.h5` and `.csv`** hold a single row indexed by the session folder name, with columns `freq (events/min)` and `amplitude` — the transient rate and the mean transient amplitude for that trace.

**`transientsOccurrences_<metric>.csv`** has one row per detected transient, with columns `timestamps` and `amplitude` and a plain integer index. There is no `.h5` companion.

**`transient_outputs_<metric>.hdf5`** holds the inputs the transient plot is drawn from: `z_score` (the NaN-free trace the detector ran on), `timestamps`, and `peaksInd` (the integer indices of the detected peaks within that trace).

**`transients_<metric>.hdf5`** appears only when **Use Transients as Events?** is on. It holds a single dataset `ts` — the detected transient times — in exactly the shape of an `<event>_<site>.hdf5` event file, which is how the transients stand in for a TTL train. The event label is `transients_z_score` (or `transients_dff`), so the recording site is the metric's own site: `transients_z_score_DMS.hdf5` is the DMS transient train, and the PSTH computed from it lands in `transients_z_score_DMS_z_score_DMS.h5` with a matching `peak_AUC_` pair. As with any event file, Step 4 rewrites `ts` in place with the subset of transients the PSTH actually used.

**`binned_metrics_<site>.h5` and `.csv`** are written only when **Compute Binned Metrics?** is enabled. They hold the same table: one row per fixed-width time bin, indexed `0..n-1` (index name `bin`). The columns are `bin_start` and `bin_end` (seconds of absolute session time, on the same timebase as `timestampNew`), `n_samples`, `mean_zscore`, `mean_dff`, and a `transient_count_z_score` and/or `transient_count_dff` column.

Bins start at the first corrected timestamp and run in **Bin Width** steps to the last. The final bin is kept even when the session does not divide evenly, so it may be shorter than the rest — compare `bin_end - bin_start` against the others to spot it. Bins are half-open, `[bin_start, bin_end)`, except the last, which includes its own end; the transient counts follow the same convention. `n_samples` counts the samples behind the means, so a bin lost entirely to artifact removal reads `0` with `NaN` means.

Which count columns appear depends on **z_score and/or ΔF/F? (transients)**: the detector only runs on the metrics you select, and only those get a count column. The two mean columns are always present.

Two cases where a bin does not correspond to a fixed stretch of wall-clock time: with **Combine Data?** enabled the sessions are re-timed onto one synthetic timeline, so a bin can straddle the boundary between two of them; and with the `concatenate` artifact-removal method the time axis is compressed where artifacts were cut, so a 60-second bin spans more than 60 seconds of the original recording.

**`binned_covariates_<site>.h5` and `.csv`** are written only when the session carries a store labeled **behavioral covariate** and **Compute Binned Metrics?** is enabled. They hold one row per time bin, on exactly the bins of the matching `binned_metrics_<site>` table, indexed `0..n-1` (index name `bin`). The columns are `bin_start` and `bin_end`, then one column per covariate holding its mean over that bin, then one `n_samples_<covariate>` column per covariate. A bin holding no score reads `NaN` with a count of `0`. Scores timestamped outside the binned span are dropped rather than folded into the first or last bin.

**`covariate_correlations_<site>.h5` and `.csv`** hold one row per (metric, covariate) pair, indexed `0..n-1` (index name `pair`), with columns `metric`, `covariate`, `pearson_r`, `spearman_rho` and `n_bins`. Every column of `binned_metrics_<site>` that is not bin geometry is correlated, so both means and any transient counts appear. `n_bins` is the number of bins where the metric and the covariate are both present — the sample size actually behind the two coefficients, which is usually much smaller than the number of samples in the recording. Coefficients read `NaN` when either series is constant across the paired bins, or when fewer than three bins pair up.

**There is no p-value column, by design.** Pearson r and Spearman rho are reported as descriptive statistics only. Both the per-bin photometry values and a behavioral score vary slowly across a session, so successive bins are correlated with each other — and the standard significance tests for r and rho assume the opposite, that every sample is independent. A p-value computed from these numbers would therefore be far too small, and this holds whether GuPPy computes it or you compute it yourself from the reported columns. Treat the coefficients as a description of this one session, and see the [how-to guide](../how-to/correlate-behavioral-covariates.md) for what you can do with them.

**The cross-correlation files** hold a `float32` DataFrame under the key `df`, one row per lag. Columns are the trial labels the two recording sites have in common, then `timestamps`, then `mean` and `err`. Despite its name, the `timestamps` column holds **lag values in seconds**, not times. Cross-correlation cannot run on a recording processed with the `concatenate` artifact-removal method; Step 4 raises instead.

---

## Step 5: Visualize the results

Step 5 writes no data. It creates a `saved_plots/` subdirectory inside the run folder, but nothing is ever written into it: the **Save As** buttons deliver the figure to the browser as a PNG or SVG download, so exported figures land wherever your browser saves downloads.

---

## Step 6: Input Metadata

*Written by: Step 6 (Input Metadata), optional.*

| File | Contents |
|------|----------|
| `nwb_metadata.yaml` | The session's NWB metadata overlay |

**`nwb_metadata.yaml`** holds everything the NWB export needs that the data itself cannot supply. Its top-level keys are `NWBFile` (session description, identifier, start time, lab, institution, experimenter), `Subject`, `DeviceModels` and `Devices` (the optical hardware, models and instances), and `FiberPhotometry` — which nests `FiberPhotometryViruses`, `FiberPhotometryVirusInjections`, `FiberPhotometryIndicators`, the `FiberPhotometryTable` and its `rows` (one per channel, keyed `<recording_site>_<role>`), and the per-role `signal` and `control` entries naming which rows each response series covers.

The file is a session-level overlay, not a complete metadata document: at export it is applied on top of what GuPPy and the acquisition files already supply, so it only ever adds or replaces. It is written only for sessions read from raw acquisition files — a session GuPPy processed out of an NWB file never gets one, because its source already carries this information. See [Export a session to NWB](../how-to/export-to-nwb.md).

---

## Step 7: Export to NWB

*Written by: Step 7 (Export to NWB), optional.*

| File | Contents |
|------|----------|
| `<session_name>_output_<run_name>.nwb` | The whole session as one NWB file |

**The NWB file** is named after the run folder rather than the session, so exports from several runs or sessions can be pooled into one directory without renaming. It holds three top-level groups: `acquisition` (the raw photometry response series), `events` (one table per raw event store, plus the `GuppyEvents` table of the onsets GuPPy analyzed), and `processing/guppy` (the derived traces, transients, PSTHs, peak/AUC summaries, cross-correlations, valid-signal intervals and the parameters used). Re-running Step 7 overwrites the file; each export rebuilds from the run folder rather than adding to what is there.

For a session GuPPy read from an NWB file — a local `.nwb` or a DANDI asset — the output is a **copy of that source** with the GuPPy outputs added, written on the extension versions the source used. The source file is never modified.

See [Fiber photometry data in NWB](../explanation/nwb.md) for what each object holds.

---

## Group analysis: group directories

*Written by: the Group Analysis step.*

A group directory is named `<group_name>_group` and sits in the destination directory chosen in the Group Analysis card. It holds the same filename patterns as a run folder, with the per-member values combined:

| File | Contents |
|------|----------|
| `group_members.json` | The run folders this group averaged, in averaging order |
| `GuPPyParamtersUsed.json` | The parameters the averaging ran under |
| `storesList.csv` | The store mapping, listing only the events the group holds a PSTH for |
| `z_score_<site>.hdf5`, `dff_<site>.hdf5` | Empty placeholders |
| `<event>_<site>_<metric>.h5` | Group PSTH: one column per member run |
| `peak_AUC_<event>_<site>_<metric>.h5` and `.csv` | Every member's peak/AUC rows concatenated |
| `freqAndAmp_<metric>.h5` and `.csv` | One row per member run |
| `cross_correlation_output/corr_<event>_<metric-prefix>_<siteA>_<siteB>.h5` | One column per member run |

The group PSTH has the same shape as a per-session PSTH, but its trial columns are replaced by one column per member run, labeled with the run folder's name, followed by the same `timestamps`, `mean` and `err` columns. Column order matches `group_members.json`, so column *n* is member *n*. The `z_score_<site>.hdf5` and `dff_<site>.hdf5` files hold an empty `data` dataset; only their filenames carry information, naming the group's recording sites.

`group_members.json` has a single key, `member_run_folders`, holding the absolute paths of the runs that were averaged. GuPPy uses it to reload a group into the form, and to confirm a directory is one of its own before rebuilding it.

Because the step averages what its members already hold, `storesList.csv` lists only the events a PSTH was actually written for. An event that no member recorded is dropped rather than listed.

---

## Provenance: `GuPPyParamtersUsed.json`

*Written by: Steps 2, 3 and 4, and by Select Artifact Windows.*

A JSON snapshot of the analysis parameters, written into every run folder the step operated on. Each of Steps 2, 3 and 4 rewrites it, so the file always reflects the most recent step to touch that run. When a session has no run folder yet, the snapshot is written at the session folder root instead. The Group Analysis step writes one into the group directory too, recording the parameters the averaging ran under.

The first key is `guppy_version`, the installed version of the `guppy-neuro` package that produced the run. The rest are the analysis parameters themselves: `combine_data`, `isosbestic_control`, `control_fit_method`, `controlFitWindowMode`, `controlFitWindowStart`, `controlFitWindowEnd`, `photobleaching_detrend`, `timeForLightsTurnOn`, `filter_window`, `removeArtifacts`, `artifactsRemovalMethod`, `noChannels`, `zscore_method`, `baselineWindowStart`, `baselineWindowEnd`, `nSecPrev`, `nSecPost`, `computeCorr`, `useTransientsAsEvents`, `timeInterval`, `bin_psth_trials`, `use_time_or_trials`, `baselineCorrectionStart`, `baselineCorrectionEnd`, `peak_startPoint`, `peak_endPoint`, `auc_units`, `selectForComputePsth`, `selectForTransientsComputation`, `moving_window`, `highAmpFilt`, `transientsThresh`, `computeBinnedMetrics`, `binnedMetricsWidth` and `visualize_zscore_or_dff`. See the [Input parameter reference](parameters.md) for what each one controls.

`removeArtifacts` and `artifactsRemovalMethod` describe what was applied to that run rather than what the form currently holds: each step carries them forward from whatever the run folder already records. Saving artifact windows patches these two keys in place and leaves everything else untouched. In a group directory both take their defaults, since the Group Analysis step removes no artifacts of its own.

---

## Files written outside the run folder

| File | Location | Contents |
|------|----------|----------|
| `<name>.csv` | session folder | An imported custom event: one column, header `timestamps` |
| `cntrl<i>.csv` | session folder | The synthetic control trace, non-isosbestic runs only |
| `.storesList.json` | your home directory | Cache of previously used store labels |
| `guppy.log` | platform log directory | Application log |

**Custom events** imported through the Import Custom Events step are written into the session folder as a single-column CSV, where Step 1 discovers them alongside the acquisition system's own stores. See [Import custom events](../how-to/import-custom-events.md).

**`cntrl<i>.csv`** is written by the synthetic-control path in Step 3, alongside the `cntrl<i>.hdf5` placeholder, with columns `timestamps`, `data` and `sampling_rate`. It lands in the session folder rather than the run folder, which means a later Step 1 on the same session will discover it as an available store.

**`.storesList.json`** in your home directory maps each `store_id` you have ever labeled to the labels you gave it, and is used to pre-populate the Label Stores dropdowns. It is shared across all sessions and projects, and is not part of any run's output.

**`guppy.log`** is written to the platform log directory (`~/Library/Logs/guppy/guppy.log` on macOS, the equivalent under `%LOCALAPPDATA%` on Windows and `~/.local/state` on Linux). `guppy --export-logs` copies it to your desktop as `guppy_log_<timestamp>.log`, ready to attach to a bug report.

---

## Complete run folder map

```
<session_folder>/
  <name>.csv                                       imported custom event
  cntrl<i>.csv                                     step 3, non-isosbestic only
  <session_name>_output_<run_name>/
    storesList.csv                                 step 1
    .npm_params.json                               step 1, NPM only
    GuPPyParamtersUsed.json                        steps 2, 3, 4
    <store_id>.hdf5                                step 2   store_id, timestamps, data,
                                                            sampling_rate, npoints, channels
    timeCorrection_<site>.hdf5                     step 3   timestampNew, correctionIndex,
                                                            sampling_rate, timeRecStart,
                                                            recordingStart
    signal_<site>.hdf5                             step 3   data
    control_<site>.hdf5                            step 3   data
    cntrl<i>.hdf5                                  step 3   non-isosbestic placeholder
    <event>_<site>.hdf5                            step 3   ts   (rewritten by step 4)
    z_score_<site>.hdf5                            step 3   data
    dff_<site>.hdf5                                step 3   data
    cntrl_sig_fit_<site>.hdf5                      step 3   data
    combine_storesList.csv                         step 3, Combine Data? only
    coordsForPreProcessing_<site>.npy              Select Artifact Windows
    tonic_epochs_<site>.csv                        Tonic Analysis   label, start, end
    tonic_<site>.h5                                Tonic Analysis   DataFrame, key "df"
    <event>_<site>_<metric>.h5                     step 4   DataFrame, key "df"
    <event>_<site>_baselineUncorrected_<metric>.h5 step 4   DataFrame, key "df"
    peak_AUC_<event>_<site>_<metric>.h5            step 4   DataFrame, key "df"
    peak_AUC_<event>_<site>_<metric>.csv           step 4
    freqAndAmp_<metric>.h5                         step 4   DataFrame, key "df"
    freqAndAmp_<metric>.csv                        step 4
    transientsOccurrences_<metric>.csv             step 4
    transient_outputs_<metric>.hdf5                step 4   z_score, timestamps, peaksInd
    binned_metrics_<site>.h5                       step 4   DataFrame, key "df"
    binned_metrics_<site>.csv                      step 4
    binned_covariates_<site>.h5                    step 4   DataFrame, key "df"
    binned_covariates_<site>.csv                   step 4
    covariate_correlations_<site>.h5               step 4   DataFrame, key "df"
    covariate_correlations_<site>.csv              step 4
    cross_correlation_output/
      corr_<event>_<metric-prefix>_<siteA>_<siteB>.h5   step 4   DataFrame, key "df"
    saved_plots/                                   step 5, created empty
    nwb_metadata.yaml                              step 6, optional
    <session_name>_output_<run_name>.nwb           step 7, optional

<group destination directory>/
  <group_name>_group/                              Group Analysis step
    group_members.json
    GuPPyParamtersUsed.json
    storesList.csv
    z_score_<site>.hdf5                            empty placeholder
    dff_<site>.hdf5                                empty placeholder
    <event>_<site>_<metric>.h5
    peak_AUC_<event>_<site>_<metric>.h5 and .csv
    freqAndAmp_<metric>.h5 and .csv
    cross_correlation_output/
      corr_<event>_<metric-prefix>_<siteA>_<siteB>.h5
```

# Comparing Two Parameter Sets

Some parameters have no right answer until you try them. Is `standard z-score` or
`baseline z-score` the better normalization for this recording? Is the moving-average
window wide enough to suppress the noise without flattening the transients? The way to
decide is to analyze the session both ways and look at both results.

GuPPy keeps parameter sets apart with **runs**. A run is one analysis pass over a
session — its stores, its parameter set, and its own output folder — and one session can
hold as many runs as you want. Give each parameter set its own run and the second
analysis never overwrites the first.

In this tutorial you analyze the sample session twice, once under each of two z-score
methods, and then open both results together. The same workflow applies to any
parameter.

By the end you will have:

- Created two named runs on one session, `standard_zscore` and `baseline_zscore`
- Run Steps 2 through 4 into each run, one run at a time
- Opened both runs in the Visualization GUI to compare them
- Confirmed on disk that each run kept its own data and its own parameter snapshot

## Prerequisites

- **[Your First Analysis](first_analysis.md), completed.** This tutorial repeats that
  pipeline twice, so it describes each step only briefly and leaves the detail to the
  first tutorial. You need the same setup: GuPPy installed from source, and the sample
  session at `stubbed_testing_data/csv/sample_data_csv_1/` pulled down with Git LFS.

Launch GuPPy, and in the **Individual Analysis** card select
`stubbed_testing_data/csv/sample_data_csv_1/` as before:

```bash
guppy
```

## Build the first run

Click **Label Stores** in the sidebar and label the three channels exactly as you did in
the first tutorial:

| Channel | Type | Name |
|---------|------|------|
| `Sample_Control_Channel` | `control` | `A` |
| `Sample_Signal_Channel` | `signal` | `A` |
| `Sample_TTL` | `event TTLs` | `RewardPort` |

Click **Select Label Stores**, fill in the rows, and click **Show Selected
Configuration**.

Now the part that is new. Set **over-write storeslist file or create a new one?** to
`create_new_file`, and instead of leaving **Run name** blank, type `standard_zscore` into
it. Click **Save**.

A run's name is the suffix on its output folder, `<session>_output_<run name>`, so this
one creates `sample_data_csv_1_output_standard_zscore/`. Left blank, the name defaults to
the next free integer — `1`, then `2` — which is what the first tutorial got, and which
tells you nothing later about which folder held which parameters. Naming a run for the
parameter that varies does: the name becomes the folder name, the heading on the
visualization dashboard, and the only reminder you get months later of what you were
testing.

Run names may not be empty, contain path separators or `..`, or contain the substring
`_output_`.

## Analyze it with the standard z-score

Back on the homepage, open **Output Folder Selection** and, under **Existing runs (steps
2–5)**, select `sample_data_csv_1_output_standard_zscore` — and only that one. Every step
you run acts on the runs selected here.

Leave the parameters at their defaults; **z-score computation Method** is already
`standard z-score`, which is the parameter set this run is named for.

Run **Step 2: Read Raw Data**, then **Step 3: Preprocess**, then **Step 4: PSTH
Computation**, waiting for each progress bar to fill before clicking the next. When Step 4
finishes, this run holds a complete result.

## Build the second run

Click **Label Stores** again and label the channels a second time, with exactly the same
names as before — `control_A`, `signal_A`, `RewardPort`. The stores must match for the
comparison to mean anything; only the analysis parameters should differ.

Set **over-write storeslist file or create a new one?** to `create_new_file` again, type
`baseline_zscore` into **Run name**, and click **Save**.

```{image} ../_static/images/compare_parameters_run_name.png
:alt: The Label Stores GUI's "Choose how to save this store_array" section, with the save menu button below it and the Run name field filled in with baseline_zscore
:width: 70%
```

Leave `over_write_file` alone here. It deletes the contents of an existing run folder,
which would throw away the result you just computed.

## Analyze it with the baseline z-score

Back on the homepage, open **Output Folder Selection** again. Both runs are now listed
under **Existing runs (steps 2–5)**. Select `sample_data_csv_1_output_baseline_zscore`,
and make sure the first run is *not* selected.

```{image} ../_static/images/compare_parameters_existing_runs.png
:alt: The Output Folder Selection card's file browser listing the session's three run folders, sample_data_csv_1_output_1 and sample_data_csv_1_output_standard_zscore on the left, with sample_data_csv_1_output_baseline_zscore moved into the Selected files list
:width: 100%
```

Selecting a run loads the parameters saved in it back into the form, which is why you
start from the new run rather than editing the form first.

In the **Individual Analysis** card, change the one parameter you are testing:

- **z-score computation Method** → `baseline z-score`
- **Baseline Window Start Time (s)** → `2`
- **Baseline Window End Time (s)** → `60`

The window has to sit inside the recording, and preprocessing drops the first second (the
**Time for Lights to Turn On** parameter), leaving a signal that spans 1–411 s. Its first
event lands at 139 s, so 2–60 s is a stretch of recording that precedes every trial. Ask
for a window outside that span and GuPPy stops with
`baselineWindowStart=0 is before the signal start 1.001s; signal timespan is [1.001, 411]s`.
The [z-score explainer](../explanation/zscore.md) covers what the methods do with the
window.

Now run **Step 2: Read Raw Data**, **Step 3: Preprocess**, and **Step 4: PSTH
Computation** again, this time into the new run.

Step 2 is not optional even though you already ran it for the first run. Each run
folder holds its own copy of the raw HDF5 data, and a run folder fresh from Label Stores is
empty until Read Raw Data fills it. That is also why each run costs another copy of the raw
data on disk — worth knowing before you set up eight of them on a large session.

Run one run at a time through Steps 2–4. The parameters in the form apply to every run you
have selected, so selecting both here would compute both runs with identical parameters and
defeat the comparison.

## Compare the two runs

Select **both** runs under **Existing runs (steps 2–5)**, then click **Open Visualization
GUI**. GuPPy opens one dashboard per selected run, each in its own browser tab, so you can
flip between them with the same event and view selected.

Selecting two runs whose saved parameters differ shows the notification "Selected output
runs have different saved parameters; the form was left unchanged." That is expected here,
and harmless: the parameters are already baked into each run's outputs, and Step 5 only
reads them.

Every dashboard's browser tab is titled `Visualization GUI`, so the tabs look identical.
The run folder name is the heading at the top of each page — that is how you tell which is
which. Put the two tabs on the same z-score trace and you are looking at the answer to the
question you started with.

To keep a figure, use **Save As...** above each plot; the file arrives through your
browser's downloads.

## What landed on disk

The session folder now holds three run folders, the two from this tutorial beside the one
from the first:

```text
stubbed_testing_data/csv/sample_data_csv_1/
├── sample_data_csv_1_output_1/
├── sample_data_csv_1_output_standard_zscore/
└── sample_data_csv_1_output_baseline_zscore/
```

Each is a complete, independent result — its own raw HDF5 copies, preprocessed traces,
PSTH outputs, and a `GuPPyParamtersUsed.json` recording the parameters the run was analyzed
with. Nothing is shared between run folders, so deleting one leaves the others intact. See
[Output data model](../reference/outputs.md) for the full file layout.

## Gotchas

- **`GuPPyParamtersUsed.json` records the form, not the step.** It holds the form's values
  at the time of the last step that touched the run folder, not the values each individual
  step used. Change a parameter only in step with re-running the steps it affects —
  otherwise a run folder's snapshot can disagree with the data sitting next to it.
- **Not every change needs the full pipeline.** Once a run folder has been through Steps 2
  and 3, changing a Step 3 parameter (z-score method, baseline window, moving-average
  window, control fitting, isosbestic correction, artifact removal) means re-running Steps
  3 and 4 in it, while changing a Step 4 parameter (PSTH window, peak and AUC windows,
  baseline correction, transient thresholds) means re-running only Step 4.
- **A new run always starts at Step 2.** Even if you are only varying a Step 4 parameter,
  a run fresh from Label Stores needs Steps 2 and 3 before Step 4 has anything to read.
- **Re-running into the same folder overwrites it.** Pointing the pipeline at a run folder
  you have already computed replaces that result instead of sitting beside it, and you end
  up with nothing to compare.

## Next steps

- See [Custom Plots from GuPPy Outputs](custom_plots.md) to load both run folders in
  Python and put their traces on one set of axes.
- See the [parameter reference](../reference/parameters.md) for what every parameter you
  might vary this way actually controls.
- See [Explanation](../explanation/index.md) for background on the z-score methods and the
  isosbestic correction.

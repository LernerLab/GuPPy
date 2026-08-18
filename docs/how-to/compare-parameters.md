# Compare parameter choices on the same session

Some parameters have no right answer until you try them. Is `standard z-score`
or `baseline z-score` the better normalization for this recording? Is the
moving-average window wide enough to suppress the noise without flattening the
transients? The way to decide is to analyze the session both ways and look at
both results.

GuPPy keeps parameter sets apart with **runs**. A run is one analysis pass over
a session — its stores, its parameter set, and its own output folder — and one
session can hold as many runs as you want. Give each parameter set its own run
and the second analysis never overwrites the first.

This guide compares two z-score methods on one session. The same workflow
applies to any parameter.

## Before you start

Analyze the session once, through **Step 4: PSTH Computation**, as in
[Your First Analysis](../tutorials/first_analysis.md). This guide covers the
second pass.

The mistake to avoid is re-running the pipeline against the same output folder
with a new parameter value. Every step writes into the run folder you select,
so that replaces your first result instead of sitting beside it, and you end up
with nothing to compare.

## Naming your runs

A run's name is the suffix on its output folder, `<session>_output_<run name>`.
You set it in the **Run name** field of the Label Stores GUI. Left blank, it
defaults to the next free integer — `1`, then `2` — which tells you nothing
later about which folder held which parameters.

Name each run for the parameter that varies instead: `standard_zscore` and
`baseline_zscore`, or `filter_100` and `filter_500`. The name becomes the
folder name, the label on the visualization dashboard, and the only reminder
you get months later of what you were testing.

Run names may not be empty, contain path separators or `..`, or contain the
substring `_output_`.

## Setting up the second run

1. Click **Label Stores** and label the session's recording sites exactly as
   you did the first time — same `signal_<site>` / `control_<site>` names. The
   stores must match for the comparison to mean anything; only the analysis
   parameters should differ.

2. Set **over-write storeslist file or create a new one?** to `create_new_file`
   and type your run name into **Run name**, then click **Save**. GuPPy creates
   a second output folder alongside the first.

   ```{image} ../_static/images/compare_parameters_run_name.png
   :alt: The Label Stores GUI's "Choose how to save this store_array" section, with the selector set to create_new_file and the Run name field filled in with baseline_zscore
   :width: 70%
   ```

   Leave `over_write_file` alone here — it deletes the contents of an existing
   run folder, which is the opposite of what you want.

3. Back on the homepage, open **Output Folder Selection** and, under **Existing
   runs (steps 2–5)**, select the new run on its own. Selecting a run loads the
   parameters saved in it back into the form, so start from the new run to
   avoid inheriting anything unintended.

   ```{image} ../_static/images/compare_parameters_existing_runs.png
   :alt: The Output Folder Selection card's file browser listing two sibling run folders, sample_data_csv_1_output_baseline_zscore and sample_data_csv_1_output_standard_zscore, with the baseline one moved into the Selected files list
   :width: 100%
   ```

4. In the **Individual Analysis** card, change the one parameter you are testing
   — here **z-score computation Method** to `baseline z-score`, plus the
   **Baseline Window Start/End Time (s)** it needs. See the
   [parameters screenshot](../tutorials/first_analysis.md#set-parameters) in the
   getting-started tutorial for where these sit, and the
   [z-score explainer](../explanation/zscore.md) for what the methods do.

Select **one** run at a time for Steps 2–4. The parameters in the form apply to
every run you have selected, so selecting both would compute both runs with
identical parameters and defeat the comparison.

## Running the pipeline for each run

With the new run selected, run **Step 2: Read Raw Data**, **Step 3:
Preprocess**, and **Step 4: PSTH Computation**.

Step 2 is not optional here even though you already ran it. Each run folder
holds its own copy of the raw HDF5 data, and a run folder fresh from Step 1 is
empty until Step 2 fills it. This means each run costs another copy of the raw
data on disk — worth knowing before you set up eight of them on a large
session.

If you are only varying a Step 4 parameter — the PSTH window, peak and AUC
windows, baseline correction, transient thresholds — the new run still needs
Steps 2 and 3 before Step 4 can read from it.

## Comparing the results

Select **both** runs under **Existing runs (steps 2–5)**, then click **Open
Visualization GUI**. GuPPy opens one dashboard per selected run, each in its own
browser tab, so you can flip between them with the same event and view
selected.

Every dashboard's browser tab is titled `Visualization GUI`, so the tabs look
identical. The run folder name is the heading at the top of each page — that is
how you tell which is which.

Selecting two runs whose saved parameters differ shows the notification
"Selected output runs have different saved parameters; the form was left
unchanged." That is expected here, and harmless: the parameters are already
baked into each run's outputs, and Step 5 only reads them.

To keep a figure, use **Save As...** above each plot; the file arrives through
your browser's downloads.

## What lands on disk

Each run folder is a complete, independent result — its own raw HDF5 copies,
preprocessed traces, PSTH outputs, and provenance:

| File | Contents |
|------|----------|
| `GuPPyParamtersUsed.json` | The parameters this run was analyzed with |

Nothing is shared between run folders, so deleting one leaves the other intact.
See [Output data model](../reference/outputs.md) for the full file layout.

## Notes

- Selecting a run under **Existing runs (steps 2–5)** loads that run's saved
  parameters back into the form, so switching between runs restores the
  parameters that produced each one.
- `GuPPyParamtersUsed.json` records the form's values at the time of the last
  step that touched the run folder, not the values each individual step used.
  Change a parameter only in step with re-running the steps it affects —
  otherwise a run folder's snapshot can disagree with the data sitting next to
  it.
- Once a run folder has been through Steps 2 and 3, changing a Step 3 parameter
  (z-score method, baseline window, moving-average window, control fitting,
  isosbestic correction, artifact removal) means re-running Steps 3 and 4 in it,
  while changing a Step 4 parameter (PSTH window, peak and AUC windows, baseline
  correction, transient thresholds) means re-running only Step 4.

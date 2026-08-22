# Average results across sessions

Group analysis pools an event's response — PSTH, peak/AUC, transient frequency and amplitude,
and cross-correlation — across a set of sessions into one averaged output, with one column per
session instead of one column per trial. Use it to see a cohort- or animal-level average rather
than reading each session's plot separately. It is **optional**: most analyses only need
per-session results, produced without touching the Group Analysis card at all.

## Before you start

Group analysis averages results that already exist — it does not compute a PSTH from raw
traces. Run **Step 1** through **Step 4** individually on every session you want to include,
with **Average Group?** left at its default `False`. See
[Your First Analysis](../tutorials/first_analysis.md) for that baseline workflow. Only once
every session has its own PSTH output on disk can it be added to a group.

Every session in the group also needs the **same recording-site labels** (the names you gave
channels in Step 1's Label Stores GUI, e.g. `DMS`, `DLS`). GuPPy checks this when you run the
average and stops with an error naming the mismatched session if the sites don't line up.

## Selecting the group

1. Open the **Group Analysis** card in the main area (collapsed by default).
2. Use its file browser to select the session folders to include. This is a separate browser
   from the one in the Individual Analysis card, so you can run an individual analysis and a
   group analysis against different folder sets without reselecting anything.

   ```{image} ../_static/images/group_analysis_card.png
   :alt: The Group Analysis card, expanded, showing its file browser for selecting session folders and the Average Group? toggle below it
   :width: 100%
   ```

3. If a session has more than one output run, a run-name selector appears for it once selected
   — pick the run to include in the average.

## Running the average

Set **Average Group? (bool)** to `True`, then click **PSTH Computation** in the sidebar — the
same button used for individual analysis. With the toggle on, it averages the selected sessions
instead of computing a new per-session PSTH.

The averaged output is written to an `average/` directory, created in the common parent
directory of the selected sessions rather than inside any single session folder.

## Visualizing the average

1. Open the **Visualization Parameters** card and set **Visualize Average Results?** to `True`.

   ```{image} ../_static/images/visualize_average_results_toggle.png
   :alt: The Visualization Parameters card showing the Visualize Average Results? toggle set to True
   :width: 100%
   ```

2. Keep the same sessions selected in the Group Analysis card's file browser — the visualizer
   uses that selection to find the averaged files.
3. Click **Open Visualization GUI** in the sidebar. The dashboard opens on the `average/`
   directory instead of a single session's output, with one line per session in place of one
   line per trial.

   ```{image} ../_static/images/group_psth_plot.png
   :alt: The Visualization dashboard's PSTH plot showing one trace per session in the averaged group
   :width: 100%
   ```

If you see an error instead, it names the specific gap:

- *"no folders are selected in the Group Analysis folder picker"* — the Group Analysis card's
  file browser is empty; select the sessions to visualize.
- *"no 'average' directory was found"* — Step 4 hasn't been run with **Average Group?** = `True`
  yet for this folder set; run it first.
- *"contains no PSTH outputs for the '...' metric"* — Step 4's average was run for a different
  metric (z-score vs. ΔF/F) than the one selected here; either re-run the average with the
  matching metric or change **z-score or ΔF/F? (for visualization)**.

## What lands on disk

In the `average/` directory:

| File | Contents |
|------|----------|
| `storesList.csv` | The store mapping the group was averaged under |
| `<event>_<site>_<metric>.h5` | Group PSTH: one column per session |
| `peak_AUC_<event>_<site>_<metric>.h5` / `.csv` | Every session's peak/AUC rows concatenated |
| `freqAndAmp_<metric>.h5` / `.csv` | One row per session |
| `cross_correlation_output/corr_*.h5` | One column per session |

See [Output data model](../reference/outputs.md#group-analysis-the-average-directory) for the
full file layout, including the empty placeholder files also written there.

## Re-running

Group averaging fully recomputes `average/` from the selected sessions' current outputs each
time — nothing compounds. To add a session, re-analyze one, or drop one from the group, adjust
the selection in the Group Analysis card and click **PSTH Computation** again.

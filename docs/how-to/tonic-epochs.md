# Measure tonic signal levels across a session

Some experiments ask what the signal level *is* during each phase of a session
rather than how it responds to an event — a drug injected partway through, an
optogenetic manipulation held for minutes, a state change scored from behavior.
This guide shows how to name those phases and get the mean z-score and ΔF/F of each.

One **optional** step handles this, and it sits between *Remove Artifacts* and
*Step 4: PSTH Computation* in the sidebar:

- **Define Tonic Epochs** — name the windows and compute their means.

Naming and computing happen together: **Save** writes the windows and the means in
one go.

If your question is event-triggered, skip it. PSTH computation is unaffected, and
nothing downstream requires it.

## Before you start

Run **Step 1: Label Stores**, **Step 2: Read Raw Data**, and **Step 3: Preprocess**
on the sessions you want to measure. The traces you average are the ones Step 3
produced.

If the recording also needs artifact removal, run **Select Artifact Windows** and
**Remove Artifacts** first. The means are computed from the traces on disk when you
save.

For a recording with a step change partway through — a drug injection, say — set
**Control Fit Window** to `baseline epoch` in Step 3 and give it a window from
before the change. A full-trace fit treats the step as part of the control-to-signal
relationship and absorbs most of it, leaving little for the epoch means to measure.
See the [parameters reference](../reference/parameters.md).

## Defining the epochs

1. Select your session folder(s) and output run(s) on the homepage as usual.
2. Click **Define Tonic Epochs**. A page opens in a new browser tab showing the
   z-score and ΔF/F traces for one recording site on a shared time axis. The main
   GuPPy tab stays responsive.

   ```{image} ../_static/images/define_tonic_epochs_button.png
   :alt: The sidebar with the optional Define Tonic Epochs step positioned between Remove Artifacts and Step 4 PSTH Computation
   :width: 50%
   ```

   ```{image} ../_static/images/define_tonic_epochs.png
   :alt: The Define Tonic Epochs page: a Recording site selector above the z-score and delta F over F traces of a bolus-injection recording, with the baseline, drug, and washout windows shaded orange; below them one row per epoch giving its label, start, and end, the Add epoch and Apply to all recording sites buttons, and Save
   ```

3. Pick the recording site you want to measure with the **Recording site** selector.
4. Click **+ Add epoch** and enter a **label** for the phase along with its start and
   end time, in **seconds**. The shaded spans on the traces update as you type, and
   the arrow keys nudge a bound by 0.1 s at a time.

   Add an epoch for each phase and give each one a distinct label; the trash button
   on a row deletes it. For a session with an injection at 60 s and the drug clearing
   from 120 s, three equal windows placed clear of both transitions might read:

   | label | start (s) | end (s) |
   |------------|-----|-----|
   | `baseline` | 10  | 50  |
   | `drug`     | 70  | 110 |
   | `washout`  | 138 | 178 |

5. If the manipulation reaches every recording site at the same time — a systemic
   injection usually does — click **Apply to all recording sites** to copy the
   current site's epochs to the others.
6. Click **Save**.

The labels you type are the identifiers you see in the results: on the bar chart, in
the table, and in the baseline selector.

Save checks every recording site before writing anything, so a rejected window
leaves the run folder untouched.

Leave a site's rows empty to skip it. The page says so in place of the rows, and no
files are written for that site.

Re-opening the page shows the epochs you saved, so you can adjust a bound without
starting over.

## Choosing epoch bounds

Each epoch's value is the plain mean of the samples inside its window, so place the
windows over the flat stretches and keep the transitions out. A window that spans a
washout ramp averages the ramp.

Windows of equal duration compare most cleanly. They may leave gaps, and they need
not cover the whole session.

A window that overhangs the start or end of the recording is accepted and clamped to
the samples available. Samples masked as NaN by artifact removal are left out of the
mean.

## Reading the results

Run **Step 5: Visualization** and open the **Tonic** tab. A run with no saved epochs
shows a short note in its place.

**Baseline epoch** picks which of your epochs the others are measured against.

The two bar panels show each epoch's change from that baseline, in mean z-score and
mean ΔF/F. The baseline epoch sits at zero by definition; the dashed line marks no
change, so bars above it rose relative to baseline and bars below it fell.

```{image} ../_static/images/tonic_results.png
:alt: The Tonic tab of the visualization: Recording site and Baseline epoch selectors above two bar panels, change in mean z-score and change in mean delta F over F, each showing the baseline epoch highlighted at zero with the drug and washout epochs raised above the dashed no-change line
```

Below the bars, the traces show the analysed windows shaded, and the table gives the
absolute means (`mean_zscore`, `mean_dff`) alongside the differences (`diff_zscore`,
`diff_dff`).

Changing **Baseline epoch** re-bases the bars and the two difference columns. The
absolute means do not move.

## Adjusting the epochs

Defining epochs is re-runnable. Open **Define Tonic Epochs** again, change the
epochs, and save; both files are rewritten for each site.

The means are computed when you save, from the traces the page loaded when it
opened. If you re-run Step 3 or **Remove Artifacts** afterwards, open the page and
save again so the stored means match the current traces.

Clearing a site's rows does not delete its files. To drop a recording site's epochs,
delete its two files from the run folder.

## What lands on disk

Per recording site with at least one epoch window, in the output run folder:

| File | Contents |
|------|----------|
| `tonic_epochs_<site>.csv` | The epoch windows: `label`, `start`, `end` |
| `tonic_<site>.h5` | Each epoch's `mean_zscore` and `mean_dff`, indexed by label |

Both are rewritten each time you save.

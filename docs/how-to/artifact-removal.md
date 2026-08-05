# Remove artifacts from a recording

Some recordings contain stretches that should not reach the analysis at all — a
knocked patchcord, a chewing bout, an electrical transient. This guide shows how
to mark those periods and exclude them.

Two **optional** steps handle this, and they sit between *Step 3: Preprocess* and
*Step 4: PSTH Computation* in the sidebar:

- **Select Artifact Windows** — mark the contaminated periods. No computation runs.
- **Remove Artifacts** — apply them.

If your recordings are clean, skip both. Nothing downstream requires them.

For background on which artifacts this addresses and which are corrected
automatically, see the [artifacts explainer](../explanation/artifacts.md).

## Before you start

Run **Step 3: Preprocess** first. Both optional steps read its outputs, and the
traces you mark on are the ones it produced. If you open Select Artifact Windows
before Step 3 has run, GuPPy tells you so.

## Marking the periods

1. Select your session folder(s) and output run(s) on the homepage as usual.
2. Click **Select Artifact Windows**. A page opens in a new browser tab showing
   the control, signal, and fitted-control traces for one recording site. The main
   GuPPy tab stays responsive.

   ```{image} ../_static/images/select_artifact_windows_button.png
   :alt: The sidebar with the two optional artifact steps, Select Artifact Windows and Remove Artifacts, positioned between Step 3 Preprocess and Step 4 PSTH Computation
   :width: 50%
   ```

3. Pick the recording site you want to mark with the **Recording site** selector.
4. Click **+ Add period** and enter the start and end time of the contaminated
   stretch, in **seconds**. The shaded spans on the traces update as you type, so you
   can check the marking against the data — and the arrow keys nudge a bound by 0.1 s
   at a time if you need to fine-tune an edge.

   Add a period for each contaminated stretch; the trash button on a row deletes it.

5. If the same artifact appears across every recording site — a motion artifact
   usually does, since it hits all sites at the same instant — click **Apply to all
   recording sites** to copy the current site's periods to the others.
6. Choose the **Removal method** (see below).
7. Click **Save**.

You are marking the periods to **remove**. Everything outside them is kept, so
marking nothing keeps the entire recording.

Re-opening the page later shows the periods you marked, so you can widen one or add
another without starting over.

## Choosing a removal method

**replace with NaN** (the default) keeps the trace at its original length and
masks the marked samples with NaN, which downstream analyses treat as missing.
The recording timeline is preserved.

**concatenate** drops the marked sections and splices the surviving ones together,
so the resulting trace is shorter than the input. This re-times the kept samples
onto a new timeline, which breaks alignment to the acquisition clock. It is
unsupported by NWB export and cannot be combined with cross-correlation. Prefer
*replace with NaN* unless you have a specific reason not to.

## Applying the removal

Click **Remove Artifacts**. A progress bar appears in the sidebar directly below
the button and fills as the work runs.

The step recomputes the control fit, z-score, and dF/F with the marked periods
excluded, so it takes about as long as Step 3 itself.

When it finishes, GuPPy opens a review tab showing the cleaned traces, so you can
confirm the result.

If you click Remove Artifacts without having saved any periods, GuPPy tells you
which run is unmarked.

## Adjusting a marked period

Marking is re-runnable. Open **Select Artifact Windows** again, change the periods,
save, and click **Remove Artifacts** again. Each run recomputes from the raw data,
so removals do not compound.

## What lands on disk

Per recording site, in the output run folder:

| File | Contents |
|------|----------|
| `coordsForPreProcessing_<site>.npy` | The periods to **keep** (the complement of what you marked) |

`GuPPyParamtersUsed.json` records what was applied to the run: `removeArtifacts`
becomes `true` once Remove Artifacts has run, and `artifactsRemovalMethod` records
the method you chose.

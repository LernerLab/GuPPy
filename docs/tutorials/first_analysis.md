# Your First Analysis

This tutorial walks through a complete GuPPy pipeline run from raw CSV data to a PSTH plot. You will use the small sample dataset that lives inside the GuPPy repository (under Git LFS), so the only setup is the source-install steps below.

By the end you will have:

- Launched the GuPPy GUI
- Selected your data and reviewed the analysis parameters
- Labeled your channels with stores
- Loaded the raw data into HDF5
- Preprocessed the signal
- Computed the PSTH
- Visualized the results

## Prerequisites

- **GuPPy installed from source, with Git LFS.** This tutorial uses sample CSV files that live in the GuPPy repository under `stubbed_testing_data/csv/sample_data_csv_1/`, which is tracked by [Git LFS](https://git-lfs.com). A plain `git clone` will fetch only LFS pointer files (a few hundred bytes each) instead of the real CSVs, so you need both Git LFS installed on your machine and a clone with the relevant LFS payload pulled in:

  ```bash
  # Install Git LFS once per machine; see https://git-lfs.com for OS-specific instructions.
  git lfs install

  git clone https://github.com/LernerLab/GuPPy.git
  cd GuPPy
  git lfs pull --include="stubbed_testing_data/csv/sample_data_csv_1/*"
  pip install -e .
  ```

  See [installation](../installation.md) for the full installation guide, including how to set up a conda environment first. The plain `pip install guppy-neuro` path also works for installing the GUI itself, but you would still need to clone the repo (with Git LFS) separately to access the sample data.

- The sample data lives at `stubbed_testing_data/csv/sample_data_csv_1/` inside the cloned repository. It contains three CSV files:

  | File | Description |
  |------|-------------|
  | `Sample_Control_Channel.csv` | Isosbestic (405 nm) control channel |
  | `Sample_Signal_Channel.csv` | Calcium-dependent (470 nm) signal channel |
  | `Sample_TTL.csv` | Event timestamps |

  Each signal CSV has three columns: `timestamps` (seconds), `data` (fluorescence), and `sampling_rate`. The TTL CSV has a single `timestamps` column.

## Step 0: Launch GuPPy and choose your data

```bash
guppy
```

A browser tab opens showing the GuPPy dashboard.

```{image} ../_static/images/01_homepage.png
:alt: The GuPPy homepage on first launch: the sidebar of pipeline steps on the left, and on the right the Data Source toggle above an open Root Folder Selection card asking for the input and output root folders
:align: center
```

The page is split into a **sidebar on the left** and a **main area on the right**. The sidebar lists the pipeline buttons in run order, from *Label Stores* at the top through *Export to NWB* at the bottom, with a status line under each step that says whether it has run. The main area is where you select your data folder and configure parameters; settings are grouped into collapsible cards, of which this tutorial uses *Root Folder Selection*, *Input Folder Selection*, *Output Folder Selection* and *Parameter Selection*. The **Step N** labels in the sidebar (Label Stores through Export to NWB) match the numbered pipeline steps below, and the unnumbered buttons between them are optional steps this tutorial does not use; choosing your data and setting parameters in the main area, covered here in Step 0, is the setup that precedes them.

Before running the pipeline you pick the session folder you want to analyze, then look over (but not change) the analysis parameters that the rest of the pipeline will use.

### Select your data

Open the **Root Folder Selection** card at the top of the page. It holds the two folders a project keeps for the life of its analyses, and it opens itself while either is unset. Set the **Input Root Folder** — the folder your session folders live under — to `stubbed_testing_data/csv/`, and the **Output Root Folder** to wherever you want results written; anywhere outside your data is fine. GuPPy mirrors each session's path below the input root into the output root, so these two decide the shape of your results tree. It remembers them, so the card folds away next time.

In the **Input Folder Selection** card's **Session folders** browser, navigate to `stubbed_testing_data/csv/sample_data_csv_1/` and click **`>>`** to move that folder into the **Selected files** pane on the right. The browser supports selecting multiple session folders at once for batch analysis; for this tutorial we are running a single session. Every session you select has to sit under the input root folder.

```{image} ../_static/images/02_data_selection.png
:alt: The GuPPy homepage with Root Folder Selection folded away and the Input Folder Selection card open below it, its Session folders browser listing sample_data_csv_1 among the sample sessions
:align: center
```

The **Data Source** toggle above the cards lets you switch between `Local` (the default, file-system browsing) and `DANDI` (streaming NWB sessions directly from DANDI). We are using local files here; see [Analyze data streamed from the DANDI Archive](../how-to/analyze-dandi-data.md) for the DANDI workflow.

### Set parameters

The **Parameter Selection** card below lists the parameters that drive the rest of the pipeline, grouped into titled sections named for the operation each one configures. Every parameter carries a **?** beside its label that explains what it does. **For this tutorial the defaults are fine**, so you do not need to change anything; the screenshot below is for orientation, not for hunting and clicking.

```{image} ../_static/images/02_parameters.png
:alt: GuPPy Parameter Selection card showing the Parallel Execution, Control Channel Fitting, Signal Filtering and Z-Score Normalization sections
:align: center
```

Each parameter is documented in the parameter reference.

<!-- TODO: link to docs/reference/parameters.md (per-parameter reference: type, default, valid range, effect) once that page exists. -->

:::{note}
There is no separate "save parameters" action. Each pipeline step records the parameters it ran with into its output folder as `GuPPyParamtersUsed.json`, so that file always reflects the configuration that was actually applied. If you come back later, reselect an existing output folder under **Output Folder Selection**, and GuPPy reloads that run's saved parameters into the form so you can continue (for example, run Preprocess and PSTH) without accidentally overwriting them.
:::


## Step 1: Label your channels

A **store** is the human-readable label GuPPy uses for one of your data channels. Raw acquisition files come with cryptic, format-specific names (here, the CSV filenames `Sample_Control_Channel`, `Sample_Signal_Channel`, `Sample_TTL`); GuPPy needs you to map each one to a meaningful name like `control_A`, `signal_A`, or `RewardPort`. Those mapped names are what every downstream step (preprocessing, PSTH, plots, group analysis) refers to. The mapping is saved as `storesList.csv` inside a run folder created in the output root folder.

Click **Label Stores** in the sidebar. A Label Stores tab opens beside the form, with the Label Stores panel for the selected folder. With several folders selected, the panel has a tab for each.

```{image} ../_static/images/02_label_stores.png
:alt: The Label Stores page with the three sample CSV channels listed as available options
:align: center
```

The three CSV filenames appear in the left list (**Filter available options**) of the **Store Selection** widget. Walk through these substeps:

1. **Move all three channels to the right list.** Click each of `Sample_Control_Channel`, `Sample_Signal_Channel`, `Sample_TTL` in the left list and use the `>>` button to move it to the right. (Or shift-click to select all three at once before clicking `>>`.)

2. **Click "Select Stores".** A *Label Stores* section appears below, with one row per channel under the headings **Store**, **Type** and **Name or Signal**. Each row has a **Type** dropdown, and beside it either a text box or, for a control row, a picker of the signal rows.

   ```{image} ../_static/images/02b_label_stores_configured.png
   :alt: The Label Stores page after clicking Select Stores, showing the Label Stores section with Store, Type and Name or Signal columns for each of the three channels
   :align: center
   ```

3. **Fill out one row per channel.** For a signal row, the **Name or Signal** text box takes a *recording-site identifier*, not a description of the channel's role. For the control row, pick the signal it belongs to: GuPPy uses that pairing to match the isosbestic (405 nm) control trace with the calcium-dependent (470 nm) signal trace from the same fiber, then fits and subtracts one from the other during preprocessing to remove motion artifacts and photobleaching. In a multi-fiber recording, each fiber gets its own identifier (e.g. `DMS`, `DLS`) so each control is paired with its own signal. In this single-fiber tutorial we use `A`. The event TTL row uses a free-form event name instead.

   <!-- TODO: link to docs/explanation/fiber_photometry.md (signal vs. isosbestic, z-score methods) once that page exists, for the deeper "why" of pairing. -->


   | Channel | Type | Name or Signal |
   |---------|------|----------------|
   | `Sample_Control_Channel` | `Control` | `Sample_Signal_Channel – A` |
   | `Sample_Signal_Channel` | `Signal` | `A` |
   | `Sample_TTL` | `Event TTLs` | `RewardPort` |

   GuPPy combines each row's type and name into the final store: `Signal` + `A` becomes `signal_A`, the control paired with it becomes `control_A`, and the event TTL keeps the name as-is (`RewardPort`). If a control row is left on `— select signal —`, GuPPy rejects the configuration with an alert asking you to select the signal each control belongs to.

4. **Click "Show Selected Configuration".** This applies your row entries to the JSON editor below, which should now read something like:

   ```json
   {
     "Sample_Control_Channel": ["control_A"],
     "Sample_Signal_Channel": ["signal_A"],
     "Sample_TTL": ["RewardPort"]
   }
   ```

5. **Choose where to save the store labels.** Leave **Create New Run** selected, and leave the **Run Name** GuPPy fills in, `1`.

   This choice picks the **run folder** for the entire analysis pipeline. From this point on, every downstream step (Read Raw Data, Preprocess, PSTH Computation, Visualization) writes its outputs (HDF5 files, PSTH results, plots) into that directory and reads `storesList.csv` from it to know which raw channel maps to which store. **Create New Run** makes a fresh directory named `output_<run name>/` inside the session's mirror in the output root folder — for this tutorial, `<output root folder>/sample_data_csv_1/`. The run name GuPPy fills in is the next free integer, so the first run gets `output_1`, the second `output_2`, and so on. Your raw session folder is left exactly as it was.

   :::{note}
   The other option, **Overwrite Existing Run**, is for re-running on a session that already has a run folder. It lets you pick an existing `output_<run name>/` under **Run to Overwrite**, deletes everything inside it (the previous `storesList.csv` plus any HDF5 and PSTH results from the previous run), and starts that directory over fresh. Pick it only when you genuinely want that destructive behavior. For the tutorial, ignore it.
   :::

6. **Click Save.** GuPPy creates the run folder (`<output root folder>/sample_data_csv_1/output_1/`) and writes `storesList.csv` into it. The downstream steps will read and write inside this folder.

You can close this Label Stores tab and return to the original homepage tab. Open **Output Folder Selection** and, under **Existing Runs (Steps 2–5)**, select `output_1` under `sample_data_csv_1`. Every step from here on acts on the runs selected there.

## Step 2: Load the raw data

Click **Read Raw Data**. A progress bar appears in the sidebar directly below the button and fills as the work runs. That bar is the primary signal that the step is in progress.

```{image} ../_static/images/04_read_progress.png
:alt: GuPPy homepage sidebar with Read Raw Data running, its progress bar just appeared below the button
:align: center
```

Every step has a status line under its button. It reads *Not run yet* until you run the step, shows the progress bar and *Running…* while it runs, and then reads *✓ Done at* the time it finished, or *✗ Failed at* if it stopped with an error.

GuPPy loads each CSV file and writes the data into the output folder you created in Step 1, one HDF5 file per store (so for this tutorial: `sample_data_csv_1/output_1/control_A.hdf5`, `.../signal_A.hdf5`, `.../RewardPort.hdf5`). Each file holds the channel's `data`, `timestamps`, and `sampling_rate` datasets plus a few metadata fields. HDF5 is a binary format that stores large numerical arrays efficiently and supports partial reads, which speeds up the later pipeline steps.

When the step finishes, its status line changes to *✓ Done at* the time it finished. Confirmation messages are also logged to the terminal where you launched `guppy`.

## Step 3: Preprocess the signal

Click **Preprocess**. As with Read Raw Data, a progress bar appears in the sidebar directly below the button and fills as the work runs.

```{image} ../_static/images/05_preprocess_progress.png
:alt: GuPPy homepage sidebar with the Preprocess progress bar partially filled
:align: center
```

GuPPy runs the following on the raw signal:

1. Trims the first few seconds from both channels (the *Eliminate First Few Seconds (s)* parameter, internally `timeForLightsTurnOn`).
2. Applies a moving-average filter to reduce high-frequency noise (the *Window for Moving Average Filter* parameter, internally `filter_window`).
3. Fits the control channel to the signal channel, then subtracts it. This removes motion artifacts and photobleaching that affect both channels equally. The fitting method is configurable (the *Control Channel Fitting Method* parameter, internally `control_fit_method`): the default `IRWLS` is a robust regression that down-weights outliers, with ordinary least-squares (`OLS`) regression available as an alternative.
4. Computes the z-score and the dF/F (delta F over F) of the corrected signal.

The results are written into the same output folder as Step 2, in four new HDF5 files per recording site. You do not choose the location or the file names; they follow a fixed convention:

| File | Contents |
|------|----------|
| `z_score_A.hdf5` | The z-scored trace |
| `dff_A.hdf5` | The dF/F trace |
| `cntrl_sig_fit_A.hdf5` | The fitted control trace (used internally and for artifact-removal plots) |
| `timeCorrection_A.hdf5` | Corrected timestamps, sampling rate, and a few related metadata fields |

Step 3 also rewrites the per-channel files under their store labels. The [Output data model](../reference/outputs.md) reference lists every file and the datasets inside it, and [Custom Plots from GuPPy Outputs](custom_plots.md) shows how to load them in Python.

When preprocessing finishes, a Preprocessing tab opens beside the form with the results. If you close it, the **View Results →** button in the status line under **Preprocess** opens it again. Pick a recording site and its five traces stack up together — control, signal, signal with the fitted control, z-score, and ΔF/F — sharing one time axis, so zooming any of them zooms them all. You can switch back to the Home tab and keep working, and the results tab stays open.

<!-- TODO: add a screenshot of the preprocessing results tab that opens after preprocess completes (e.g. 05_preprocess_plot.png). It is a tab inside the homepage, so the screenshot script will need to run Preprocess to capture it. -->

### Optional: remove artifacts

If the traces in that results tab show stretches you want excluded — a knocked patchcord, a chewing bout — two optional steps sit between Step 3 and Step 4 for that: **Select Artifact Windows**, where you mark the contaminated periods, and **Remove Artifacts**, which applies them. The sample data for this tutorial is clean, so you can skip both. See [Remove artifacts from a recording](../how-to/artifact-removal.md) when you need them.


## Step 4: Compute the PSTH

Click **PSTH Computation**. As with the previous two steps, a progress bar appears in the sidebar directly below the button and fills as the work runs.

```{image} ../_static/images/06_psth_progress.png
:alt: GuPPy homepage sidebar with the PSTH Computation progress bar partially filled
:align: center
```

GuPPy aligns the z-scored trace to each event timestamp in `Sample_TTL.csv`, extracts the window defined by the *Seconds Before 0* / *Seconds After 0* parameters (internally `nSecPrev`, `nSecPost`) around each event, and averages across all trials. The result is a peri-stimulus time histogram (PSTH).

The default window is -10 to +20 seconds. With the sample data you will get a small number of trials (the TTL file has just a handful of timestamps), so the average will be noisy. This is expected for a minimal example dataset.

The outputs land in the same `sample_data_csv_1/output_1/` directory you have been using since Step 1, with one set of files per (event, recording site) pair. For this tutorial that is the single pair `(RewardPort, A)`:

| File | Contents |
|------|----------|
| `RewardPort_A.hdf5` | The peri-event timestamps (the x-axis of the PSTH) |
| `RewardPort_A_z_score_A.h5` | The PSTH dataframe: one column per trial, plus `mean` and `err` (standard error) columns and the `timestamps` column |
| `RewardPort_A_baselineUncorrected_z_score_A.h5` | Same dataframe before baseline correction was applied (kept for inspection) |
| `peak_AUC_RewardPort_A_z_score_A.csv` and matching `.h5` | Peak amplitude and area-under-curve for the trial-mean PSTH |

The visualization step in Step 5 reads these files; you do not need to inspect them by hand. If you do want to open them yourself, the [Output data model](../reference/outputs.md) reference gives the columns of each table and the datasets in each HDF5 file, and [Custom Plots from GuPPy Outputs](custom_plots.md) walks through loading them and plotting them yourself.

## Step 5: Visualize the results

Click **Open Visualization** in the sidebar. A Visualization tab opens with the dashboard for this session, organized into tabs. This tutorial uses the first two, **PSTH** and **Heat Map**.

A **Metric** selector sits in the *Display Settings* card at the top of both tabs. It lists the metrics Step 4 actually computed for this run, so here it offers only `Z-Score`; had you run Step 4 with `Both`, you could switch both tabs between the two without re-running the step. Whichever metric is selected also labels the y axis of every PSTH plot.

```{image} ../_static/images/03_visualization.png
:alt: GuPPy's Visualization page showing the PSTH tab with the RewardPort event selected
:align: center
```

The **PSTH** tab is the default view. A *Display Settings* card at the top holds what every plot on the tab shares: the *Metric*, the *Saved Width* that a figure saved with *Save As…* is rendered at, the plot *Height*, and a *Hide Minor Ticks* checkbox under *Style*. On screen, each plot spans the full width of the window.

Below it, three cards each hold one plot, with a row of labeled fields above it:

- *Single-Event PSTH*: *Event* picks which TTL channel to align to (here `RewardPort`), and *View* shows the trial average (`mean`) or all trials at once (`All`).
- *Multi-Event Comparison*: overlays the mean traces of the events you move into the right-hand list of the *Events* panel beside the plot. *Palette* sets their colours, and the *Colors* button beside it opens a colour picker for each event.
- *Selected Trials*: overlays the trials of the *Single-Event PSTH* event that you move into the right-hand list of the *Trials* panel beside the plot. *Show* draws their *Mean*, the trials themselves (*Just Trials*), or both. Each trial reads as its number and event time, such as `3 · 113.40 s`.

Each card also has its own *X Range* and *Y Range* fields and a *Save Plot* field, which holds a format menu and a *Save As…* button. Each side panel can be dragged wider by the handle between it and the plot, or hidden with the button on that handle. With a TTL file containing only a handful of timestamps, the average will be noisy; this is expected for the minimal sample dataset.

The **Heat Map** tab shows trials stacked vertically with colour encoding the metric, which is the better view for inspecting trial-to-trial variability. Its own *Display Settings* card holds the *Metric*, *Saved Width*, *Height* and *Style* checkboxes, and the *Trial Heat Map* card below it has *Event*, *Color Map* (`plasma`, `viridis`, …), *X Range*, *Color Scale* and *Save Plot* fields, with a *Trials* panel beside the plot.

For this tutorial, the goal is just to reach a rendered PSTH; feel free to play with the controls. A deeper walkthrough of the visualization options will live in an upcoming how-to guide.

## Next steps

- See [Custom Plots from GuPPy Outputs](custom_plots.md) to open this run folder in Python and build your own figures from it.
- See [Export a session to NWB](../how-to/export-to-nwb.md) to package this run — raw traces, events, and every derived result — into a single shareable NWB file.
- See [How-to Guides](../how-to/index.md) for task-specific instructions (TDT data ingestion, artifact removal, group analysis, etc.).
- See [Explanation](../explanation/index.md) for background on the isosbestic correction and z-score methods.

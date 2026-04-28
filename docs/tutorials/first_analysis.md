# Your First Analysis

This tutorial walks through a complete GuPPy pipeline run from raw CSV data to a PSTH plot. You will use the small sample dataset that lives inside the GuPPy repository, so the only setup is cloning the repo.

By the end you will have:

- Launched the GuPPy GUI
- Assigned names to your recording channels (storenames)
- Ingested raw data into HDF5
- Preprocessed the signal
- Computed a PSTH
- Opened the visualization panel

## Prerequisites

- **GuPPy installed from source.** This tutorial uses sample CSV files that live in the GuPPy repository but are not included in the PyPI package, so you need a local clone:

  ```bash
  git clone https://github.com/LernerLab/GuPPy.git
  cd GuPPy
  pip install -e .
  ```

  See the [README](https://github.com/LernerLab/GuPPy#installation) for the full installation guide, including how to set up a conda environment first. The plain `pip install guppy-neuro` path also works for installing the GUI itself, but you would still need to clone the repo separately to access the sample data.

- The sample data lives at `stubbed_testing_data/csv/sample_data_csv_1/` inside the cloned repository. It contains three CSV files:

  | File | Description |
  |------|-------------|
  | `Sample_Control_Channel.csv` | Isosbestic (405 nm) control channel |
  | `Sample_Signal_Channel.csv` | Calcium-dependent (470 nm) signal channel |
  | `Sample_TTL.csv` | Event timestamps |

  Each signal CSV has three columns: `timestamps` (seconds), `data` (fluorescence), and `sampling_rate`. The TTL CSV has a single `timestamps` column.

## Step 1: Launch the GUI

```bash
guppy
```

A browser tab opens showing the GuPPy dashboard. The main panel has three collapsible cards: **Individual Analysis**, **Group Analysis**, and **Visualization Parameters**.

```{image} ../_static/images/01_homepage.png
:alt: GuPPy Input Parameters GUI homepage
:align: center
```

## Step 2: Set the analysis parameters

Inside the **Individual Analysis** card, use the file selector to navigate to `stubbed_testing_data/csv/sample_data_csv_1/` and select that folder.

Set the following parameters (leave everything else at its default):

| Parameter | Value | Why |
|-----------|-------|-----|
| Isosbestic Control Channel? | True | The sample data includes a 405 nm control channel. |
| Eliminate first few seconds | 1 | Drops the first second of data, which can contain transient noise from the LED turning on. |
| Window for Moving Average filter | 100 | Smoothing window in data points. The default works well for ~1 kHz recordings. |

The z-score and PSTH parameters can remain at their defaults for this tutorial.

## Step 3: Label your channels

A **storename** is the human-readable label GuPPy uses for one of your data channels. Raw acquisition files come with cryptic, format-specific names (here, the CSV filenames `Sample_Control_Channel`, `Sample_Signal_Channel`, `Sample_TTL`); GuPPy needs you to map each one to a meaningful name like `control_A`, `signal_A`, or `RewardPort`. Those mapped names are what every downstream step (preprocessing, PSTH, plots, group analysis) refers to. The mapping is saved as `storesList.csv` inside an output folder created next to the session.

Click **Open Storenames GUI** in the sidebar. A new browser tab opens with the Storenames panel for the selected folder.

```{image} ../_static/images/02_storenames.png
:alt: Storenames GUI with the three sample CSV channels listed as available options
:align: center
```

The three CSV filenames appear in the left list (**Filter available options**) of the **Store Names Selection** widget. Walk through these substeps:

1. **Move all three channels to the right list.** Click each of `Sample_Control_Channel`, `Sample_Signal_Channel`, `Sample_TTL` in the left list and use the `>>` button to move it to the right. (Or shift-click to select all three at once before clicking `>>`.)

2. **Click "Select Storenames".** A new *Configure Storenames* section appears below, with one row per channel. Each row has a **Type** dropdown and a **Name** text field.

   ```{image} ../_static/images/02b_storenames_configured.png
   :alt: Storenames GUI after clicking Select Storenames, showing the Configure Storenames section with a Type dropdown and Name field for each of the three channels
   :align: center
   ```

3. **Fill out one row per channel.** The **Name** field is a *brain region identifier*, not a description of the channel's role. The control and signal rows must use the **same Name** because GuPPy uses that match to pair the isosbestic (405 nm) control trace with the calcium-dependent (470 nm) signal trace from the same fiber, then fits and subtracts one from the other during preprocessing to remove motion artifacts and photobleaching. In a multi-fiber recording, each fiber gets its own identifier (e.g. `DMS`, `DLS`) so each control is paired with its own signal. In this single-fiber tutorial we use `A`. The event TTL row uses a free-form event name instead.

   <!-- TODO: link to docs/explanation/fiber_photometry.md (signal vs. isosbestic, z-score methods) once that page exists, for the deeper "why" of pairing. -->


   | Channel | Type | Name |
   |---------|------|------|
   | `Sample_Control_Channel` | `control` | `A` |
   | `Sample_Signal_Channel` | `signal` | `A` |
   | `Sample_TTL` | `event TTLs` | `RewardPort` |

   GuPPy combines the dropdown and text field into the final storename: `control` + `A` becomes `control_A`, `signal` + `A` becomes `signal_A`, and the event TTL keeps the name as-is (`RewardPort`). If the control and signal Name fields differ, saving will fail with `Mismatched signal/control region pairs — Every 'signal_<region>' must have a matching 'control_<region>'.`

4. **Click "Show Selected Configuration".** This applies your row entries to the JSON editor below, which should now read something like:

   ```json
   {
     "Sample_Control_Channel": ["control_A"],
     "Sample_Signal_Channel": ["signal_A"],
     "Sample_TTL": ["RewardPort"]
   }
   ```

5. **Choose the output directory.** Use the **over-write storeslist file or create a new one?** menu button and select `create_new_file`.

   Despite the menu's name, this choice does more than name a file. It picks the **output directory** for the entire analysis pipeline. From this point on, every downstream step (Read Raw Data, Preprocess, PSTH Computation, Visualization) writes its outputs (HDF5 files, PSTH results, plots) into that directory and reads `storesList.csv` from it to know which raw channel maps to which storename.

   - `create_new_file` makes a fresh subdirectory inside the session folder, named `<session>_output_<N>/` with `<N>` auto-incremented (`_output_1` on the first run, `_output_2` on the second, and so on). Pick this for a first run.
   - `over_write_file` is for re-running on a session that already has an output subdirectory. It lets you point at an existing `<session>_output_<N>/`, **deletes everything inside it** (the previous `storesList.csv` plus any HDF5 and PSTH results from the previous run), and starts that subdirectory over fresh.

6. **Click Save.** GuPPy creates the output subdirectory (e.g. `sample_data_csv_1_output_1/`) and writes `storesList.csv` into it. The downstream steps will read and write inside this folder.

You can close this Storenames tab and return to the original homepage tab to continue.

## Step 4: Read raw data

Click **Read Raw Data**. A progress bar appears in the sidebar directly below the button and fills as the work runs. That bar is the primary signal that the step is in progress.

```{image} ../_static/images/04_read_progress.png
:alt: GuPPy homepage sidebar with the Read Raw Data progress bar partially filled
:align: center
```

GuPPy loads each CSV file and writes the data into the output folder you created in Step 3, one HDF5 file per storename (so for this tutorial: `sample_data_csv_1_output_1/control_A.hdf5`, `.../signal_A.hdf5`, `.../RewardPort.hdf5`). Each file holds the channel's `data`, `timestamps`, and `sampling_rate` datasets plus a few metadata fields. HDF5 is a binary format that stores large numerical arrays efficiently and supports partial reads, which speeds up the later pipeline steps.

When the progress bar reaches 100% the step is complete. Confirmation messages are also logged to the terminal where you launched `guppy`.

## Step 5: Preprocess

Click **Preprocess**. As with Read Raw Data, a progress bar appears in the sidebar directly below the button and fills as the work runs.

```{image} ../_static/images/05_preprocess_progress.png
:alt: GuPPy homepage sidebar with the Preprocess and Remove Artifacts progress bar partially filled
:align: center
```

GuPPy runs the following on the raw signal:

1. Trims the first `timeForLightsTurnOn` seconds from both channels.
2. Applies a moving-average filter to reduce high-frequency noise.
3. Fits the control channel to the signal channel using a linear regression, then subtracts it. This removes motion artifacts and photobleaching that affect both channels equally.
4. Computes the z-score and the dF/F (delta F over F) of the corrected signal.

The results are written into the same output folder as Step 4, in four new HDF5 files per region. You do not choose the location or the file names; they follow a fixed convention:

| File | Contents |
|------|----------|
| `z_score_A.hdf5` | The z-scored trace |
| `dff_A.hdf5` | The dF/F trace |
| `cntrl_sig_fit_A.hdf5` | The fitted control trace (used internally and for artifact-removal plots) |
| `timeCorrection_A.hdf5` | Corrected timestamps, sampling rate, and a few related metadata fields |

When preprocessing finishes, GuPPy opens a matplotlib window showing the preprocessed trace plotted against time. The default is the z-score; the `plot_zScore_dff` input parameter controls this and can be set to `dff` or `Both` instead. Close the matplotlib window to return control to the GUI.

<!-- TODO: add screenshot of the matplotlib plot that pops up after preprocess completes (e.g. 05_preprocess_plot.png). This is a separate window, not part of the panel page, so the screenshot script will need a different capture path. -->


## Step 6: Compute PSTH

Click **Compute PSTH**. As with the previous two steps, a progress bar appears in the sidebar directly below the button and fills as the work runs.

```{image} ../_static/images/06_psth_progress.png
:alt: GuPPy homepage sidebar with the PSTH Computation progress bar partially filled
:align: center
```

GuPPy aligns the z-scored trace to each event timestamp in `Sample_TTL.csv`, extracts a window of `nSecPrev` seconds before and `nSecPost` seconds after each event, and averages across all trials. The result is a peri-stimulus time histogram (PSTH).

The default window is -10 to +20 seconds. With the sample data you will get a small number of trials (the TTL file has just a handful of timestamps), so the average will be noisy. This is expected for a minimal example dataset.

Output files are written to the session folder.

## Step 7: Visualize

Back on the homepage, expand the **Visualization Parameters** card. Leave both settings at their defaults: **z-score or ΔF/F?** stays at `z_score` (the metric we computed in Step 5), and **Visualize Average Results?** stays at `False`. The latter is a group-analysis feature for averaging across multiple sessions and requires `Average Group?` to have been enabled during PSTH computation; we have a single session, so it does not apply here.

Click **Open Visualization GUI** in the sidebar. A new browser tab opens with the Visualization GUI for this session, organized into two tabs.

```{image} ../_static/images/03_visualization.png
:alt: GuPPy Visualization GUI showing the PSTH tab with the RewardPort event selected
:align: center
```

The **PSTH** tab is the default view. It shows the trial-aligned trace for one event with controls running down the left:

- *Event selector* — which TTL channel to align to (here `RewardPort`).
- *X* and *Y* dropdowns — what to plot on each axis. X is typically `timestamps`, Y can be `mean` (the trial average) or an individual trial like `trial_1`.
- *X Limit* and *Y Limit* range sliders — restrict the displayed window.
- *Width Plot*, *Height Plot*, *Y Label*, *Save options* dropdowns and a *Save PSTH* button — figure dimensions and export.

On the right is a trial multi-select (*Trial # - Timestamps*) and a *Select mean and/or just trials* checkbox group, which together let you overlay any combination of individual trials and the mean. With a TTL file containing only a handful of timestamps, the average will be noisy; this is expected for the minimal sample dataset.

The **Heat Map** tab shows trials stacked vertically with colour encoding the metric, which is the better view for inspecting trial-to-trial variability. Its controls mirror the PSTH tab: event selector, *Color map* (`plasma`, `viridis`, …), width and height, save options, and a trial picker.

For this tutorial, the goal is just to reach a rendered PSTH — feel free to play with the controls. A deeper walkthrough of the visualization options will live in an upcoming how-to guide.

## Next steps

- See [How-to Guides](../how-to/index.md) for task-specific instructions (TDT data ingestion, artifact removal, group analysis, etc.).
- See [Explanation](../explanation/index.md) for background on the isosbestic correction and z-score methods.

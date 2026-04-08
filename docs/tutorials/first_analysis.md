# Your First Analysis

This tutorial walks through a complete GuPPy pipeline run from raw CSV data to a PSTH plot. You will use the small sample dataset that ships with the repository, so no data download is required.

By the end you will have:

- Launched the GuPPy GUI
- Assigned names to your recording channels (storenames)
- Ingested raw data into HDF5
- Preprocessed the signal
- Computed a PSTH
- Opened the visualization panel

## Prerequisites

- GuPPy installed (`pip install guppy-neuro`).

- The sample data lives at `stubbed_testing_data/csv/sample_data_csv_1/` inside the repository root. It contains three CSV files:

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

## Step 2: Set the analysis parameters

Expand **Individual Analysis** if it is not already open.

Use the file selector to navigate to `stubbed_testing_data/csv/sample_data_csv_1/` and select that folder.

Set the following parameters (leave everything else at its default):

| Parameter | Value | Why |
|-----------|-------|-----|
| Isosbestic Control Channel? | True | The sample data includes a 405 nm control channel. |
| Eliminate first few seconds | 1 | Drops the first second of data, which can contain transient noise from the LED turning on. |
| Window for Moving Average filter | 100 | Smoothing window in data points. The default works well for ~1 kHz recordings. |

The z-score and PSTH parameters can remain at their defaults for this tutorial.

## Step 3: Assign storenames

Click **Save to File** at the top of the GUI. This opens the Storenames panel.

GuPPy reads the CSV filenames and lists them as available channels. You need to tell it which channel is control (isosbestic), which is signal, and what the TTL represents.

Map the channels using the naming convention `control_<region>` and `signal_<region>`:

| Channel filename | Name to assign |
|-----------------|----------------|
| `Sample_Control_Channel` | `control_A` |
| `Sample_Signal_Channel` | `signal_A` |
| `Sample_TTL` | `RewardPort` |

Select each storename from the list, type its name in the text field, then click **Select Storenames**. When all three are assigned, choose **create new** and click **Save**.

GuPPy writes a `storeslist.json` file inside the session folder that records these assignments. The pipeline reads that file during preprocessing.

## Step 4: Read raw data

Click **Read Raw Data**.

GuPPy loads each CSV file and writes the data into an HDF5 file (`{session_folder}/data.hdf5`). HDF5 is a binary format that stores large numerical arrays efficiently and supports partial reads, which speeds up the later pipeline steps.

When the step completes the console will log a confirmation message.

## Step 5: Preprocess

Click **Preprocess**.

GuPPy runs the following on the raw signal:

1. Trims the first `timeForLightsTurnOn` seconds from both channels.
2. Applies a moving-average filter to reduce high-frequency noise.
3. Fits the control channel to the signal channel using a linear regression, then subtracts it. This removes motion artifacts and photobleaching that affect both channels equally.
4. Computes the z-score of the corrected signal using the selected method (standard z-score by default).

The z-scored trace is saved back into the HDF5 file.

## Step 6: Compute PSTH

Click **Compute PSTH**.

GuPPy aligns the z-scored trace to each event timestamp in `Sample_TTL.csv`, extracts a window of `nSecPrev` seconds before and `nSecPost` seconds after each event, and averages across all trials. The result is a peri-stimulus time histogram (PSTH).

The default window is -10 to +20 seconds. With the sample data you will get a small number of trials (the TTL file has just a handful of timestamps), so the average will be noisy. This is expected for a minimal example dataset.

Output files are written to the session folder.

## Step 7: Visualize

Expand **Visualization Parameters**. Set **Visualize Average Results?** to `True` and click **Visualize**.

The visualization panel opens and shows the average PSTH trace with a shaded confidence interval. The x-axis is time relative to the event, and the y-axis is z-score.

## Next steps

- See [How-to Guides](../how-to/index.md) for task-specific instructions (TDT data ingestion, artifact removal, group analysis, etc.).
- See [Explanation](../explanation/index.md) for background on the isosbestic correction and z-score methods.

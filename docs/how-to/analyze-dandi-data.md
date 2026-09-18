# Analyze data streamed from the DANDI Archive

Point GuPPy at a public dandiset on the [DANDI Archive](https://dandiarchive.org)
and stream the NWB assets you select straight through the pipeline. Nothing is
downloaded except GuPPy's own outputs. Follow this to reanalyze published fiber
photometry data, or to check your own dataset after uploading it. If your data is
already on disk, follow [Your First Analysis](../tutorials/first_analysis.md)
instead.

## Before you start

Get a DANDI API key and set it before you launch GuPPy. Streaming an asset's data
needs one **even for a public dandiset**. Copy your key from your account page on
dandiarchive.org and export it in the shell you launch `guppy` from:

```bash
export DANDI_API_KEY=<your key>
```

Or run `dandi login` once to store the key in your keyring, where GuPPy also
finds it.

Set it up front even though the panel appears to work without it. Searching the
archive, listing a dandiset's assets and previewing a file all read public data,
so a missing key stays invisible until **Step 1: Label Stores**, which stalls
while the terminal that launched `guppy` waits at a token prompt you cannot see.

GuPPy reads the dandiset's `draft` version.

## Choosing a dandiset and assets

In **Input Folder Selection**, set **Data Source** to `dandi`. The local file
browser is replaced by the DANDI panel, which walks you through three screens: a
list of dandisets, one dandiset's page, and that dandiset's files.

```{image} ../_static/images/dandi_source_selection.png
:alt: The Input Folder Selection card with the Data Source toggle switched from local to dandi, showing the DANDI source panel with its search box holding the word photometry, the filter checkbox beneath it, and a table of matching dandisets
:width: 100%
```

### Finding a dandiset

1. Start from the list already on screen. The panel opens having searched the
   archive for `photometry`, so you have candidates without typing anything.

   To look for something else, replace the term and search again, or type a
   six-digit Dandiset ID to go straight to one dandiset. Click a column header to
   sort the table.

   ```{image} ../_static/images/dandi_catalog_search.png
   :alt: The DANDI panel's list screen, with photometry in the search box, the "Filter dandisets for GuPPy-readable fiber photometry (slow)" checkbox below it, and a sortable table of dandisets giving each one's identifier, name, species, subject count, file count and size
   :width: 100%
   ```

2. Tick **Filter dandisets for GuPPy-readable fiber photometry** to find out
   which of the listed dandisets GuPPy can actually read. The search matches only
   what a dataset's authors wrote about it — its title, abstract and keywords —
   so mentioning photometry is not the same as holding it; the filter settles the
   question by reading the listed dandisets' NWB files and dropping the ones
   GuPPy cannot use.

   Expect minutes where the search took seconds, which is why it starts off.
   Watch the progress bar, and click **Stop** to end it early. Filter a narrow
   search rather than a broad one: filtering one result is seconds where
   filtering twenty is minutes. Verdicts are remembered between sessions, so
   filtering the same datasets again is immediate, and unticking the box brings
   the unread dandisets back with the verdicts still in hand.

   Treat what survives as a floor rather than the whole truth. A dataset can
   record real fiber photometry and still be dropped, if it stores its traces as
   a type GuPPy does not read.

### Reading a dandiset's page

Click a row to open that dandiset's page, and read it to decide whether the
dataset is the one you want: its citation details, license, species, subject and
file counts, size, keywords and abstract, with a link to its page on
dandiarchive.org. Click **← Back to results** to return to the list with your
search intact.

When you have found the dandiset you want, click **Analyze this dandiset** to
open its files.

### Selecting the NWB files

1. Click **Scan for fiber photometry** before you go hunting. A dandiset usually
   holds far more assets than recordings, and the folder names give no sign of
   which is which — in `000971`, 63 of 4139 assets carry traces, spread across 40
   of its 168 subject folders.

   The scan reads every listed file's header straight from the archive, then
   ticks **Show only files GuPPy can read** so the tree holds just those. Expect
   a couple of seconds for a typical dandiset and well under a minute for one the
   size of `000971`. Untick the box to bring the whole listing back.

2. Browse the subject folders and select one or more NWB files. Navigation works
   as in local mode — click a folder to descend, Ctrl/Cmd-click to multi-select.
   Only `.nwb` assets are listed.

   ```{image} ../_static/images/dandi_asset_browser.png
   :alt: The DANDI panel's files screen for Dandiset 000971 after a scan, with a Back to dandisets link, the "Show only files GuPPy can read" checkbox ticked, a status line reading "Showing 63 of 4139 NWB asset(s) in the tree below. Scanned 4139 file(s): 63 hold fiber photometry", the surviving subject folders listed in the File Browser pane, and sub-112-283/sub-112-283_ses-FP-PS-2019-06-20T09-32-04_behavior.nwb moved into the Selected files pane
   :width: 100%
   ```

### Checking what is inside a file

1. Click **Preview selected file** to stream a file's header and see what it
   holds, none of which is on the dandiset's page on dandiarchive.org.

   A preview is of one file, while the pipeline runs on every file you selected.
   Choose which of your selected files to read with **File to preview** beside
   the button; picking a different one leaves your selection alone, so you can
   look through a batch before analyzing it.

   ```{image} ../_static/images/dandi_dataset_preview.png
   :alt: The preview of sub-112-283_ses-FP-PS-2019-06-20T09-32-04_behavior.nwb, listing one response series of 4 channels at 1017.25 Hz over 61.6 minutes, the session's event objects and subject, a table of four channels giving their brain regions, indicator and wavelengths, and an overlay plotting the first 60 seconds of all four channels
   :width: 100%
   ```

2. Read the three parts of the preview to decide whether the file is worth
   analyzing:

   - **The response series** gives its channel count, sampling rate and duration,
     plus the event objects GuPPy can align a PSTH to and the subject the session
     came from.
   - **The channel table** gives each channel's recording site, indicator and
     excitation and emission wavelengths. Read its **Store name** column as the
     store id **Step 1: Label Stores** will offer you for that channel, and
     **Suggested label** as the GuPPy label its wavelength and site imply —
     405–420 nm is isosbestic, so `control_<site>`, and anything longer is
     `signal_<site>`. Where a file stores derived traces as their own series a
     site repeats across them, so pick the pair you want to analyze.
   - **The first 60 seconds of every channel** let you see the traces before
     committing to a full streaming run. Switch between series with **Traces from
     series** when the file holds more than one. The window starts at the
     recording's first sample, so a session that begins with the LED turning on
     opens on that transient — pan and zoom past it, and set **Eliminate first
     few seconds** when you run the pipeline.

   If the file holds no `FiberPhotometryResponseSeries`, the preview says so and
   lists whatever event objects it does hold; pick a different file.

3. Click **Hide preview** when you are done with it.

### Choosing an output directory

Choose where GuPPy should write. It creates one session folder per selected
asset, named after the asset filename minus `.nwb`.

## Labeling the streamed stores

Run **Step 1: Label Stores** as you would for local data; the page behaves
exactly as in
[Step 1 of the tutorial](../tutorials/first_analysis.md#step-1-label-your-channels).

The store names you are labeling come from inside the NWB file rather than from
filenames. A 2-D `FiberPhotometryResponseSeries` contributes one store per
column, named `<series name>_<column index>`; 1-D series and event objects keep
their own names.

Column order is not self-describing, so map each store onto its site and
wavelength using the channel table from the preview. For
`sub-112-283_ses-FP-PS-2019-06-20T09-32-04_behavior.nwb` in dandiset `000971`, a
two-site recording, that gives:

| Store | Label |
|-------|-------|
| `fiber_photometry_response_series_0` | `signal_DMS` |
| `fiber_photometry_response_series_1` | `control_DMS` |
| `fiber_photometry_response_series_2` | `signal_DLS` |
| `fiber_photometry_response_series_3` | `control_DLS` |
| `left_nose_poke_times` | `LeftNosePoke` |

## Running the rest of the pipeline

Run Steps 2–5 as in local mode. **Step 2: Read Raw Data** is the only one that
touches the network, writing each store into the run folder as it streams — the
62-minute, four-channel recording above took about 20 seconds.

Once Step 2 finishes you can re-run preprocessing, PSTH and visualization offline
and without a key, since Steps 3–5 read the local files Step 2 wrote.

## What lands on disk

Nothing from the archive is cached. Under the output directory you chose:

| Path | Contents |
|------|----------|
| `<asset name>/` | Session folder, one per selected asset |
| `<asset name>/<asset name>_output_1/` | Run folder |
| `.../storesList.csv` | Store-to-label mapping from Step 1 |
| `.../<store id>.hdf5` | One raw stream per store from Step 2, named by store id (e.g. `fiber_photometry_response_series_0.hdf5`) |

Step 3 onward writes the usual per-site files (`signal_DMS.hdf5`,
`z_score_DMS.hdf5`, and so on). See
[Output data model](../reference/outputs.md) for the full layout.

## Notes

- Select several assets to queue them as separate sessions, streamed one after
  another.
- Re-select an asset to reuse its existing session folder; Step 1 then creates an
  `_output_2` run alongside the first.
- The asset browser lists zero-byte placeholders standing in for the dandiset's
  real files, so you can navigate it without downloading anything. They live
  under your system temp directory.
- After Step 2 the sessions are ordinary local folders, so group analysis and
  **Combine Data?** apply normally.

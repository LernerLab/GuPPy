# Analyze data streamed from the DANDI Archive

GuPPy can point at a public dandiset on the
[DANDI Archive](https://dandiarchive.org) and stream the NWB assets you select
straight through the pipeline. Nothing is downloaded except GuPPy's own
outputs. Use this to reanalyze published fiber photometry data, or to check
your own dataset after uploading it. If your data is already on disk, follow
[Your First Analysis](../tutorials/first_analysis.md) instead.

## Before you start

Streaming an asset's data requires a DANDI API key, **even for a public
dandiset**. Copy your key from your account page on dandiarchive.org and export
it in the shell you launch `guppy` from:

```bash
export DANDI_API_KEY=<your key>
```

Running `dandi login` once instead stores the key in your keyring, where GuPPy
also finds it.

Searching the archive, listing a dandiset's assets and previewing a file all
read public data, so they work without a key. If the key is missing, the failure
surfaces later: **Step 1: Label Stores** stalls while the terminal that launched
`guppy` silently waits at a token prompt you cannot see.

GuPPy reads the dandiset's `draft` version.

## Choosing a dandiset and assets

In **Input Folder Selection**, set **Data Source** to `dandi`. The local file
browser is replaced by the DANDI panel, which walks three screens: a list of
dandisets, one dandiset's page, and that dandiset's files.

```{image} ../_static/images/dandi_source_selection.png
:alt: The Input Folder Selection card with the Data Source toggle switched from local to dandi, showing the DANDI source panel with its search box holding the word photometry, the filter checkbox beneath it, and a table of matching dandisets
:width: 100%
```

### Finding a dandiset

The panel opens having already searched the archive for `photometry`, so the
datasets GuPPy is for are listed without you typing anything. The search runs
against what a dataset's authors wrote about it — its title, abstract and
keywords.

Replace the term to search for something else, or type a six-digit Dandiset ID
to go straight to one dandiset. Clicking a column header sorts the table.

```{image} ../_static/images/dandi_catalog_search.png
:alt: The DANDI panel's list screen, with photometry in the search box, the "Filter dandisets for GuPPy-readable fiber photometry (slow)" checkbox below it, and a sortable table of dandisets giving each one's identifier, name, species, subject count, file count and size
:width: 100%
```

Mentioning photometry is not the same as holding photometry GuPPy can read.
**Filter dandisets for GuPPy-readable fiber photometry** settles that by reading
the listed dandisets' NWB files. A dataset can record real fiber photometry and
still be dropped, if it stores its traces as a type GuPPy does not read, so treat
a filtered list as a floor rather than the whole truth.

Reading files takes minutes where the search takes seconds, which is why it is
off by default. A progress bar tracks it and **Stop** ends it early. Verdicts are
remembered between sessions, so filtering the same datasets again is immediate.
Narrowing the search first narrows the work: filtering one search result is
seconds where filtering twenty is minutes.

Unticking the checkbox brings the unread dandisets back, with the verdicts still
in hand.

### Reading a dandiset's page

Selecting a row opens that dandiset's page: its citation details, license,
species, subject and file counts, size, keywords and abstract, with a link to
its page on dandiarchive.org. **← Back to results** returns to the list with
your search intact.

**Analyze this dandiset** opens its files.

### Selecting the NWB files

Browse the subject folders and select one or more NWB files. Navigation works
the same as local mode — click a folder to descend, Ctrl/Cmd-click to
multi-select. Only `.nwb` assets are listed.

```{image} ../_static/images/dandi_asset_browser.png
:alt: The DANDI panel's files screen for Dandiset 000971 after a scan, with a Back to dandisets link, the "Show only files GuPPy can read" checkbox ticked, a status line reading "Showing 63 of 4139 NWB asset(s) in the tree below. Scanned 4139 file(s): 63 hold fiber photometry", the surviving subject folders listed in the File Browser pane, and sub-112-283/sub-112-283_ses-FP-PS-2019-06-20T09-32-04_behavior.nwb moved into the Selected files pane
:width: 100%
```

A dandiset usually holds far more assets than recordings, and the folder names
give no sign of which is which: in `000971`, 63 of 4139 assets carry traces,
spread across 40 of its 168 subject folders.

**Scan for fiber photometry** answers that for the files. It reads every listed
file's header straight from the archive, then ticks **Show only files GuPPy can
read** so the tree holds just those. The scan takes a couple of seconds for a
typical dandiset and well under a minute for one the size of `000971`. Unticking
the checkbox brings the whole listing back.

### Reading what is inside a file

**Preview selected file** streams a file's header and reports what it holds,
none of which is on the dandiset's page on dandiarchive.org.

A preview is of one file, while the pipeline runs on every file you selected, so
**File to preview** beside the button lists the files you have selected and
chooses which of them to read. Picking a different one there leaves your
selection alone, so you can look through a batch before analyzing it.

```{image} ../_static/images/dandi_dataset_preview.png
:alt: The preview of sub-112-283_ses-FP-PS-2019-06-20T09-32-04_behavior.nwb, listing one response series of 4 channels at 1017.25 Hz over 61.6 minutes, the session's event objects and subject, a table of four channels giving their brain regions, indicator and wavelengths, and an overlay plotting the first 60 seconds of all four channels
:width: 100%
```

The preview covers the three things that decide whether a file is worth
analyzing:

- **The response series**, with its channel count, sampling rate and duration,
  plus the event objects GuPPy can align a PSTH to and the subject the session
  came from.
- **A channel table**, giving each channel's recording site, indicator and
  excitation and emission wavelengths. Its **Store name** column is the store id
  **Step 1: Label Stores** will show for that channel, and **Suggested label**
  is the GuPPy label its wavelength and site imply — 405–420 nm is isosbestic,
  so `control_<site>`, and anything longer is `signal_<site>`. A file that
  stores derived traces as their own series repeats a site across them; pick the
  pair you want to analyze.
- **The first 60 seconds of every channel**, so you can see the traces before
  committing to a full streaming run. When the file holds more than one response
  series, **Traces from series** switches between them. The window starts at the
  recording's first sample, so a session that begins with the LED turning on
  opens on that transient — which is what **Eliminate first few seconds** is
  for. Pan and zoom to look past it.

A file with no `FiberPhotometryResponseSeries` in it says so instead, and lists
whatever event objects it does hold.

**Hide preview** puts it away once you are done with it.

### Choosing an output directory

GuPPy creates one session folder per selected asset, named after the asset
filename minus `.nwb`.

## Labeling the streamed stores

Store names come from inside the NWB file, not from filenames. A 2-D
`FiberPhotometryResponseSeries` contributes one store per column, named
`<series name>_<column index>`; 1-D series and event objects keep their own
names.

Column order is not self-describing, which is what the preview's channel table
is for: it maps every store name onto the site and wavelength behind it. For
`sub-112-283_ses-FP-PS-2019-06-20T09-32-04_behavior.nwb` in dandiset `000971`,
a two-site recording, that gives:

| Store | Label |
|-------|-------|
| `fiber_photometry_response_series_0` | `signal_DMS` |
| `fiber_photometry_response_series_1` | `control_DMS` |
| `fiber_photometry_response_series_2` | `signal_DLS` |
| `fiber_photometry_response_series_3` | `control_DLS` |
| `left_nose_poke_times` | `LeftNosePoke` |

The page itself behaves exactly as in
[Step 1 of the tutorial](../tutorials/first_analysis.md#step-1-label-your-channels).

## Running the rest of the pipeline

Steps 2–5 are identical to local mode. **Step 2: Read Raw Data** is the only one
that touches the network, writing each store into the run folder as it streams —
the 62-minute, four-channel recording above took about 20 seconds.

Steps 3–5 read those local files, so once Step 2 finishes you can re-run
preprocessing, PSTH, and visualization offline and without a key.

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

- Selecting several assets queues them as separate sessions, streamed one after
  another.
- The asset browser lists zero-byte placeholders standing in for the dandiset's
  real files, so you can navigate it without downloading anything. They live
  under your system temp directory.
- Re-selecting an asset reuses its existing session folder; Step 1 then creates
  an `_output_2` run alongside the first.
- After Step 2 the sessions are ordinary local folders, so group analysis and
  **Combine Data?** apply normally.

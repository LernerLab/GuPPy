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
browser is replaced by the DANDI browser, which has its own numbered steps 1–4.
These are not the pipeline's Steps 1–5 in the sidebar.

```{image} ../_static/images/dandi_source_selection.png
:alt: The Input Folder Selection card with the Data Source toggle switched from local to dandi, showing the DANDI source panel's four steps, the collapsed "Find a fiber photometry dandiset" card, a Dandiset ID field containing 000971, and a status line reading "Dandiset 000971: 4139 NWB asset(s) found ranging 212 KB - 564.0 MB."
:width: 100%
```

### Step 1: Find a dandiset

Open **Find a fiber photometry dandiset** and press **Search DANDI**. This runs
the archive's full-text search for the photometry terms and builds a catalog of
every dandiset that matches, one row each.

```{image} ../_static/images/dandi_catalog_search.png
:alt: The Find a fiber photometry dandiset card with the Brain region filter set to Substantia nigra, a status line reading "Showing 4 of 25 dandiset(s)", and a sortable table of four dandisets with their species, subject counts, file counts, sizes and detected brain regions
:width: 100%
```

The filters below the search box narrow the catalog without going back to the
archive, so they respond immediately:

| Filter | What it matches |
|--------|-----------------|
| **Search terms** | Every word must appear somewhere in the dandiset's title, abstract, keywords or study targets |
| **Brain region** | A region named anywhere in that same text |
| **Indicator** | A sensor family named there — GCaMP, dLight, GRAB-DA, and so on |
| **Species** | The species DANDI recorded for the dandiset's subjects |
| **Approach / technique** | DANDI's own experimental-approach and measurement-technique terms |
| **Min. subjects**, **Min. NWB files** | The dandiset's totals, for finding datasets large enough to group |
| **Published versions only** | Drops draft-only dandisets, which can still change |

Each dropdown offers only the values present in the current catalog, so every
option narrows the table rather than emptying it. The **Brain regions** and
**Indicators** columns are read out of the text a submitter wrote, which is the
only place DANDI records either — so a dataset whose abstract never names its
target site shows a blank there even though the files know the site. Clearing
**Fiber photometry datasets only** sends your search terms to the archive
itself instead, which reaches all of DANDI rather than the photometry catalog.

Selecting a row shows that dandiset's full metadata underneath: its citation
details, license, subjects, keywords and abstract, with a link to its page on
dandiarchive.org.

### Reading what is inside a file

**Inspect largest NWB file** streams the header of the dandiset's biggest asset
and reports what it holds. Within a dandiset the recordings are the large files,
so the biggest one is a representative recording.

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
whatever event objects it does hold. That is worth knowing before Step 2: many
dandisets store each session's behavioral events in a small NWB file of their
own alongside the recording, and those files carry no trace for GuPPy to read.

### Step 2: Load the dandiset

**Analyze this dandiset** fills in the Dandiset ID for you and loads its assets.
You can also type a six-digit ID straight into the field and skip the catalog.
Either way the status line reports how many NWB assets were found and the range
of their sizes. A malformed ID or an unknown dandiset is reported inline.

### Step 3: Select the NWB files

Browse the subject folders and select one or more NWB files. Navigation works
the same as local mode — click a folder to descend, Ctrl/Cmd-click to
multi-select. Only `.nwb` assets are listed.

```{image} ../_static/images/dandi_asset_browser.png
:alt: The DANDI asset browser after a scan, with the Show only files with fiber photometry checkbox ticked, a status line reading "Showing 63 of 4139 NWB asset(s) in the tree below. Scanned 4139 file(s): 63 hold fiber photometry", the surviving subject folders listed in the File Browser pane, and sub-112-283/sub-112-283_ses-FP-PS-2019-06-20T09-32-04_behavior.nwb moved into the Selected files pane
:width: 100%
```

Most dandisets hold far more assets than they hold recordings. Many of them
store each session's behavioral events in a small NWB file of its own beside
the recording, so a dandiset of a few thousand assets may carry only a few
dozen photometry sessions, scattered across subject folders that give no sign
of which is which. In `000971`, 63 of 4139 assets carry traces, and they sit in
40 of its 168 subject folders.

**Scan for fiber photometry** answers that directly. It reads the header of
every listed file straight from the archive and reports which of them hold a
fiber photometry table, then ticks **Show only files with fiber photometry** so
the tree holds just those. A progress bar tracks the scan, which takes a couple
of seconds for a typical dandiset and well under a minute for one the size of
`000971`. Nothing is downloaded: the scan reads a few hundred kilobytes per
file, enough to answer the question and no more.

The scan is a report about the files rather than a rule applied to them, so
unticking the checkbox brings the whole listing back with the verdicts still in
hand. If you already know the file you want, leave the checkbox off and browse
to it.

**Preview selected file** runs the same preview as the catalog's inspect action,
against the file you selected. Use it to confirm a specific session before
running it, and to read off the store labels Step 1 will ask for.

### Step 4: Choose an output directory

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

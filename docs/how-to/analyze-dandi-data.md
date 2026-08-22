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

Browsing a dandiset does not need a key, so the asset list loads either way. If
the key is missing, the failure surfaces later: **Step 1: Label Stores** stalls
while the terminal that launched `guppy` silently waits at a token prompt you
cannot see.

GuPPy reads the dandiset's `draft` version.

## Choosing a dandiset and assets

1. In **Input Folder Selection**, set **Data Source** to `dandi`. The local file
   browser is replaced by the DANDI browser.

   ```{image} ../_static/images/dandi_source_selection.png
   :alt: The Input Folder Selection card with the Data Source toggle switched from local to dandi, showing the DANDI source panel, a Dandiset ID field containing 000971, and a status line reading "Dandiset 000971: 4139 NWB asset(s) loaded."
   :width: 100%
   ```

   The DANDI panel has its own numbered steps 1–3. These are not the pipeline's
   Steps 1–5 in the sidebar.

2. Enter a six-digit Dandiset ID. Its assets load automatically and the status
   line reports how many were found. A malformed ID or an unknown dandiset is
   reported inline.
3. Browse the subject folders and select one or more NWB files. Navigation works
   the same as local mode — click a folder to descend, Ctrl/Cmd-click to
   multi-select. Only `.nwb` assets are listed.

   ```{image} ../_static/images/dandi_asset_browser.png
   :alt: The DANDI asset browser descended into the sub-112-283 folder, listing that subject's NWB session files, with sub-112-283_ses-FP-PS-2019-06-20T09-32-04_behavior.nwb moved into the Selected files pane
   :width: 100%
   ```

4. Choose a local output directory. GuPPy creates one session folder per
   selected asset, named after the asset filename minus `.nwb`.

## Labeling the streamed stores

Store names come from inside the NWB file, not from filenames. A 2-D
`FiberPhotometryResponseSeries` contributes one store per column, named
`<series name>_<column index>`; 1-D series and event objects keep their own
names.

Column order is not self-describing, so read the file's `FiberPhotometryTable`
to map columns onto recording sites: each row's `location` names the site, and
its excitation wavelength tells you the role — 465 nm is the calcium signal,
405 nm the isosbestic control. For
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

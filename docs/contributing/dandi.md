# How the DANDI browser works

GuPPy can take its input from the [DANDI Archive](https://dandiarchive.org) instead of from disk,
streaming the NWB assets a user selects straight through the pipeline. Finding those assets is a
problem in its own right, and this page is the whole of it; [Architecture](architecture.md) only
points here. The reading half — turning a `dandi://` URI into a recording GuPPy can analyze — is
`DandiNwbRecordingExtractor`, covered in
[Adding a new acquisition format](new_recording_format.md).

## Why it is three modules

DANDI's structured metadata carries no notion of fiber photometry. There is no measurement
technique or approach for it, because those fields are derived by dandi-cli from the core NWB types
it recognizes, and the photometry types are an extension: `FiberPhotometryResponseSeries` is not
among them, so no dandiset in the archive lists it. A query cannot ask for photometry datasets, and
a dandiset's metadata cannot tell you it has any.

That splits the problem in two, and the modules follow the split. Each imports only the one before
it.

| Module | Answers |
| --- | --- |
| [`utils/dandi_search.py`](https://github.com/LernerLab/GuPPy/blob/main/src/guppy/utils/dandi_search.py) | What the archive says about dandisets and their assets |
| [`utils/dandi_filter.py`](https://github.com/LernerLab/GuPPy/blob/main/src/guppy/utils/dandi_filter.py) | Whether an asset, or a whole dandiset, holds photometry GuPPy can read |
| [`utils/dandi_preview.py`](https://github.com/LernerLab/GuPPy/blob/main/src/guppy/utils/dandi_preview.py) | What exactly one file holds |

`dandi_search` can only match free text — a dataset's title, abstract and keywords — which proposes
candidates. Everything authoritative has to be read out of the files, which is what the other two
do. A `DandisetSummary` therefore reports only what DANDI itself asserts; it does not guess at the
recording site or indicator from the prose, because those are in the file's own fiber photometry
table and `dandi_preview` reads them there.

## Deciding whether one file holds photometry

`asset_holds_photometry` answers off a single open remote file, in two stages.

The first is the `FiberPhotometry` container under `/general`, which ndx-fiber-photometry writes its
metadata table into. It is checked as a neurodata type rather than by name, since the name is
whichever one the file's author passed and differs between writers. A file without the container
holds no photometry, and that settles most files out of the prefetched window alone.

The converse does not hold, which is why there is a second stage. The container travels with the
metadata table, and a file can write that table while storing its traces as some other series type
— dandiset 000689 writes `RoiResponseSeries` — so a file that has the container is walked for the
`FiberPhotometryResponseSeries` GuPPy actually reads traces from. That walk reaches the series' own
object headers, which sit beside their data in the middle of the file and cost a few range requests
of their own.

The cached extension namespaces under `/specifications` are not a usable signal for either stage:
they record which extensions the conversion session had loaded, not which types it wrote, so a
behavior-only file written by a photometry pipeline still declares ndx-fiber-photometry.

### Why the verdict has three values

`asset_holds_photometry` returns `True`, `False` or `None`, and the third is load-bearing. A read
that failed is not evidence of absence, and recording it as one would let a dropped connection turn
a photometry dandiset into a behavior-only one.

Unreadable assets are therefore retried, but only on the way to a negative: one asset holding
photometry settles its dandiset whatever the others did, so a retry is only ever needed when
nothing has been found and some reads failed. A dandiset whose assets cannot all be read stays
unresolved rather than empty, and nothing about it is cached.

## What a read costs

Answering the question for one file touches about five kilobytes. The cost is not those bytes but
the round trips: h5py finds them by pointer-chasing through the superblock and object headers,
sixteen requests each waiting on the last.

`PrefetchedRemoteFile` is an `io.RawIOBase` that fetches a 64 KB head window and a 256 KB tail
window in parallel and serves h5py's reads out of them. HDF5 places those headers at whatever was
the end of the file when they were last written, so the two windows cover any file written in a
single session — which is what a one-shot conversion produces. A read that falls between the
windows still works, at one range request apiece.

The scan runs in a `ProcessPoolExecutor` rather than a thread pool because h5py serializes on a
global lock, which would otherwise collapse the concurrency to roughly one file at a time. That
constrains what crosses the boundary: `asset_holds_photometry` takes and returns only picklable
values, and it answers once rather than retrying, because whether a failed read is worth repeating
depends on what the dandiset's other assets said — which only the caller knows.

## Which assets to read first

Reading is what everything costs, so the orderings exist to reach an answer in as few reads as
possible.

**Within a dandiset,** `scan_order` sorts by size and then walks both ends inward, repeatedly
bisecting what is left, so any prefix of the order spans the whole size range. Which asset carries
the photometry depends on what else the dandiset carries: where the recordings are the bulk of it
they are the largest files and the behavior-only sidecars the smallest, but where photometry
accompanies electrophysiology it is the other way round. Dandiset 000689 is that second
arrangement — its photometry files are 5 MB against 19 GB of ephys, ranking 33rd of 53 by size — so
reading from either end alone would miss one of the two layouts entirely.

**Across dandisets,** `order_for_verification` reads the smallest first. Confirming a dandiset takes
one file, but ruling one out means reading every asset it has, so the largest dandisets are the
slowest to settle either way. Reading them last means the answer fills in steadily from the start
rather than stalling on one dataset of thousands of files.

That same asymmetry is why the UI separates searching from verifying and leaves verification off by
default: the list narrows quickly at first and then slows, which is a poor thing to make someone
wait through before they have seen any results at all.

## What is remembered

`PhotometryVerdictCache` persists to JSON under the user's cache directory and holds two kinds of
verdict:

- **Per asset**, keyed by the asset's immutable DANDI ID. An asset's content never changes under
  its ID, so this verdict never expires.
- **Per dandiset**, keyed by identifier *and* the asset count it was reached at. A positive is
  permanent, but a negative is only true of the assets that existed when it was taken, so a
  dandiset that grows is read again.

A repeat of a run that has already settled therefore costs no requests at all.

## The panels

Three Panel components mirror the three questions, in `frontend/`. `DandiFilePanel` is the outer
one: `input_parameters.py` builds it, and it owns one of each of the others.

| Panel | Screen |
| --- | --- |
| [`dandi_search_panel.py`](https://github.com/LernerLab/GuPPy/blob/main/src/guppy/frontend/dandi_search_panel.py) | Find a dandiset: the search box, the results table, and one dandiset's page |
| [`dandi_file_panel.py`](https://github.com/LernerLab/GuPPy/blob/main/src/guppy/frontend/dandi_file_panel.py) | Pick files from it, and hand `dandi://` URIs to the pipeline |
| [`dandi_preview_panel.py`](https://github.com/LernerLab/GuPPy/blob/main/src/guppy/frontend/dandi_preview_panel.py) | Render one `AssetPreview` |

The seams are narrow. `DandiSearchPanel` exposes `.panel`, `.open_catalog()` and an
`on_dandiset_selected` callback, which `DandiFilePanel` binds to its own `load_dandiset`; the search
panel reads nothing back. `DandiPreviewPanel` takes an `AssetPreview` through `.show()` and holds no
state of its own beyond it.

Two things about the file screen are worth knowing before changing it. Panel's `FileSelector` only
knows how to browse a filesystem, so `_build_dandiset_mirror` fabricates a local tree of zero-byte
placeholders matching the dandiset's asset layout and points the widget at that; selections are
translated back to `dandi://` URIs by `selected_uris`. And `FileSelector` caches its listing at
construction, so every dandiset change and filter toggle rebuilds the widget rather than mutating
it, which is why `attach_asset_selection_watcher` exists — watchers are re-bound to each new widget.

Both the scan and the verification run on a worker thread polled back onto the server IOLoop by
`pn.state.add_periodic_callback`, since either can take minutes and neither may block the browser.

## Tests

Each of the three modules has an offline test module and a `_live` twin, with the contract written
once in a `*_test_mixin.py` and bound to both. [Testing](testing.md#what-belongs-behind-dandi_live)
explains which assertions belong on which side of that line. The short version: the offline suites
own the behavior — `dandi_filter` runs against a local HTTP server answering real byte ranges over
real NWB files — and the live suites own only the assumptions about the archive that a substitute
cannot vouch for.

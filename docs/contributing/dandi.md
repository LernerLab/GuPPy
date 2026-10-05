# DANDI browser

GuPPy can take its input from the [DANDI Archive](https://dandiarchive.org) instead of from disk,
streaming the NWB assets a user selects straight through the pipeline. Finding those assets is a
problem in its own right, and this page is the whole of it; [Architecture](architecture.md) only
points here. The reading half — turning a `dandi://` URI into a recording GuPPy can analyze — is
`DandiNwbRecordingExtractor`, covered in
[Adding a new acquisition format](new_recording_format.md).

## Modules

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
do. A `DandisetSummary` reports what DANDI itself asserts. The recording site and indicator come
from the file's own fiber photometry table, which `dandi_preview` reads.

## The cost model

Every decision past the module split is bought with the same currency. An authoritative answer only
comes from reading a file; a dandiset is routinely thousands of assets; and the two questions cost
very differently, because confirming that a dandiset holds photometry takes one file while ruling
one out means reading every asset it has. So the work is to answer in as few reads as possible, and
to make each read as cheap as it can be.

## Reading one file cheaply

The browser reads NWB files with `h5py` rather than `pynwb`, which is the one place in GuPPy that
does — `DandiNwbRecordingExtractor` streams through `pynwb` like every other reader. Opening a file
with `pynwb` builds the whole NWB object model, which reads far more of it than these questions
need. `h5py` walks the HDF5 structure directly, so a check touches only the groups and attributes it
asks for, and the browser's questions are all answerable from structure: which typed children a
group has, what a table column holds, the shape of a dataset.

Answering the question for one file then touches about five kilobytes, but the cost is round trips
rather than bytes: h5py finds those bytes by pointer-chasing through the superblock and object
headers, sixteen requests each waiting on the last.

`PrefetchedRemoteFile` collapses that to two. It is an `io.RawIOBase` that fetches a 64 KB head
window and a 256 KB tail window in parallel and serves h5py's reads out of them. HDF5 places those
headers at whatever was the end of the file when they were last written, so the two windows cover
any file written in a single session, which is what a one-shot conversion produces. A read that
falls between the windows still works, at one range request apiece.

The scan runs in a `ProcessPoolExecutor` rather than on threads: h5py serializes on a global lock
that holds concurrent readers to roughly one file at a time. Crossing that process boundary
constrains `asset_holds_photometry`, which takes and returns only picklable values and answers once,
leaving retries to the caller — the only side that knows what the dandiset's other assets said.

## The per-file check

`asset_holds_photometry` reads in two stages so that most files settle on the cheap one.

The first stage looks for the `FiberPhotometry` container under `/general`, which
ndx-fiber-photometry writes its metadata table into. It is matched by neurodata type rather than by
name, since the name is whichever one the file's author passed and differs between writers. A file
without the container holds no photometry, and that answer comes out of the prefetched window with
no further requests at all — which is most files in most dandisets.

The second stage walks a file that does have the container for the `FiberPhotometryResponseSeries`
GuPPy reads traces from. The container travels with the metadata table, and a file can write that
table while storing its traces as some other series type — dandiset 000689 writes
`RoiResponseSeries` — so the series itself is what settles the question. That walk reaches the
series' own object headers, which sit beside their data in the middle of the file, and costs a few
range requests of its own. Paying it only for candidates is the point of the split.

### The three-valued verdict

`asset_holds_photometry` returns `True`, `False` or `None`. `None` means the file could not be read,
which is a distinct answer from an absence of photometry, and it is the one place the design spends
reads rather than saving them.

Unreadable assets are retried, but only on the way to a negative: one asset holding photometry
settles its dandiset whatever the others did, so a retry matters only when nothing has been found
and some reads failed. A dandiset whose assets cannot all be read stays unresolved, and nothing
about it is cached.

## Which files to spend reads on

Reading stops at the first asset that answers yes, so the orderings decide how soon that happens.

**Within a dandiset,** `scan_order` sorts by size and then walks both ends inward, repeatedly
bisecting what is left, so any prefix of the order spans the whole size range. That covers both
arrangements photometry appears in: where the recordings are the bulk of a dandiset they are its
largest files and the behavior-only sidecars its smallest, and where photometry accompanies
electrophysiology it is the other way round — in dandiset 000689 the photometry files are 5 MB
against 19 GB of ephys, ranking 33rd of 53 by size.

**Across dandisets,** `order_for_verification` reads the smallest first, since the largest are the
slowest to settle either way. Reading them last lets the answer fill in steadily from the start.
While the process pool scans, a thread pool fetches the listings the next dandisets will need, so
the pool is fed rather than idling between them.

That asymmetry also shapes the UI: searching and verifying are separate actions, and verification is
off until asked for, so results are on screen before any reading starts.

## Verdict cache

`PhotometryVerdictCache` persists to JSON under the user's cache directory, so a repeat of a run
that has already settled costs no requests at all. It holds two kinds of verdict:

- **Per asset**, keyed by the asset's immutable DANDI ID. An asset's content never changes under
  its ID, so this verdict never expires.
- **Per dandiset**, keyed by identifier *and* the asset count it was reached at. A positive is
  permanent, but a negative is only true of the assets that existed when it was taken, so a
  dandiset that grows is read again.

`default_verdict_cache_path()` resolves to `dandi_photometry_verdicts.json` under the platform
cache directory — `~/Library/Caches/guppy/` on macOS, `~/.cache/guppy/` on Linux,
`%LOCALAPPDATA%\LernerLab\guppy\Cache\` on Windows. Deleting it forces every verdict to be read
again, which is what to do after a change to the check that would give a file a different answer
than the one on disk:

```bash
python -c "from guppy.utils.dandi_filter import default_verdict_cache_path; default_verdict_cache_path().unlink(missing_ok=True)"
```

## Panels

`input_parameters.py` builds one `DandiFilePanel`, which owns a `DandiSearchPanel` and a
`DandiPreviewPanel` and swaps between the catalog view and the files view. The nesting runs opposite
to the names: the file screen is the outer component, and the search screen is something it opens.

The files view is a `pn.widgets.FileSelector` pointed at a fabricated filesystem. `FileSelector`
browses a directory tree and nothing else, so `_build_dandiset_mirror` writes one under the system
temp directory — a zero-byte placeholder per asset, in the dandiset's own layout — and
`selected_uris` translates the selection back to `dandi://` URIs. What that buys is that choosing
remote assets is the same gesture as choosing local ones — the same widget, the same
click-to-descend, the same multi-select — which is why the how-to can say navigation works as it
does in local mode and leave it there. What it costs is that `FileSelector` caches its listing at construction, so every
dandiset change and filter toggle rebuilds the widget rather than mutating it, and
`attach_asset_selection_watcher` exists to re-bind the selection watchers to each new one.

The scan and the verification each run on a worker thread, with progress polled back onto the
server IOLoop by `pn.state.add_periodic_callback`. Panel's own callbacks run on that loop, so
minutes of reading on it would freeze the page: the thread owns the work and the poller owns the
widgets. It is also what makes **Stop** possible: the button handler sets a flag on the shared
verification state, and the worker checks it before each dandiset.

`DandiPreviewPanel` holds no state beyond the `AssetPreview` it is handed, which is what lets both
screens drive one renderer rather than each growing its own.

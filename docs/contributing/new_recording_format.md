# Adding a new acquisition format

GuPPy currently reads TDT, Doric, Neurophotometrics (NPM), pyPhotometry, CSV, NWB, and
DANDI-streamed NWB. Adding
a new one means implementing the extractor contract and wiring it into five call sites that each own
a different concern. This page is the full recipe; [Architecture](architecture.md) only points here.

## The extractor contract

Every reader subclasses
[`BaseRecordingExtractor`](https://github.com/LernerLab/GuPPy/blob/main/src/guppy/extractors/base_recording_extractor.py)
and implements four `@abstractmethod`s:

- `discover_events_and_flags(cls) -> tuple[list[str], list[str]]`
- `read(self, *, events: list[str], outputPath: str) -> list[dict[str, Any]]`
- `save(self, *, output_dicts: list[dict[str, Any]], outputPath: str) -> None`
- `stub(self, *, folder_path: str | Path, duration_in_seconds: float = 1.0) -> None`

`discover_events_and_flags` is declared on the base class with no parameters beyond `cls` — an
inline comment there explains this is intentional, because different formats need different
discovery inputs. TDT, CSV, Doric, and NWB widen it to take only `folder_path`. NPM widens it
further, to `discover_events_and_flags(cls, folder_path, num_ch, inputParameters)`, since
demultiplexing its interleaved channels needs the channel count and, optionally, the NPM decomposition
settings chosen in the Label Stores GUI.

`read()` returns one dict per event with keys such as `store_id`, `timestamps`, `data`, and
`sampling_rate`; `save()` writes those dicts to HDF5; `stub()` copies the source folder and truncates
it to a short duration for the test suite.

### `count_samples` and `committed_samples_for_event`

Every concrete extractor also implements `count_samples(self, *, event: str) -> int`, even though it
is not one of the four `@abstractmethod`s. `orchestrate_read_raw_data` in
[`orchestration/read_raw_data.py`](https://github.com/LernerLab/GuPPy/blob/main/src/guppy/orchestration/read_raw_data.py)
calls it via `hasattr(extractor, "count_samples")`, defaulting to `0` when absent, so it is
duck-typed rather than enforced by the base class — but every extractor needs it to size the
step-2 progress bar, so treat it as a fifth required method.

`committed_samples_for_event(self, event) -> int` is an optional hook for extractors that report
progress incrementally during `read()` itself, ahead of the event finishing. Only
`DandiNwbRecordingExtractor` implements it, to convert the bytes its streaming reader has already
pulled off the wire into a samples estimate. Most new extractors will not need it.

## Start from the mock extractor

[`MockRecordingExtractor`](https://github.com/LernerLab/GuPPy/blob/main/src/guppy/testing/mock_recording_extractor.py)
implements the full contract over deterministic in-memory arrays, with no real acquisition files
involved. It is the shortest complete example of the contract and the recommended starting point to
copy from.

## Registration checklist

1. **New extractor class** in `src/guppy/extractors/<name>_recording_extractor.py`, subclassing
   `BaseRecordingExtractor`.
2. **Export it** from `src/guppy/extractors/__init__.py`.
3. **Teach format detection.** `detect_acquisition_formats()` in
   `src/guppy/extractors/detect_acquisition_formats.py` inspects the folder's files and returns the
   set of formats present; add whatever check recognizes the new format on disk.
4. **Add a routing branch** in `_build_event_to_extractor()` in
   `src/guppy/orchestration/read_raw_data.py`. This function's `if`/`elif` chain instantiates the
   right extractor for each detected format; its `else` branch raises
   `ValueError(f"Format not recognized: '{acquisition_format}'. Expected one of 'nwb', 'tdt', 'csv', 'doric', 'npm'.")` —
   add the new format name to that list too.
5. **Add the matching branch in `read_header()`, in `src/guppy/orchestration/store_labeling.py`.**
   This is a second copy of the same `if`/`elif` chain, driving step-1 event discovery instead of
   step-2 reading, with its own separate hardcoded
   `ValueError(f"Format not recognized: '{format}'. Expected one of 'nwb', 'tdt', 'csv', 'doric', 'npm'.")`.
   It is easy to update `read_raw_data.py` and forget this one, since step 1 will keep working for
   every *other* format and the omission only surfaces when a session of the new format reaches Label
   Stores. Update both files' branches and both `ValueError` messages together.
6. **Check the timestamp-correction mode.** `orchestration/preprocess.py` computes
   `mode = "tdt" if check_TDT(session_folders[i]) else "csv"`, where `check_TDT()` in
   `src/guppy/analysis/io_utils.py` just checks for a `.tsq` file in the folder. That `mode` selects
   the step-3 branch in `timestampCorrection()` in `src/guppy/analysis/timestamp_correction.py`, which
   only knows `"tdt"` and `"csv"`. Every non-TDT format currently falls into the `"csv"` branch, which
   assumes one timestamp per sample. TDT is different: each read returns discrete acquisition blocks,
   with one timestamp per block rather than per sample, so it needs its own timestamp-expansion logic.
   A new format that is also block-structured (blocks of samples sharing one timestamp) would silently
   get the wrong timebase under the `"csv"` branch and would need a third branch here. Nothing else in
   the codebase flags this, so check whether the new format's `read()` output is block-structured
   before assuming the `"csv"` branch is correct for it.
7. **Frontend component, only if needed.** A dedicated store-labeling instructions widget is only
   necessary when the format requires user input *before* its events can even be enumerated. NPM is
   the precedent: `StoreLabelingInstructionsNPM` in `src/guppy/frontend/store_labeling_instructions.py`
   asks for the timestamp column and unit and whether to split multi-value TTLs, because
   `discover_events_and_flags` cannot name NPM's derived streams without those answers. Most new
   formats can enumerate their events from the files alone and need no such component.
8. **Stubbed sample session.** Add a truncated sample session under `stubbed_testing_data/<format>/`,
   and a README section for it in `stubbed_testing_data/README.md` matching the shape of the existing
   per-session entries there (a heading naming the session path, a short description of what it is
   used to test, and a **Stores** list naming each store and what it holds). If the format can be
   truncated from a full recording rather than hand-authored, also register it in
   `src/guppy/testing/scripts/create_stubbed_testing_data.py`, whose `_sessions()` function returns a
   list of `(extractor_instance, stub_duration_in_seconds, destination_path)` tuples.

## Testing requirements

### Unit: the extractor contract

`tests/unit/extractors/recording_extractor_test_mixin.py` defines `RecordingExtractorTestMixin`, a
shared contract test suite every extractor's test class should inherit. A subclass must set these
class attributes:

- `extractor_class` — the concrete extractor class under test.
- `folder_path` — folder passed to `discover_events_and_flags` and the constructor.
- `expected_events` — at least one event name known to be discoverable there.
- `discover_kwargs` — extra keyword arguments for `discover_events_and_flags()` beyond `folder_path`
  (`{}` for TDT/Doric/CSV; NPM needs `{"num_ch": N}`).
- `extractor_instance` — an initialized instance of the extractor under test.
- `stub_extractor_kwargs` — extra constructor keyword arguments used when re-instantiating from a
  stubbed folder (`{}` unless the constructor needs more than `folder_path`, e.g. Doric's
  `event_name_to_event_type`).
- `control_event`, `signal_event` — event names for the control and signal channels.
- `ttl_event` — event name for the TTL/event channel, or `None` for sessions with no TTL channel
  (TTL-related tests become no-ops).
- `stub_ttl_test_duration_in_seconds` — duration passed to `stub()` for the TTL-pruning test; choose a
  value that captures some but not all TTL events.

And provide these fixtures, each returning the array as it should appear in the saved HDF5 file:
`expected_control_timestamps`, `expected_control_data`, `expected_signal_timestamps`,
`expected_signal_data`, and — only when `ttl_event` is not `None` — `expected_ttl_timestamps`.

In exchange, the mixin runs the whole contract for free: that `discover_events_and_flags` and
`count_samples` return the right shapes and values, that `read()` and `save()` round-trip data to
HDF5 correctly, and that `stub()` produces a truncated-but-otherwise-identical, idempotent copy.

### Integration: representative sessions

`REPRESENTATIVE_SESSIONS` lives in `tests/integration/integration_helpers.py`, not `conftest.py`.
That module's docstring explains why: a couple of integration test modules
(`test_integration_step2.py`, `test_integration_dandi.py`) need these symbols at import time, and
importing from `integration_helpers` directly avoids relying on the ambiguous bare `conftest` module
name that a plain `from conftest import ...` would require.

Adding a format means:

1. One entry in `REPRESENTATIVE_SESSIONS`, giving its `session_subdir`, `store_id_to_store_label`,
   and NPM-related keys (`None` for non-NPM formats).
2. Five new fixtures in `tests/integration/conftest.py` — `step1_output_<format>` through
   `step5_output_<format>` — following the existing chained pattern, where `step1_output_<format>`
   builds a `pipeline_state` from `tmp_path_factory` and the modality name, and each later step's
   fixture takes the previous step's fixture as its only argument and runs the next step on it (for
   example `step2_output_tdt(step1_output_tdt)` calls `_run_step2(pipeline_state=step1_output_tdt)`).
3. The new `step{1..5}_output_<format>` fixture names added to the `@pytest.mark.parametrize` lists in
   `tests/integration/test_integration_step1.py` through `test_integration_step5.py`.

See [Output data model](../reference/outputs.md) for the on-disk shape conventions (dataset shapes,
scalar vs. array `sampling_rate`, filename patterns) a new extractor's output should follow.

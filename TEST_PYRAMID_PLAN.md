# Test Pyramid Rebalancing Plan

## 1. Current State Audit

### Test counts by tier

| Tier         | Files | Tests | Notes                                      |
|--------------|-------|-------|--------------------------------------------|
| Unit         | 35    | ~396  | Good coverage of analysis, frontend, extractors |
| Integration  | 12    | ~18   | End-to-end step smoke tests; currently over-parametrized |
| Consistency  | 10    | ~10   | `@pytest.mark.full_data`; slow, require full dataset |
| UI           | 5     | ~34   | `@pytest.mark.ui`; browser-based           |

### Problems with the current shape

Problems are listed tier by tier, bottom-up, which is the same order as the
proposed changes in section 3.

**Problem 1 — Extractor unit tests do not cover all format variants.**
Eight of the fourteen format variants used in integration tests have no
corresponding extractor unit test class. This is the root blocker for trimming
integration test parametrization: a format variant cannot be removed from
integration until it is covered at the unit/extractor level.

Current extractor unit test classes (via `RecordingExtractorTestMixin`):

| Format | Session | Covered |
|--------|---------|---------|
| CSV    | `sample_data_csv_1` | ✓ `TestCsvRecordingExtractor` |
| Doric  | `sample_doric_1` | ✓ `TestDoricRecordingExtractor` |
| Doric  | `sample_doric_2` | ✗ missing |
| Doric  | `sample_doric_3` | ✓ `TestDoricRecordingExtractorV6` |
| Doric  | `sample_doric_4` | ✗ missing (no TTL channel) |
| Doric  | `sample_doric_5` | ✗ missing (no TTL channel) |
| TDT    | `Photo_63_207-181030-103332` | ✓ `TestTdtRecordingExtractor` |
| TDT    | `Photo_048_392-200728-121222` | ✗ missing |
| TDT    | `Photometry-161823` | ✗ missing (split-event) |
| NPM    | `sampleData_NPM_1` | ✓ `TestNpmRecordingExtractor` |
| NPM    | `sampleData_NPM_2` | ✗ missing (no TTL channel) |
| NPM    | `sampleData_NPM_3` | ✓ `TestNpmRecordingExtractorSession3` |
| NPM    | `sampleData_NPM_4` | ✗ missing |
| NPM    | `sampleData_NPM_5` | ✗ missing |

Sessions marked as "no TTL channel" (`sample_doric_4`, `sample_doric_5`,
`sampleData_NPM_2`) cannot use the current `RecordingExtractorTestMixin` as-is
because it requires a `ttl_event` attribute. The mixin needs a small extension
to make TTL tests conditional (see section 3A).

**Problem 2 — The orchestration layer has almost no unit test coverage.**
Only `save_parameters.py` (5 tests) and a few helpers in `storenames.py`
(14 tests for `_save`, `_fetchValues`, `show_dir`, `make_dir`) have unit tests.
The remaining five orchestration modules have zero unit coverage:

| Module              | Unit tests | Integration tests |
|---------------------|-----------|-------------------|
| `home.py`           | 0         | Indirect (all integration tests) |
| `read_raw_data.py`  | 0         | `test_integration_step3.py` |
| `preprocess.py`     | 0         | `test_integration_step4.py` |
| `psth.py`           | 0         | `test_integration_step5.py` |
| `transients.py`     | 0         | `test_integration_step5.py` |

**Problem 3 — Integration execution is inefficient for its intended role.**
Integration tests should be treated as end-to-end pipeline smoke tests in this
codebase. The current issue is not that they are end-to-end, but that they are
implemented with repeated inline setup and excessive parametrization.

**Problem 4 — End-to-end integration scope is not explicit in structure.**
Step 3/4/5 integration tests currently read like step-isolated integration, but
their actual behavior is end-to-end execution (running prior steps as setup).
The suite should make that intent explicit and optimize around it.

**Problem 5 — Integration tests are over-parametrized across data formats.**
Steps 3, 4, and 5 each parametrize over all 14 format variants (CSV, 5 Doric,
3 TDT, 5 NPM). Format-specific behaviour is an extractor responsibility that
should be tested at the unit/extractor level. Fourteen copies of the same full
pipeline run makes the integration suite slow without adding proportional value.

---

## 2. Target Pyramid

```
                  /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
                 /   E2E / Consistency       \   ~10–20 tests
                /   (full pipeline, real data) \
               /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
              /        Integration              \   ~18–30 tests
             / (end-to-end step smoke, 1–3 formats)\
            /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
           /                Unit                  \   450–500 tests
          /   (one function / class in isolation)   \
         /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
```

---

## 3. Proposed Changes

Changes are ordered tier by tier, bottom-up, mirroring the problem list.

### 3A — Expand extractor unit test coverage (addresses Problem 1) ✅ COMPLETED

**This must be completed before any integration test format variant is removed.**

#### Extend `RecordingExtractorTestMixin` for TTL-free sessions

Three sessions (`sample_doric_4`, `sample_doric_5`, `sampleData_NPM_2`) have no
TTL/event channel. Add an optional `ttl_event = None` class attribute to the
mixin. All TTL-related test methods (roundtrip TTL timestamps, stub TTL pruning)
become no-ops when `ttl_event is None`.

#### Add eight new extractor test classes

Each new class sets the class-level attributes required by the mixin and
implements the five `expected_*` fixtures. All use `stubbed_testing_data/`
sessions already present in the repository.

**`test_doric_recording_extractor.py` additions:**

| Class | Session | TTL? | Notes |
|-------|---------|------|-------|
| `TestDoricRecordingExtractorSample2` | `sample_doric_2` | yes (`DI/O-1`) | demodulated lock-in |
| `TestDoricRecordingExtractorSample4` | `sample_doric_4` | no | lock-in, no TTL channel |
| `TestDoricRecordingExtractorSample5` | `sample_doric_5` | no | lock-in, no TTL channel |

**`test_tdt_recording_extractor.py` additions:**

| Class | Session | TTL? | Notes |
|-------|---------|------|-------|
| `TestTdtRecordingExtractorSample2` | `Photo_048_392-200728-121222` | yes (`PrtN`) | second TDT variant |
| `TestTdtRecordingExtractorSplitEvent` | `Photometry-161823` | yes (`PAB/`) | split-event; `PAB/` maps to three sub-events |

**`test_npm_recording_extractor.py` additions:**

| Class | Session | TTL? | Notes |
|-------|---------|------|-------|
| `TestNpmRecordingExtractorSession2` | `sampleData_NPM_2` | no | multi-file, no event channel |
| `TestNpmRecordingExtractorSession4` | `sampleData_NPM_4` | yes (`eventTrue`) | boolean event TTL |
| `TestNpmRecordingExtractorSession5` | `sampleData_NPM_5` | yes (`event0`) | single event column |

For the split-event TDT case (`Photometry-161823`), the `expected_events` list
should include the post-split names (`PAB_0`, `PAB_16`, `PAB_2064`) and
`ttl_event` should be one of them (e.g. `PAB_0`). The discover and roundtrip
tests will confirm splitting works correctly at the extractor level.

---

### 3B — New unit tests for the orchestration layer (addresses Problem 2) ✅ COMPLETED

All files go in `tests/unit/orchestration/`.

#### `test_home.py` (new, ~5 tests)

Test that `build_homepage()` correctly exposes the testing hooks:
- Asserts `template._hooks` contains `"onclickProcess"` and
  `"getInputParameters"`, and that both are callable.
- Asserts `template._widgets` contains `"files_1"` and that it is a Panel
  widget with a `.value` attribute.
- Asserts that calling `template._hooks["onclickProcess"]()` with no folders
  selected does not raise.

No data files required; uses the `GUPPY_BASE_DIR` env var set by conftest.py.

#### `test_read_raw_data.py` (new, ~6–8 tests)

Unit-test the routing logic in `_build_event_to_extractor` using mock extractor
classes that return known event lists from `discover_events_and_flags`. Inject
mocks via `monkeypatch.setattr` on `detect_acquisition_formats` and the
extractor constructors — no real data files needed:
- CSV-only folder: all events route to `CsvRecordingExtractor`.
- TDT-only folder: all events route to `TdtRecordingExtractor`.
- Mixed TDT + CSV: events partition to the correct extractor.
- Unknown format string raises `ValueError`.
- First-seen extractor wins for duplicate event names (no clobbering).

#### `test_preprocess.py` (new, ~8–10 tests)

Unit-test the orchestration functions in `preprocess.py` using programmatically
constructed HDF5 inputs built with `h5py` and NumPy inside `tmp_path` fixtures.
No real data or preceding pipeline steps required:
- `execute_timestamp_correction`: given minimal raw-signal HDF5 files, assert
  that `timeCorrection_<region>.hdf5` is created with a `timestampNew` dataset.
- `execute_zscore`: given pre-written raw-signal HDF5s, assert the z-score HDF5
  is created with the correct key.
- Parameter routing: confirm `zscore_method="baseline z-score"` invokes the
  baseline-window code path rather than the standard path.

#### `test_psth.py` (new, ~6–8 tests)

Unit-test the orchestration-level PSTH functions using programmatically
constructed HDF5 inputs in `tmp_path`:
- `execute_compute_psth`: given a synthetic timestamp-aligned HDF5, assert the
  PSTH `.h5` output is created with `timestamps` and `mean` columns.
- `execute_compute_psth_peak_and_area`: given a synthetic PSTH `.h5`, assert
  peak/AUC `.h5` and `.csv` outputs are created.
- `execute_cross_correlation` (with `compute_corr=True`): assert the output CSV
  is created.

---

### 3C — Restructure and slim end-to-end integration tests (addresses Problems 3, 4, and 5)

#### Add `tests/integration/conftest.py` with session-scoped fixtures

Treat the integration suite as explicit end-to-end smoke coverage and replace
inline step2→...→stepN call chains with session-scoped fixtures using
`tmp_path_factory`. Each fixture runs the pipeline through step N−1 exactly
once and yields the output directory path. All tests for step N share that
cached output:

- `step2_output_csv`, `step2_output_tdt`, `step2_output_npm` — run step2 once
  per representative format, yield output directory.
- `step2_output_doric` — run step2 once for a representative Doric session,
  yield output directory.
- `step3_output_csv`, `step3_output_tdt`, `step3_output_npm` — take the
  corresponding `step2_output_*` fixture, run step3, yield updated output directory.
- `step3_output_doric` — take `step2_output_doric`, run step3, yield updated
  output directory.
- `step4_output_csv`, `step4_output_tdt`, `step4_output_npm` — same pattern
  through step4.
- `step4_output_doric` — same pattern through step4.

`test_integration_step3.py`, `test_integration_step4.py`, and
`test_integration_step5.py` remain end-to-end by design. They should validate
that each stage successfully extends a pipeline state produced by prior steps,
rather than attempt seam-level isolation.

Note: because session-scoped fixtures cannot use the function-scoped `tmp_path`,
they use `tmp_path_factory.mktemp(...)` to create isolated directories.

#### Reduce format parametrization in step3/step4/step5

Once every removed variant is covered by an extractor unit test class (from 3A),
trim integration parametrization to four representative formats:
- CSV: `sample_data_csv_1` (simplest, fewest dependencies)
- TDT: `Photo_63_207-181030-103332` (TDT-clean; also used in consistency tests)
- NPM: `sampleData_NPM_1`
- Doric: `sample_doric_1`

This reduces each of `test_integration_step3`, `test_integration_step4`, and
`test_integration_step5` from 14 parametrized cases to 4.

---

### 3D — No changes to consistency / UI tests

The consistency tests (`@pytest.mark.full_data`) and UI tests (`@pytest.mark.ui`)
appropriately occupy the E2E tier. No restructuring is needed there.

---

## 4. Summary of Work Items

Listed in the same tier-by-tier order as the problems and changes above.

| Priority | Work item | Files affected |
|----------|-----------|----------------|
| 1 | ✅ Extend `RecordingExtractorTestMixin` to make `ttl_event` optional | `recording_extractor_test_mixin.py` (edit) |
| 2 | ✅ Add 8 missing extractor test classes (Doric ×3, TDT ×2, NPM ×3) | `test_doric/tdt/npm_recording_extractor.py` (edit each) |
| 3 | ✅ Add `test_home.py` orchestration unit tests | `tests/unit/orchestration/test_home.py` (new) |
| 4 | ✅ Add `test_read_raw_data.py` orchestration unit tests | `tests/unit/orchestration/test_read_raw_data.py` (new) |
| 5 | ✅ Add `test_preprocess.py` orchestration unit tests | `tests/unit/orchestration/test_preprocess.py` (new) |
| 6 | ✅ Add `test_psth.py` orchestration unit tests | `tests/unit/orchestration/test_psth.py` (new) |
| 7 | Add `tests/integration/conftest.py` with session-scoped `tmp_path_factory` fixtures | `tests/integration/conftest.py` (new) |
| 8 | Slim step3/step4/step5 integration parametrization to CSV, TDT-clean, NPM-1, Doric-1 | `test_integration_step3/4/5.py` (edit) |

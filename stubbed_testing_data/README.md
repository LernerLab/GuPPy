# Testing Data

Real fiber photometry recordings used by the GuPPy test suite. Each session has been truncated from the corresponding full recording in `testing_data/` to reduce file size while retaining enough data to exercise the relevant code paths — for sessions with TTL events, the duration was chosen to capture at least 5 complete pulses. To regenerate from the full recordings, run `src/guppy/testing/scripts/create_stubbed_testing_data.py`.

The `tdt/ME112-ME113-260420-114630` session is an exception: its source recording is kept local (not uploaded to the shared `testing_data/` Google Drive), so it's stubbed via its own one-off script `src/guppy/testing/scripts/stub_me112_me113_session.py`. `npm/sampleData_NPM_6` is another: its source is third-party sample data, stubbed via `src/guppy/testing/scripts/stub_phat_npm_session.py`.

---

# TDT

Tucker-Davis Technologies (TDT) Synapse tank format. Each session is a folder of binary files (`.tev`, `.tsq`, `.Tbk`, `.Tdx`, `.tin`, `.tnt`).

## `tdt/Photo_63_207-181030-103332`

Standard clean recording used as the baseline TDT case. Also used for cross-correlation testing with a dual-region storename map (DMS + DLS). Duration: 157.5 s (5th port-entry event at ~157.4 s).

**Stores:**
- `Dv1A`: isosbestic control recording from the dorsomedial striatum
- `Dv2A`: calcium signal recording from the dorsomedial striatum
- `Dv3B`: isosbestic control recording from the dorsolateral striatum
- `Dv4B`: calcium signal recording from the dorsolateral striatum
- `Fi1i`: 2xN array of modulated signals: dms and dls
- `Fi1r`: 2xN array of modulated commanded voltages: dms and dls
- `LNRW`: TTL event for each rewarded nose poke
- `LNnR`: TTL event for each unrewarded nose poke
- `PrtN`: TTL event for each unrewarded port entry
- `PrtR`: TTL event for each rewarded port entry
- `RNPS`: TTL event of unknown meaning

## `tdt/Photometry-161823`

Recording whose TTL store (`PAB/`) exhibits non-contiguous event blocks that the extractor splits into sub-events. Used to test TDT split-TTL handling. Duration: 215.9 s (5th TTL event at ~215.8 s).

`PAB/` splits into sub-events `PAB_0`, `PAB_16`, `PAB_2064` during extraction.

**Stores:**
- `405R`: 405 nm excitation channel (isosbestic reference)
- `490R`: 490 nm excitation channel (calcium signal)
- `DelF`: Online dF/F signal
- `Fi1i`: 2xN array of modulated signals: dms and dls
- `Fi1r`: Modulated commanded voltage
- `PAB/`: TTL event store (splits into `PAB_0`, `PAB_16`, `PAB_2064` sub-events)
    - `PAB_0`: TTL event of unknown meaning
    - `PAB_16`: TTL event of unknown meaning
    - `PAB_2064`: TTL event of unknown meaning
- `Tick`: Regular Video synchronization TTL (every 1 s)
- `Vid1`: Video timestamp capture TTL

## `tdt/ME112-ME113-260420-114630`

Recording whose TTL store (`Widt`) carries *float-valued* event codes (0.1, 0.2, 0.4, 0.8, 10.0 — stimulus widths) that `_event_needs_splitting` detects and splits into sub-events. Used to test that the TDT split mechanism preserves unique float codes (suffix format `0p1`, `0p2`, …) rather than collapsing them via `int()`. Duration: 25.0 s (Widt codes cycle every 4 s, so 25 s captures the full cycle twice and every unique code at least once).

`Widt` splits into sub-events `Widt0p1`, `Widt0p2`, `Widt0p4`, `Widt0p8`, `Widt10` during extraction.

**Stores:**
- `415A`: 415 nm excitation channel (isosbestic control)
- `465A`: 465 nm excitation channel (calcium signal)
- `Widt`: TTL event store with float-valued codes (splits into `Widt0p1`, `Widt0p2`, `Widt0p4`, `Widt0p8`, `Widt10` sub-events)
- Additional channels present but unused by tests: `415B`–`D`, `465B`–`D`, `560A`–`D` (additional excitation/region channels), `Cam1`, `Cn1/`, `Fi1i`, `Fi1r`, `Fi2i`, `Fi2r`, `PtC0`, `PtC1`, `Tick`, `Wid\`

## `tdt/Photo_048_392-200728-121222`

Recording with artifactual transients in the raw signal. Primary purpose is testing artifact removal (the artifact-removal pipeline must detect and either concatenate around or replace with NaN). Duration: 184.3 s (5th port-entry event at ~184.2 s).

**Stores:**
- `Dv1A`: isosbestic control recording from the dorsomedial striatum
- `Dv2A`: calcium signal recording from the dorsomedial striatum
- `Dv3B`: isosbestic control recording from the dorsolateral striatum
- `Dv4B`: calcium signal recording from the dorsolateral striatum
- `Fi1i`: 2xN array of modulated signals: dms and dls
- `Fi1r`: 2xN array of modulated commanded voltages: dms and dls
- `LNRW`: TTL event for each rewarded nose poke
- `LNnR`: TTL event for each unrewarded nose poke
- `PrtN`: TTL event for each unrewarded port entry
- `PrtR`: TTL event for each rewarded port entry

---

# Doric

Doric Lenses photometry recordings. Three file format variants appear across the sessions: V1 (`.doric` HDF5 with flat channel keys), V6 (`.doric` HDF5 with hierarchical path keys), and CSV export (`.csv`).

## `doric/sample_doric_1`

Standard Doric V1 recording with a TTL channel. Baseline case for Doric format testing. Duration: 115.2 s (5th TTL pulse end at ~115.1 s).

**File:** `D2-EPConsole_0039.doric` (Doric V1)

**Stores:**
- `AIn-1 - Raw`: Raw fiber photometry signal from analog input 1 (could be signal or control)
- `AIn-2 - Raw`: Raw fiber photometry signal from analog input 2 (could be signal or control)
- `DI--O-1`: digital input/output TTL channel

## `doric/sample_doric_2`

Doric CSV export format: a `.csv` file that follows Doric channel naming conventions rather than the GuPPy generic CSV layout. Used to test CSV export format detection and parsing. Duration: 78.0 s (5th TTL pulse end at ~77.9 s).

**File:** `12282020-cfc-pppda7_0000.csv` (Doric CSV export)

**Stores:**
- `AIn-1 - Dem (ref)`: analog input 1 demodulated reference channel (isosbestic control)
- `AIn-1 - Dem (da)`: analog input 1 demodulated dopamine channel
- `Raw`: raw analog input channel
- `DI/O-1`: digital input/output TTL channel
- `AOut-1`: analog output channel 1
- `AOut-2`: analog output channel 2
- `Unnamed: 7`: Empty column (not sure why)

## `doric/sample_doric_3`

Doric V6 recording. The V6 format stores channels under hierarchical HDF5 paths (e.g., `CAM1_EXC1/ROI01`) rather than the flat keys used in V1. Used to test V6 path parsing. Duration: 16.0 s (TTL events occur at ~0.1 s intervals so 16 s captures many pulses).

**File:** `BFPD_Acq_0000.doric` (Doric V6)

**Stores:**
- `CAM1_EXC1/ROI01`: camera 1, excitation 1, region of interest 1
- `CAM1_EXC1/ROI02`: camera 1, excitation 1, region of interest 2
- `CAM1_EXC1/ROI03`: camera 1, excitation 1, region of interest 3
- `CAM1_EXC2/ROI01`: camera 1, excitation 2, region of interest 1
- `CAM1_EXC2/ROI02`: camera 1, excitation 2, region of interest 2
- `CAM1_EXC2/ROI03`: camera 1, excitation 2, region of interest 3
- `DigitalIO/CAM1`: digital I/O channel synchronized to camera 1 (TTL)
- `DigitalIO/EXC1`: digital I/O channel synchronized to excitation 1
- `DigitalIO/EXC2`: digital I/O channel synchronized to excitation 2

## `doric/sample_doric_4`

Doric V1 lock-in amplifier recording with no TTL events. One of two independent examples of the no-TTL lock-in format. Duration: 16.0 s.

**File:** `LiCl_0001.doric` (Doric V1 LockIn)

**Stores:**
- `Series0001/AIN01xAOUT01-LockIn`: lock-in amplifier output for analog input 01 demodulated by analog output 01
- `Series0001/AIN01xAOUT02-LockIn`: lock-in amplifier output for analog input 01 demodulated by analog output 02
- `Series0001/AIN03xAOUT01-LockIn`: lock-in amplifier output for analog input 03 demodulated by analog output 01
- `Series0001/AIN03xAOUT02-LockIn`: lock-in amplifier output for analog input 03 demodulated by analog output 02
- `AnalogIn/AIN01`: raw analog input channel 01
- `AnalogIn/AIN03`: raw analog input channel 03
- `AnalogOut/AOUT01`: analog output channel 01
- `AnalogOut/AOUT02`: analog output channel 02

## `doric/sample_doric_5`

Second independent example of the Doric V1 lock-in amplifier format with no TTL events. Duration: 16.0 s.

**File:** `saline_0001.doric` (Doric V1 LockIn)

**Stores:**
- `Series0001/AIN01xAOUT01-LockIn`: lock-in amplifier output for analog input 01 demodulated by analog output 01
- `Series0001/AIN01xAOUT02-LockIn`: lock-in amplifier output for analog input 01 demodulated by analog output 02
- `Series0001/AIN03xAOUT01-LockIn`: lock-in amplifier output for analog input 03 demodulated by analog output 01
- `Series0001/AIN03xAOUT02-LockIn`: lock-in amplifier output for analog input 03 demodulated by analog output 02
- `AnalogIn/AIN01`: raw analog input channel 01
- `AnalogIn/AIN03`: raw analog input channel 03
- `AnalogOut/AOUT01`: analog output channel 01
- `AnalogOut/AOUT02`: analog output channel 02

---

# CSV

GuPPy generic CSV format: one two-column file per channel (timestamps, data values).

## `csv/sample_data_csv_1`

Standard generic CSV recording. Baseline case for Steps 2–5 integration tests and consistency tests (z-score methods, no-isosbestic control, dFF). Duration: 411.0 s (5th TTL event at ~410.9 s).

**Files:** `Sample_Control_Channel.csv`, `Sample_Signal_Channel.csv`, `Sample_TTL.csv`

**Stores:**
- `Sample_Control_Channel`: isosbestic control channel
- `Sample_Signal_Channel`: calcium signal channel
- `Sample_TTL`: TTL event channel

## `csv/sample_data_csv_injection_1`

Synthetic session (not truncated from a real recording) modelling a bolus-injection experiment, used by the baseline-epoch control-fitting tests (`tests/integration/test_baseline_epoch_fit.py`) and the tonic/basal analysis tests (`tests/integration/test_tonic.py`). The 465 nm signal steps up at the injection while the 405 nm isosbestic control does not, so a full-trace control fit is corrupted by the step but a fit estimated on the pre-injection window is not. Sampling rate 100 Hz, duration 180.0 s. Regenerate with `_write_injection_csv_session` in `src/guppy/testing/scripts/create_stubbed_testing_data.py` (deterministic seed).

The signal holds three equal 60 s phases, giving a tonic analysis three plateaus to average over:

| Phase | Window | Drug effect on the signal |
|---|---|---|
| baseline | 0–60 s | none; control and signal share the true linear relationship |
| drug on board | 60–120 s | +40, held |
| washout | 120–180 s | clears exponentially (τ = 6 s) to a 25% residual, so ~+10 from t≈140 s |

The rise is instantaneous while the fall is exponential: a bolus reaches the tissue fast relative to this timebase and clears more slowly, and it is the discontinuity at t=60 s that breaks a full-trace control fit. Clearance is deliberately incomplete, so the three phases sit at three separated levels rather than the washout landing back on the baseline.

**Files:** `Sample_Control_Channel.csv`, `Sample_Signal_Channel.csv`, `Sample_TTL.csv`

**Stores:**
- `Sample_Control_Channel`: isosbestic control channel (bleaching + motion; carries no drug effect)
- `Sample_Signal_Channel`: calcium signal channel (linear in control, plus the drug effect above)
- `Sample_TTL`: TTL event channel (8 pulses on a 20 s grid from 20 s; the 60 s pulse marks the injection and the 120 s pulse the washout onset)

## `csv/sample_data_csv_covariate_1`

Synthetic session (not truncated from a real recording) carrying two hand-scored behavioral covariates alongside the photometry. Used by the covariate correlation tests (`tests/integration/test_covariate_correlations.py`) and by the documentation screenshot script (`docs/take_screenshots.py`), which runs Steps 1-4 on it to photograph the Covariates tab. Sampling rate 100 Hz, duration 600.0 s, which gives 12 bins at the 50 s bin width both consumers use. Regenerate with `_write_covariate_csv_session` in `src/guppy/testing/scripts/create_stubbed_testing_data.py` (deterministic seed).

The two covariates are built the same way and differ only in whether they drive the signal:

| Covariate | Drives the 465 nm signal? | Pearson r against `mean_zscore` |
|---|---|---|
| `akinesia` | yes, at 8 signal units per point | 0.84 |
| `grooming` | no; generated independently | 0.33 |

`grooming` is a **null pseudo-covariate and is meant to stay that way** — it is not a defect to be corrected. Two slowly varying, autocorrelated series still land at a moderate correlation over a dozen bins even when nothing connects them, and showing that is the point: it sets the expectation the covariate how-to argues for in prose.

The signal also carries a slow nuisance component that no covariate explains. Without it, averaging over 50 s bins would annihilate the white noise and leave `akinesia` correlating at ~1.0, which would misrepresent what the analysis produces. Its amplitude is set so the nuisance contributes ~0.62 times the effect's per-bin spread. Transients are added at a constant 0.4 Hz, so per-bin transient counts vary without tracking either covariate.

**Files:** `Sample_Control_Channel.csv`, `Sample_Signal_Channel.csv`, `Sample_TTL.csv`, `akinesia.csv`, `grooming.csv`

**Stores:**
- `Sample_Control_Channel`: isosbestic control channel (bleaching + motion; carries no covariate effect, so the control fit cannot regress the effect out of the signal)
- `Sample_Signal_Channel`: calcium signal channel (linear in control, plus the akinesia effect, the nuisance component and the transients)
- `Sample_TTL`: TTL event channel (19 pulses on a 30 s grid from 30 s)
- `akinesia`: behavioral covariate, 24 scores every 25 s spanning 0.4-4.4; drives the signal
- `grooming`: behavioral covariate, 24 scores every 25 s spanning 0.3-3.6; drives nothing

---

# NPM (Neurophotometrics)

Neurophotometrics fiber photometry recordings. Two format generations are present: v2 (files contain a `LedState` header column) and legacy (no `LedState` header, rows interleaved by LED state).

`NpmRecordingExtractor` demultiplexes the raw files into per-channel and per-event streams **in memory** and writes nothing back into the session folder. Store names below (`file0_chev*`, `file0_chod*`, `event*`) are those in-memory stream names, not files on disk. Older GuPPy versions did write these as intermediate CSVs into the session folder; any such leftovers in a session folder are stale and will be picked up as `csv` data in preference to the live NPM demultiplexing, so delete them.

A session whose photometry file offers more than one timestamp column needs both `Timestamp column` and `Time unit` set in the Label Stores NPM configuration. The event file carries a single unnamed column and is always on the acquisition's **absolute** clock, so it cannot follow the column choice — picking the other column puts events and photometry on different timelines. The per-session entries below record the settings each stub needs.

## `npm/sampleData_NPM_1`

NPM v2 recording with a separate stimuli event file. The stimuli file contains multiple named event types; with `split_events=True`, each type becomes its own event store. Duration: 161.5 s (5th stimuli event at ~161.4 s after the photometry starts).

**Required settings:** `npm_timestamp_column_name="ComputerTimestamp"`, `npm_time_unit="milliseconds"`.
The photometry file carries two timestamp columns — `SystemTimestamp` (seconds, raw span
`[1891.3, 2052.8]`) and `ComputerTimestamp` (milliseconds, raw span `[4.98849e7, 5.00464e7]`) —
which are the same clock at 1000× different scale. The stimuli file is on the `ComputerTimestamp`
clock. Default resolution picks the *first* timestamp column, `SystemTimestamp`, which leaves the
events ~48,000,000 s away from the photometry.

**Files:** `bl72bl82_12feb2024_fp.csv` (photometry, v2), `bl72bl82_12feb2024_stimuli.csv` (events)

**Stores (after discover + split):**
- `file0_chev1`: Isosbestic control channel
- `file0_chod1`: Calcium Signal Channel
- `eventAfVn`: TTL of unknown meaning
- `eventAfVu`: TTL of unknown meaning
- `eventAmVf`: TTL of unknown meaning
- `eventpinknoise`: TTL event for pink noise stimulus delivery
- `eventwhitenoise`: TTL event for white noise stimulus delivery

## `npm/sampleData_NPM_2`

NPM v2 recording split across two source files (one per excitation wavelength), with no TTL events. Used to test multi-file v2 discovery and cross-file channel alignment. Duration: 16.0 s.

**Files:** `FiberData415.csv` (415 nm excitation, v2), `FiberData470.csv` (470 nm excitation, v2)

**Stores (after discover):**
- `file0_chev1`: Stimulation column from 415 nm channel (all zeros. Garbage output.)
- `file0_chev2`: Output0 column from 415 nm channel (all zeros. Garbage output.)
- `file0_chev3`: Output1 column from 415 nm channel (all zeros. Garbage output.)
- `file0_chev4`: Input0 column from 415 nm channel (all zeros. Garbage output.)
- `file0_chev5`: Input1 column from 415 nm channel (all zeros. Garbage output.)
- `file0_chev6`: Region0G column from 415 nm channel (isosbestic control data).
- `file0_chev7`: Region1G column from 415 nm channel (isosbestic control data).
- `file1_chev1`: Stimulation column from 470 nm channel (all zeros. Garbage output.)
- `file1_chev2`: Output0 column from 470 nm channel (all zeros. Garbage output.)
- `file1_chev3`: Output1 column from 470 nm channel (all zeros. Garbage output.)
- `file1_chev4`: Input0 column from 470 nm channel (all zeros. Garbage output.)
- `file1_chev5`: Input1 column from 470 nm channel (all zeros. Garbage output.)
- `file1_chev6`: Region0G column from 470 nm channel (calcium signal data).
- `file1_chev7`: Region1G column from 470 nm channel (calcium signal data).

## `npm/sampleData_NPM_3`

NPM v2 recording with 4 fiber channels and non-standard timestamp columns. The photometry file uses a `ComputerTimestamp` column (milliseconds) rather than the default timestamp column (seconds). Copied as-is from the original (too small to stub without breaking tests). These recordings also contain large artifacts in the beginning of the recording. 

**Required settings:** `npm_timestamp_column_name="ComputerTimestamp"`, `npm_time_unit="milliseconds"`.
Same two-column shape as `sampleData_NPM_1`: `ttls.csv` is on the `ComputerTimestamp` clock, so the
default `SystemTimestamp` decouples events from the photometry.

**Files:** `signals.csv` (photometry, v2, 4 channels), `ttls.csv` (events, values 1 and 3)

**Stores (after discover + split):**
- `file0_chev1`: Region G0 isosbestic control
- `file0_chev2`: Region G1 isosbestic control
- `file0_chev3`: Region G2 isosbestic control
- `file0_chev4`: Region G3 isosbestic control
- `file0_chod1`: Region G0 calcium signal
- `file0_chod2`: Region G1 calcium signal
- `file0_chod3`: Region G2 calcium signal
- `file0_chod4`: Region G3 calcium signal
- `event1`: TTL events with value 1 in the event column
- `event3`: TTL events with value 3 in the event column

## `npm/sampleData_NPM_4`

NPM legacy format (no `LedState` header, rows interleaved by LED state). The event file contains boolean `True`/`False` values; with `split_events=True`, these become separate `eventTrue` and `eventFalse` stores. Also used for Step 2 idempotency testing (running Step 2 twice must not corrupt modality detection). Duration: 578.0 s (10th TTL event — 5 True + 5 False — at ~577.3 s). True/False is some user annotation of unknown meaning.

**Files:** `PagCeAVgatFear_14421.csv` (photometry, legacy), `PagCeAVgatFear_1442_ts0.csv` (events)

**Stores (after discover + split):**
- `file0_chev1`: Reigon0G isosbestic control
- `file0_chev2`: Region1G isosbestic control
- `file0_chev3`: Region2G isosbestic control
- `file0_chod1`: Region0G calcium signal
- `file0_chod2`: Region1G calcium signal
- `file0_chod3`: Region2G calcium signal
- `eventTrue`: TTL events where the event column value is `True`
- `eventFalse`: TTL events where the event column value is `False`

## `npm/sampleData_NPM_5`

Second NPM legacy format recording. Unlike `sampleData_NPM_4`, the event file contains a single event type with no boolean split. Copied as-is from the original (too small to stub without breaking tests). Also used for stub idempotency and duration unit tests.

**Files:** `PagCeAVgatFear_1512_1.csv` (photometry, legacy), `PagCeAVgatFear_1512_ts0.csv` (events)

**Stores (after discover):**
- `file0_chev1`: first column isosbestic control
- `file0_chev2`: second column isosbestic control
- `file0_chev3`: third column isosbestic control
- `file0_chod1`: first column calcium signal
- `file0_chod2`: second column calcium signal
- `file0_chod3`: third column calcium signal
- `event0`: single event type

## `npm/sampleData_NPM_6`

PhAT's `Sample2_NPM_1fiber.csv`, from the Donaldson Lab's [PhAT toolkit](https://github.com/donaldsonlab/PhAT) (MIT licensed, `LICENSE` alongside), and the reproducer for issue #337. Its header line `,Timestamp,msTimestamp,,Region0R,Region1G,,,LedState` carries four blank header cells and two timestamp columns, one of them named exactly `Timestamp`. Photometry only, with no TTL events. Duration: 16.0 s.

**Files:** `Sample2_NPM_1fiber.csv` (photometry, v2), `LICENSE`

The first row's `LedState` is `0`, which lights no LED; after it the state cycles `4`/`1`/`2` (560/415/470 nm), each crossed with the `Region0R` and `Region1G` regions. `Timestamp` is in seconds and `msTimestamp` is the same clock in milliseconds. The stub is truncated on the raw text lines rather than through `NpmRecordingExtractor.stub()`, so the committed file keeps its blank header cells.

---

# NWB (Neurodata Without Borders)

NWB format recordings using the `ndx-fiber-photometry` extension for the photometry signal. Event timestamps are stored either via the `ndx-events` extension or, as of NWB Schema 2.10.0 / pynwb 4.0, the core NWB `EventsTable` type. Each mock file is generated programmatically rather than truncated from real recordings; the regeneration script (and, where an extension pin is required, the isolated conda environment) is noted per file below.

## `nwb/mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2`

Minimal synthetic NWB file for testing the NWB recording extractor against the current version of `ndx-fiber-photometry` with `ndx-events==0.2`. Contains 3000 samples at 30 Hz across 2 channels (control and signal) and three event types. To regenerate, run `src/guppy/testing/scripts/create_mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2.py`.

**File:** `mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2.nwb`

**Photometry data:**
- `fiber_photometry_response_series`: 3000-sample × 2-channel array at 30 Hz; column 0 = control (isosbestic, 405 nm excitation), column 1 = signal (470 nm excitation)

**Events:**
- `events`: 10 timestamps (45–54 s); plain `ndx_events.Events`
- `labeled_events`: 15 timestamps (40–54 s) with 3 labels (`label_1`, `label_2`, `label_3`); `ndx_events.LabeledEvents`
- `AnnotatedEventsTable`: two event types; `ndx_events.AnnotatedEventsTable`
  - `Reward`: timestamps at 41–45 s
  - `Punishment`: timestamps at 55–59 s

## `nwb/mock_nwbfile_ndx_fiber_photometry_v0_1_ndx_events_v0_2`

Identical synthetic data to `mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2`, but the NWB file was created using `ndx-fiber-photometry==0.1.0`. The ndx-fiber-photometry v0.1.0 API differs from the current version in that device classes (`Indicator`, `OpticalFiber`, `ExcitationSource`, `Photodetector`, `DichroicMirror`, `BandOpticalFilter`) all live directly in `ndx_fiber_photometry` (no separate `ndx-ophys-devices` dependency), and `FiberPhotometry` only holds a `FiberPhotometryTable` (no virus/injection/indicator containers). Used to verify that the NWB extractor can read files produced by the older extension version. To regenerate, create the isolated conda environment at `src/guppy/testing/scripts/environment_ndx_fiber_photometry_v0_1_ndx_events_v0_2.yaml` and run `src/guppy/testing/scripts/create_mock_nwbfile_ndx_fiber_photometry_v0_1_ndx_events_v0_2.py`.

**File:** `mock_nwbfile_ndx_fiber_photometry_v0_1_ndx_events_v0_2.nwb`

**Photometry data:**
- `fiber_photometry_response_series`: 3000-sample × 2-channel array at 30 Hz; column 0 = control (isosbestic, 405 nm excitation), column 1 = signal (470 nm excitation)

**Events:**
- `events`: 10 timestamps (45–54 s); plain `ndx_events.Events`
- `labeled_events`: 15 timestamps (40–54 s) with 3 labels (`label_1`, `label_2`, `label_3`); `ndx_events.LabeledEvents`
- `AnnotatedEventsTable`: two event types; `ndx_events.AnnotatedEventsTable`
  - `Reward`: timestamps at 41–45 s
  - `Punishment`: timestamps at 55–59 s

## `nwb/mock_nwbfile_ndx_fiber_photometry_v0_2_core_events`

Identical photometry data to `mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2`, but the NWB file was created with `pynwb>=4`, whose core schema (NWB Schema 2.10.0) provides the `EventsTable` type natively (via `NWBFile.add_events_table()`) — no events extension is required. Core `EventsTable` has no categorical/meanings machinery; distinct event types within one table are distinguished by an optional text `annotation` column. Used to verify that the NWB extractor reads files that use the core pynwb 4.0 event types. To regenerate, create the isolated conda environment at `src/guppy/testing/scripts/environment_ndx_fiber_photometry_v0_2_core_events.yaml` and run `src/guppy/testing/scripts/create_mock_nwbfile_ndx_fiber_photometry_v0_2_core_events.py`.

**File:** `mock_nwbfile_ndx_fiber_photometry_v0_2_core_events.nwb`

**Photometry data:**
- `fiber_photometry_response_series`: 3000-sample × 2-channel array at 30 Hz; column 0 = control (isosbestic, 405 nm excitation), column 1 = signal (470 nm excitation)

**Events:**
- `simple_events`: 10 timestamps (45–54 s); plain core `EventsTable` with no annotation column; discovers as one event `simple_events`
- `annotated_events`: 10 timestamps (41–59 s) with a text `annotation` column; discovers as two events:
  - `annotated_events_Reward`: timestamps at 41–45 s
  - `annotated_events_Punishment`: timestamps at 55–59 s

---

# pyPhotometry

Open acquisition hardware from Akam and Walton (2019), a MicroPython Pyboard with two analog inputs and two onboard LED drivers, also sold by Open Ephys. A recording is one `.ppd` file: a two-byte little-endian header length, a header, then unsigned 16-bit little-endian words whose top fifteen bits are an analog sample and whose bottom bit is a digital line. The words cycle through the board's analog inputs, so a digital line rides in the low bit of the words of the input it shares a slot with and takes that input's rate and start time. There is no separate digital stream, which is why every session below still carries fluorescence even when it is here for its events.

Nothing in the file says how the words are laid out except the header's `mode` field, and the spelling of that field changed twice as the acquisition software evolved: one and the same acquisition is `GCaMP/RFP` under header version 0.1, `2 colour continuous` under 0.2 and 0.3, and `2EX_2EM_continuous` from 1.0 onward, and a fourth generation packs the mode as a byte before the JSON header exists at all. The mode-named sessions below cover those generations, since the vocabulary is what the reader dispatches on; three further sessions cover the shape of the digital pulses instead, and `full_pipeline_session` exists only to be long enough for the analysis steps to have something to work on.

Stores are named `detector_<n>_excitation_<m>` for the photodetector that was read and the excitation source that was lit, not for the board's two sockets. A shared `detector_<n>` prefix means a shared optical fiber and so one recording site: `1 colour time div.` reads one photodetector under two excitations, which is the signal-plus-isosbestic configuration, while `2 colour time div.` reads a detector per source and is two fibers.

The board has no simultaneous analog-to-digital converters, so the analog inputs in a file were never sampled at the same instants. In the strobed modes the sampling timer runs at the header's rate times the number of analog inputs and the interrupt advances one input per tick, so consecutive inputs are staggered by exactly one tick.

## `pyphotometry/two_colour_time_division`

Header version 0.2, mode `2 colour time div.`, 130 Hz, 20 s, 2,600 samples per analog input. Two fluorophores on separate fibers, strobed in turn. Version 0.2 differs from 0.3 only in carrying `LED_current`, so the two share a vocabulary and a layout, and this file is here for its mode string rather than for its version. Cut from a 93-minute recording at a cycle boundary, so both inputs keep the same sample count; the header is copied byte for byte.

**Stores:**
- `detector_1_excitation_1`: first slot, in volts
- `detector_2_excitation_2`: second slot, a different fiber, one tick of the 260 Hz sampling timer later
- `digital_1`: 1 pulse onset
- `digital_2`: 3 pulse onsets

Provenance: Blanco-Pozo, M., Akam, T. and Walton, M. (2023), Two-step_dopamine, OSF `osf.io/u6xrc`, CC BY 4.0. Original name `m28_DMS_L-2019-05-06-121700.ppd`.

## `pyphotometry/full_pipeline_session`

The integration-suite session, and the only one here that is not chosen for a property of the format. The mode-coverage fixtures above are cut for what their headers say, which for most of them means a few seconds of recording; that is enough to read but not enough to analyze, because Step 4's default PSTH window runs from -10 s to +20 s and an event needs that much recording on either side of it to produce a trial. A 20-second session yields a PSTH with no usable trials, which the step-4 and step-5 tests would pass anyway since they assert the output files exist rather than that they hold anything. This session is cut long enough to avoid that, and is named for the job rather than for its mode so the requirement is visible.

Header version 0.2, mode `2 colour time div.`, 130 Hz, 180 s, 23,400 samples per analog input. Cut from the start of the same 93-minute recording as `two_colour_time_division`, at a cycle boundary, with the header copied byte for byte.

**Stores:**
- `detector_1_excitation_1`: first slot, mapped to `signal_region` by the integration fixtures
- `detector_2_excitation_2`: second slot, mapped to `control_region`, one tick of the 260 Hz sampling timer later. This mode reads a detector per source, so the two are two fibers rather than a signal-plus-isosbestic pair; pairing them is a fixture convenience, as the pipeline needs a signal and a control to run
- `digital_1`: 8 pulse onsets
- `digital_2`: 33 pulse onsets, mapped to `ttl`; 30 of them survive as PSTH trials, the other 3 falling too close to an edge for the window to fit

Provenance: Blanco-Pozo, M., Akam, T. and Walton, M. (2023), Two-step_dopamine, OSF `osf.io/u6xrc`, CC BY 4.0. Original name `m28_DMS_L-2019-05-06-121700.ppd`.

## `pyphotometry/two_colour_continuous`

Header version 0.3, mode `2 colour continuous`, 1 kHz, 7.6 s. Both LEDs lit continuously rather than strobed, which is the only mode that reaches the board's maximum rate since nothing has to be alternated. Its two inputs share the timebase the header states: the conversions are still sequential, but by an amount the file does not record, so no offset is claimed. Kept whole.

**Stores:**
- `detector_1_excitation_1`, `detector_2_excitation_2`: the two slots, both starting at 0 s
- `digital_1`, `digital_2`: both recorded, neither fires

Provenance: Formozov, Dieter and Wiegert (2023), G-Node GIN `10.12751/g-node.37lm4m`, CC0. Original name `405_0-2021-02-27-094810.ppd`.

## `pyphotometry/four_colour_time_division`

Header version 0.3, mode `4 colour time div.`, header rate 65 Hz, 190 s. Written by a laboratory fork of the acquisition software for a fused fiber coupler setup, and **the one session here that GuPPy refuses to read**. Each analog line alternates two excitation sources, so the file holds four signals at 32.5 Hz, half the rate the header advertises, and nothing but the mode string distinguishes it from an ordinary two-signal recording. The layout is stated nowhere the file or the firmware can be asked, so reading it is refused rather than guessed at, and the error names the fork and its paper instead of calling the mode unknown. Kept whole, as the fixture for that refusal.

**Stores:** none. `discover_events_and_flags` raises.

Provenance: as above, `10.12751/g-node.37lm4m`, CC0. Original name `ABtest_4BP-2021-03-19-174356.ppd`.

## `pyphotometry/two_excitation_two_emission_pulsed`

Header version 1.0, mode `2EX_2EM_pulsed`, 130 Hz, 15 s, 2,000 samples per analog input. The current vocabulary, and the only generation whose header states its signal counts (`n_analog_signals`, `n_digital_signals`), so a reader can cross-check the mode against them rather than trusting either alone. Used as the unit-test contract session.

**Stores:**
- `detector_1_excitation_1`, `detector_2_excitation_2`: the two slots, one tick of the 260 Hz sampling timer apart
- `digital_1`: 15 pulse onsets
- `digital_2`: 7 pulse onsets

Provenance: synthetic. The header fields and sample packing of this generation with generated samples, a subject id of `synthetic` and placeholder timestamps; no measurement from any recording is present.

## `pyphotometry/gcamp_rfp_dif`

Header version 0.1, mode `GCaMP/RFP_dif`, 130 Hz, 15 s, 2,000 samples per analog input. The earliest JSON header, whose mode field names the fluorophores rather than the acquisition. Its version is written as a JSON number (`0.1`) rather than the string it becomes from 0.3 onward, so a reader comparing versions has to survive both spellings. Fewer header fields than any later generation: no `LED_current`, no `end_time`, no signal counts.

**Stores:** as above (`detector_1_excitation_1`, `detector_2_excitation_2`, `digital_1` with 15 onsets, `digital_2` with 7).

Provenance: synthetic, as above.

## `pyphotometry/two_signals_200hz`

The generation before the JSON header, where the 42 bytes behind the length are a fixed layout: bytes 0-11 are the subject id and bytes 12-30 the timestamp, both fixed-width slices rather than a delimited pair, byte 31 is a mode code indexing the three acquisition modes that generation offered, bytes 32-33 are the sampling rate, and the last eight are the volts per division of each photodetector as 32-bit integers scaled by a billion. A failed JSON parse is the only thing that identifies the generation, so it is a version signal rather than a damaged file. 200 Hz, 10 s, 2,000 samples per slot. Its mode byte is 3, which is `GCaMP/RFP_dif` and strobed, so its slots are staggered like any other strobed recording.

**Stores:** as above, staggered by one tick of the 400 Hz sampling timer.

Provenance: synthetic, as above.

## `pyphotometry/narrow_pulses_and_idle_line`

Header version 0.3, mode `1 colour time div.`, 130 Hz, 5 s, 649 samples per line, from the start of the recording. The ordinary digital case, and beside it a line that was recorded and never fired.

**Stores:**
- `detector_1_excitation_1`, `detector_1_excitation_2`: one photodetector read under two excitation sources strobed in turn, which is the signal-plus-isosbestic pair on one fiber. The shared `detector_1` prefix is what says they are one recording site
- `digital_1`: a pulse train at about 30 Hz beginning 3.75 s in, 37 pulses one or two samples wide. Low before the train starts and low at the last sample, so every high period closes
- `digital_2`: never goes high, which is an event type that existed in the recording and has no occurrences

Provenance: Formozov, Dieter and Wiegert (2023), G-Node GIN `10.12751/g-node.37lm4m`, CC0. Original name `671_FFC50-202-2021-06-08-165248.ppd`.

## `pyphotometry/starts_and_ends_high`

Both boundary cases in one window: a pulse whose onset is before the file, and a pulse whose end is after it. Header version 0.3, mode `1 colour time div.`, 130 Hz, 1.1 s, 143 samples per line, beginning 15.46 s into the same recording. The window is positioned on two-sample-wide runs on purpose, since a one-sample pulse cannot be entered or left mid-run.

**Stores:**
- `detector_1_excitation_1`, `detector_1_excitation_2`: as above
- `digital_1`: already high at the first sample, so the pulse that opens the file has no onset to report; still high at the last sample, so the final pulse IS reported because its rising edge was observed. 33 onsets between them
- `digital_2`: never goes high

Provenance: as above, `10.12751/g-node.37lm4m`, CC0.

## `pyphotometry/wide_pulses_on_both_lines`

Two lines both firing, with pulses wide enough to be several samples of a high period rather than one or two. Header version 0.2, mode `2 colour time div.`, 130 Hz, 8.5 s, 1,100 samples per line, beginning 4,732.79 s into the recording, so the file states the recording's start time and not the window's. Every edge of the window falls on a low sample of both lines, so every high period in it is closed.

**Stores:**
- `detector_1_excitation_1`, `detector_2_excitation_2`: as above
- `digital_1`: two pulses 8 samples wide (61.5 ms)
- `digital_2`: two pulses 6 samples wide (46.2 ms), starting one tick of the 260 Hz sampling timer after `digital_1`

Provenance: Blanco-Pozo, M., Akam, T. and Walton, M. (2023), Two-step_dopamine, OSF `osf.io/u6xrc`, CC BY 4.0. Original name `m28_DMS_L-2019-05-06-121700.ppd`.

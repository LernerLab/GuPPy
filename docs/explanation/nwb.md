# Fiber photometry data in NWB

[Neurodata Without Borders](https://nwb.org) (NWB) is a common data format for neurophysiology, holding one experimental session per file. What NWB is and how it is structured are covered by [NWB's own documentation](https://nwb-overview.readthedocs.io/); this page covers only GuPPy's side of it — what GuPPy puts in a file, what it needs from you in order to do that, and where its export stops.

For the mechanics of running an export, see [Export a session to NWB](../how-to/export-to-nwb.md).

## What GuPPy writes

An export has three parts:

```text
<session_name>_output_<run_name>.nwb
  acquisition/                     the raw photometry, as it was recorded
    <signal series>                FiberPhotometryResponseSeries, one column per recording site
    <control series>               the isosbestic channel, same shape
  events/
    <event store>                  one EventsTable per raw behavioral event store
    GuppyEvents                    the onsets GuPPy actually analyzed
  processing/guppy/
    recording_sites                registry: one row per recording site
    events                         registry: one row per event GuPPy aligned to
    <trace_type>_<site>            control_fit, dff and z_score traces
    transients_<site>_<metric>     detected transient peaks
    transient_summary              frequency and mean amplitude per trace
    psth_<site>_<metric>           peri-event PSTHs
    peak_auc_<site>_<metric>       peak and area summaries
    cross_correlation_<metric>_<siteA>_<siteB>
    valid_signal_intervals         what survived artifact removal
    guppy_parameters               the parameters the run was produced with
```

The `acquisition` group is the recording, unmodified — exporting does not replace your raw data with GuPPy's version of it. Everything GuPPy computed lives under `processing/guppy`, which is how NWB separates what was measured from what was inferred.

## The GuPPy extension types

The core NWB schema and the fiber-photometry extension cover the raw side. GuPPy's derived products have no core type, so [`ndx-guppy`](https://github.com/catalystneuro/ndx-guppy) defines them:

| Type | What it holds |
|------|---------------|
| `GuppyRecordingSitesTable` | The registry of recording sites, one row each |
| `GuppyEventsTable` | The registry of events GuPPy aligned to, one row each |
| `GuppyDerivedResponseSeries` | One derived trace — control fit, ΔF/F or z-score — for one recording site |
| `GuppyTransientsTable` | Detected transient peaks for one recording site and trace type |
| `GuppyTransientSummaryTable` | Per-session transient frequency and mean amplitude |
| `GuppyPSTH` | A peri-event PSTH for one recording site and trace type |
| `GuppyPeakAUC` | The peak and area-under-curve summary of a PSTH |
| `GuppyCrossCorrelation` | A peri-event cross-correlation between two recording sites |
| `GuppyValidSignalIntervals` | The intervals kept as valid signal after artifact removal |
| `GuppyParameters` | The analysis parameters the session was processed with |

The two registry tables are what hold this together. A GuPPy **recording site** is a processing-level entity — a signal and isosbestic pair collapsed into one derived trace — and every product above references its row rather than repeating its name as a string. The registry in turn points back at the rows of the acquisition's fiber photometry table, so a z-score trace reaches the physical fiber it came from through one chain of links.

## Metadata GuPPy cannot know

An analysis output describes signals, not the apparatus that produced them. GuPPy can read your `storesList.csv` and tell that a session has two recording sites, each with a signal and a control channel — but not which fiber, which LED wavelength, which virus, or which animal. NWB requires that chain, and no converter can invent it.

That is what *Step 6: Input Metadata* exists to collect, and why it is a form rather than something derived. It is also why the same information is often identical across an entire cohort: it describes your rig and your preparation, so it is written once and reused.

## When GuPPy is not the whole experiment

One session, one file is the point. A session that also produced electrophysiology, behavioral video, or pose estimates should have all of it in that one file — but GuPPy only ever sees the photometry, so its export can only write the photometry.

The way past that is [NeuroConv](https://neuroconv.readthedocs.io), the conversion library GuPPy's own export is built on. NeuroConv provides an *interface* per acquisition system or analysis tool — one for GuPPy's outputs, others for the ephys, video and tracking formats — and runs several of them into a single file. Working at that level is more effort than clicking **Export to NWB**, and it is the right amount of effort when the GUI's one-session-of-photometry scope is not your experiment. [Beyond the GUI](../how-to/export-to-nwb.md#beyond-the-gui-combining-with-other-data) links the relevant NeuroConv guides.

## Sharing on DANDI

An exported file is ready to upload to DANDI, provided the subject metadata is complete — which is why *Step 6* marks subject ID, sex and species as required.

The traffic runs both ways: GuPPy can also *read* a session from DANDI or from a local NWB file. In that case the export adds GuPPy's outputs to a copy of the source rather than building a file from scratch, and the source is left untouched.

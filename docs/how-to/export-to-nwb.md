# Export a session to NWB

[NWB](https://nwb.org) is a common data format for neurophysiology, and the one
the [DANDI Archive](https://dandiarchive.org) publishes. GuPPy can package a
finished run — the raw traces, the events, and every derived result — into a
single `.nwb` file.

Two **optional** steps handle this, and they follow *Step 5: Visualization* in
the sidebar:

- **Step 6 : Input Metadata** — describe the hardware, the reagents and the
  subject. Writes `nwb_metadata.yaml` into the run folder.
- **Step 7 : Export to NWB** — write the file.

If you only need GuPPy's plots and tables, skip both.

These steps cover a session whose data GuPPy read. If the same experiment also
produced electrophysiology, video, or anything else GuPPy does not ingest, that
data belongs in the *same* NWB file — see
[Beyond the GUI](#beyond-the-gui-combining-with-other-data).

For what an exported file holds and why, see
[Fiber photometry data in NWB](../explanation/nwb.md).

## Before you start

Run **Step 1: Label Stores**, **Step 2: Read Raw Data** and **Step 3:
Preprocess** on the sessions you want to export. **Step 4: PSTH Computation**
is optional, but PSTHs, transients, peak/AUC summaries, cross-correlations and
binned metrics reach the file only if it ran, and the same holds for **Tonic
Analysis** and the epoch means it writes.

One Step 4 output does not reach the file yet: the results of **Compute PSTH
Significance?**. If the runs you selected have them, *Step 7* exports
everything else and warns you that the significance results were left out. They
stay in each run's `psth_significance_output` folder, and a future release will
export them.

Both steps read the run folder, so select the session folder(s) **and** the
output run(s) on the homepage before clicking either.

Two pipeline choices make a run unexportable. Both steps check before writing
anything and refuse the whole batch:

- **Combine Data? = True.** Combining collapses a run group into one output
  directory, so there is no single session the outputs belong to. Re-run the
  pipeline with **Combine Data?** set to false.
- **Artifacts removed with `concatenate`.** That method re-times the surviving
  samples onto a fresh timeline, which breaks alignment to the acquisition
  clock. Re-run **Remove Artifacts** after choosing *replace with NaN* on the
  **Select Artifact Windows** page, or re-run *Step 3: Preprocess* to drop the
  removal entirely. See
  [Choosing a removal method](artifact-removal.md#choosing-a-removal-method).

## Step 6: Input Metadata

GuPPy knows the shape of your recording — how many recording sites, which store
is signal and which is control — but nothing about what produced it. NWB needs
the rest, and no analysis output can supply it. This step collects it.

Click **Input Metadata**. A page opens in a new browser tab for each selected
session, titled `Metadata GUI - <session> (<run>)`. The main GuPPy tab stays
responsive.

```{image} ../_static/images/export_to_nwb_button.png
:alt: The bottom of the sidebar showing Step 6, Input Metadata, and Step 7, Export to NWB, with its progress bar, positioned below Step 5, Visualization
:width: 50%
```

1. Fill in **Core NWB metadata**: the *Session*, *Experimenter* and *Subject*
   fields. Required fields are marked with a red asterisk, and the `?` beside a
   field explains what it expects.

   **Session start time** is required for every acquisition format except TDT
   and NWB, whose files record it themselves. Give it in ISO 8601, e.g.
   `2018-10-30T10:33:32-05:00`.

   ```{image} ../_static/images/input_metadata.png
   :alt: The top of the NWB metadata page: the page header, the strip for reusing metadata from another session, and the Core NWB metadata card with its Session, Experimenter and Subject fields filled in, above the collapsed Optical hardware and Biological reagents groups
   :width: 100%
   ```

2. Define your **Optical hardware**. Each category holds *models* (the part as
   the manufacturer sells it — a fiber model, an LED model) and *instances* (the
   one in your rig, pointing at its model). Add each model first, then the
   instances that use it. The fields come from the NWB fiber-photometry schema
   itself, so they match what the format expects.

3. Define your **Biological reagents** in order: a **Virus**, then a **Virus
   injection** that references it, then an **Indicator** that references the
   injection. Each link is a dropdown listing what you have already defined.

4. Link everything under **Fiber-photometry channels**. These rows are *fixed* —
   they come from your `storesList.csv`, one group per recording site, each with
   its control and signal channel. For every channel, give the **excitation**
   and **emission wavelength**, and choose its indicator, optical fiber,
   excitation source and photodetector. The dichroic mirror and the two filters
   are optional.

   ```{image} ../_static/images/input_metadata_channels.png
   :alt: The Fiber-photometry channels card showing one recording site group containing a control and a signal channel card, each with excitation and emission wavelength fields and dropdowns linking the indicator, optical fiber, excitation source and photodetector
   :width: 100%
   ```

5. Click **Build & preview YAML**, then **Save metadata**. The alert pane above
   the buttons turns red and names every missing field by name; **Save metadata**
   refuses until it is clean. On success the file lands at
   `<run_folder>/nwb_metadata.yaml` and the path appears below the buttons.

## Reusing metadata across sessions

Most of this form describes your rig and your virus, not the session — it is
identical across a whole cohort. Fill it in once, save, then in each remaining
session use the **Reuse metadata from another session** strip at the top of the
page to load that `nwb_metadata.yaml`. The whole form repopulates, and you
change only the subject and session fields.

The YAML is the authoritative artifact, and it is plain text — readable,
diffable, and editable outside GuPPy:

```yaml
NWBFile:
  session_description: Recording during RI60 training.
  session_start_time: '2018-10-30T10:33:32-05:00'
Subject:
  subject_id: Photo_63_207
  sex: M
  species: Mus musculus
```

The **Advanced — raw YAML** card at the bottom of the page shows the same
document live. Edit it directly if the form cannot express something, then save.

## Step 7: Export to NWB

Click **Export to NWB**. A progress bar appears in the sidebar directly below
the button and advances once per session.

```{image} ../_static/images/export_to_nwb_progress.png
:alt: The bottom of the sidebar during an export, with the progress bar below the Export to NWB button partly filled
:width: 50%
```

Each session is written to its own run folder as
`<session_name>_output_<run_name>.nwb` — named after the run folder, so exports
from several runs can be pooled into one directory without renaming.

A session that fails is skipped and the batch continues; the notification at the
end names which ones failed and why. A clean batch reports
*Export to NWB complete.*

The file holds the raw photometry under `acquisition`, the behavioral events
under `events`, and everything GuPPy computed under `processing/guppy`. See
[Fiber photometry data in NWB](../explanation/nwb.md) for the full layout, and
the [Output data model](../reference/outputs.md) for the on-disk entry.

## Sessions GuPPy read from an NWB file

If GuPPy analyzed a session out of an NWB file — a local `.nwb` in the session
folder, or a DANDI asset — **Step 6 opens no form at all**. It tells you so and
stops: that file already carries the devices, the fiber-photometry chain and the
session start time, which is everything the form would ask for.

Run **Step 7** directly. It adds GuPPy's outputs to a copy of the source file,
written to the run folder under the usual name. **The source file is never
modified**, and the copy keeps the extension versions the source was written
with.

## Beyond the GUI: combining with other data

The export above handles one session's photometry: the traces GuPPy read, the
events it aligned to, and the results it computed. That is the whole story for a
photometry-only experiment.

It is not the whole story if the same session also produced electrophysiology,
behavioral video, pose estimates, or anything else GuPPy never sees. NWB's value
is that one file holds one session — splitting a session across a GuPPy export
and a second file assembled some other way gives up most of the point. GuPPy's
export is built on [NeuroConv](https://neuroconv.readthedocs.io), and the same
`GuppyInterface` behind Step 7 can be combined with NeuroConv's interfaces for
every other stream to write a single file. Reach for NeuroConv directly when
that is what you need.

- [GuPPy Fiber Photometry](https://neuroconv.readthedocs.io/en/main/conversion_examples_gallery/fiberphotometry/guppy_fp.html)
  — using `GuppyInterface` and `GuppyConverter` on the outputs Step 7 exports.
- [Annotating fiber photometry metadata](https://neuroconv.readthedocs.io/en/main/how_to/annotate_fiber_photometry_metadata.html)
  — the hardware chain Step 6 collects, in its native form.
- [Combining data interfaces](https://neuroconv.readthedocs.io/en/main/conversion_examples_gallery/combinations/spikeglx_and_phy.html)
  — running several interfaces into one NWB file.

Notes:

- Re-running Step 7 overwrites the `.nwb` in place. Each export rebuilds from
  the run folder, so exports do not compound.
- The metadata you enter in Step 6 is applied *on top of* what GuPPy and the
  acquisition files already supply — it only ever adds or replaces, and never
  removes what was read from the data.
- Group averaging is not exported. A group directory is a group product
  with no session behind it, and combined runs are refused outright.
- Export errors name both the session and the run, so a partial batch tells you
  exactly which run folder to fix.
- `guppy --export-logs` copies the full log, including the traceback behind a
  failed export, to your desktop.

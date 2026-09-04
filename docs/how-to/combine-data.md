# Combine a session split across two data files

Some acquisition systems split one continuous recording into two separate
session folders — for example a system that writes a new file when a
recording is paused and resumed, or splits a long session at a fixed size
limit. **Combine Data?** stitches the matching channels from both halves back
into a single continuous trace before preprocessing. This is an **unusual**
case — if your sessions are already one file each, skip this guide.

## Before you start

Each half needs its own session folder. The recording sites in both folders
must be labeled identically in **Step 1: Label Stores** — the same
`signal_<site>` / `control_<site>` pair names on both halves — otherwise
combining fails with a message telling you to re-run Step 1.

## Naming your session folders

GuPPy decides which half comes first, and where the combined result is
written, by sorting the two session folders' paths alphabetically — not by
recording time. Name your folders so alphabetical order matches recording
order (for example `Session_part1` / `Session_part2`, or a date-time prefix).
If the names sort the wrong way, the halves are spliced in the wrong order
with no error.

The combined result is written into whichever folder sorts first, overwriting
its Step 3 outputs. The other folder's outputs are left untouched.

## Matching the two halves

GuPPy groups sessions to combine by their **run name** — the suffix on the
`_output_<run name>` directory Step 1 creates — not by session name. Both
halves must use the same run name.

1. Run **Step 1: Label Stores** on the first half, labeling its recording
   sites as usual. Leave **Run name** blank so it defaults to `1` (or set an
   explicit name if you prefer — the second half just needs to match it).

   ```{image} ../_static/images/combine_data_run_name.png
   :alt: The Label Stores GUI's "Choose how to save this store_array" section, showing the over-write-or-create-new selector set to create a new file and an empty Run name field with placeholder text "optional — defaults to next available integer"
   :width: 70%
   ```

2. Run **Step 1: Label Stores** on the second half, using the **same**
   recording-site labels and the **same** run name as the first.
3. Back on the homepage, select both session folders together in **Input
   Folder Selection**.

   ```{image} ../_static/images/combine_data_sessions_selected.png
   :alt: The Input Folder Selection file browser with both session folders, Photo_048_392-200728-121222 and Photo_63_207-181030-103332, moved into the Selected files list
   :width: 100%
   ```

4. Open **Output Folder Selection** and set **Run name(s) for all sessions** to the run
   name both halves share (e.g. `1`), which selects each session's `..._output_1`.
5. In the **Individual Analysis** card, set **Combine Data?** to `True` (next
   to **# of cores** — see the [parameters screenshot](../tutorials/first_analysis.md#set-parameters)
   in the getting-started tutorial for where this sits).

## Running the pipeline

Run **Step 2: Read Raw Data**, then **Step 3: Preprocess**. Combining happens
automatically inside Step 3, right after per-session timestamp correction, and
writes the merged result into the first (alphabetically) session's output
folder — see [Naming your session folders](#naming-your-session-folders)
above.

Continue to **Step 4: PSTH Computation** and **Step 5: Visualization** with
**Combine Data?** still `True`. Both already know to read from the merged run
instead of treating the two sessions separately.

## What lands on disk

Combining adds one file to the destination run folder:

| File | Contents |
|------|----------|
| `combine_storesList.csv` | Merged store mapping, union of both sessions' `storesList.csv` |

The combined traces themselves overwrite that folder's
`timeCorrection_<site>.hdf5`, `<store_label>.hdf5`, and `<event>_<site>.hdf5`
files. See [Output data model](../reference/outputs.md) for the full file
layout.

## Notes

- Both files being combined must share the same sampling rate; GuPPy raises
  an error naming the mismatched files if they don't.
- Combining is re-runnable. Re-running Step 3 with **Combine Data?** on
  recomputes the merge from the raw, per-session data each time, so repeated
  runs don't compound.

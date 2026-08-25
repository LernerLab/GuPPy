# Average results across sessions

Group analysis pools an event's response — PSTH, peak/AUC, transient frequency and amplitude,
and cross-correlation — across a set of output runs into one averaged output, with one column per
member run instead of one column per trial. Use it to see a cohort- or animal-level average rather
than reading each session's plot separately. It is **optional**: most analyses only need
per-session results, produced without touching the Group Analysis card at all.

A group is a named output directory, so you can keep several groups side by side — a `saline_group`
and a `cocaine_group`, say — and visualize them together.

## Before you start

Group analysis averages results that already exist — it does not compute a PSTH from raw
traces. Run **Step 1** through **Step 4** individually on every session you want to include. See
[Your First Analysis](../tutorials/first_analysis.md) for that baseline workflow. Only once
every session has its own PSTH output on disk can its run folder be added to a group.

Every member run also needs the **same recording-site labels** (the names you gave
channels in Step 1's Label Stores GUI, e.g. `DMS`, `DLS`). GuPPy checks this when you run the
step and stops with an error naming the mismatched run if the sites don't line up. The
*behavioral event* labels may differ between members — averaging a `novelobject` session
alongside a `novelfemale1` session is supported, and each event is averaged over just the members
that recorded it.

## Creating a group

1. Open the **Group Analysis** card in the main area (collapsed by default).
2. In the first file browser, select the **output run directories** to average — the
   `<session>_output_<run>` folders inside each session, not the session folders themselves. A
   summary line below the browser echoes what you have selected. This browser is independent of
   the Individual Analysis selection, so the two never interfere, and members may live under
   different parent directories.
3. In the second file browser, select the **destination directory** the group is written into.
4. Type a **Group name**. The name may not contain path separators, `..`, `_output_`, or `_group`.
5. Click **Group Analysis** in the sidebar.

   ```{image} ../_static/images/group_analysis_card.png
   :alt: The Group Analysis card, expanded, showing the member run browser, the destination browser, the group name field and the existing-groups list
   :width: 100%
   ```

The group is written to `<destination>/<name>_group/`.

Re-running a group name rebuilds that directory from scratch, so dropping a member removes its
results rather than leaving them behind. If a directory of that name already exists but GuPPy did
not create it, the step stops rather than deleting it — pick a different name or destination.

## Visualizing a group

A group directory is an ordinary output directory to the visualizer, so there is no separate mode
to switch on:

1. In the Group Analysis card's **Existing groups** list, select the groups you want to open.
   The list shows the groups found in the selected destination directory.
2. Click **Open Visualization GUI** in the sidebar.

One dashboard opens per selected group, with one line per member run in place of one line per
trial. Selected session runs open their own dashboards at the same time, so you can compare a
group against an individual session in one click.

```{image} ../_static/images/group_psth_plot.png
:alt: The Visualization dashboard's PSTH plot showing one trace per member run in the group
:width: 100%
```

Selecting exactly one group also reloads its members and name into the fields above, so you can
inspect what a group contains or re-run it after re-analyzing a session.

## What lands on disk

In the `<name>_group/` directory:

| File | Contents |
|------|----------|
| `group_members.json` | The run folders this group averaged, in averaging order |
| `GuPPyParamtersUsed.json` | The parameters the averaging ran under |
| `storesList.csv` | The store mapping, listing the events the group actually holds |
| `<event>_<site>_<metric>.h5` | Group PSTH: one column per member run |
| `peak_AUC_<event>_<site>_<metric>.h5` / `.csv` | Every member's peak/AUC rows concatenated |
| `freqAndAmp_<metric>.h5` / `.csv` | One row per member run |
| `cross_correlation_output/corr_*.h5` | One column per member run |

Group PSTH columns are named after the member run folder, and their order matches
`group_members.json`, so column *n* is member *n*.

See [Output data model](../reference/outputs.md#group-analysis-group-directories) for the
full file layout, including the empty placeholder files also written there.

## Re-running

Group averaging fully recomputes the group directory from its members' current outputs each
time — nothing compounds. To add a member, re-analyze one, or drop one from the group, adjust the
selection and click **Group Analysis** again.

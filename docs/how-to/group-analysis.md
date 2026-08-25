# Average results across sessions

Group analysis pools an event's response — PSTH, peak/AUC, transient frequency and amplitude,
and cross-correlation — across a set of output runs into one averaged output, with one column per
member run instead of one column per trial. Use it to see a cohort- or animal-level average rather
than reading each session's plot separately. It is **optional**: most analyses only need
per-session results, produced without touching the group steps at all.

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

## Defining a group

Groups work the way individual runs do: **Label Groups** defines one, and the
**Group Output Folder Selection** card picks which defined groups you are working with — once,
for every step that touches them.

1. Click **Label Groups** in the sidebar. A new browser tab opens with the Label Groups page.
2. Leave the mode at **create_new_group**.
3. Under **1 · What goes into the group?**, pick the **member runs** — the
   `<session>_output_<run>` directories to average, not the session folders themselves. They may
   live under different parent directories.
4. Under **2 · Where does the group go?**, type a **Group name** and pick the **destination
   directory** the group is written into. The name may not contain path separators, `..`,
   `_output_`, or `_group`.
5. Click **Save group definition**.

   ```{image} ../_static/images/label_groups_page.png
   :alt: The Label Groups page, with the member-runs browser on the left and the group name and destination browser on the right
   :width: 100%
   ```

This writes `<destination>/<name>_group/` containing only `group_members.json` — the group's
definition. It holds no results yet, in the same way a run folder holds `storesList.csv` before
Step 2 fills it.

To change a group later, open Label Groups again and switch the mode to **edit_existing_group**.
Section 2 becomes a browser for the group itself: navigate to the `<name>_group` directory you
want to change, and its recorded members load into the member browser ready to adjust. Editing
needs nothing selected on the homepage first.

## Selecting and averaging

1. Open the **Group Output Folder Selection** card and tick the `<name>_group` directories you
   want to work with. This is the group counterpart of Output Folder Selection, and like that
   card you choose once: the same selection serves both averaging and visualization.
2. Click **Group Analysis** in the sidebar.

Each selected group is averaged from its recorded members into its own directory. Re-running
rebuilds a group's results from scratch, so dropping a member removes its columns rather than
leaving them behind; the definition itself is preserved.

## Visualizing a group

A group directory is an ordinary output directory to the visualizer, so there is no separate mode
to switch on and nothing to re-select — just click **Open Visualization GUI** in the sidebar.

A group can be visualized on its own: unlike Steps 2–4, Step 5 does not need an output directory
picked for every selected session, so you can leave the individual selection empty and open only
the groups.

One dashboard opens per selected group, with one line per member run in place of one line per
trial. Selected session runs open their own dashboards at the same time, so you can compare a
group against an individual session in one click.

```{image} ../_static/images/group_psth_plot.png
:alt: The Visualization dashboard's PSTH plot showing one trace per member run in the group
:width: 100%
```

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

Group averaging fully recomputes a group's results from its members' current outputs each time —
nothing compounds. To re-analyze a member, just click **Group Analysis** again. To add or drop a
member, edit the group in **Label Groups** first, then re-run.

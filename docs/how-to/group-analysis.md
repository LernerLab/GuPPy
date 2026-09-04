# Average results across sessions

Group analysis pools an event's response across several output runs into one averaged result,
covering the PSTH, peak/AUC, transient frequency and amplitude, and cross-correlation. The
averaged output has one column per member run where a single session's output has one column per
trial, which gives you a cohort- or animal-level view without reading each session's plot
separately. Group analysis is optional. Most analyses only need per-session results, and those
never touch the group steps.

A group is a named output directory, so several can sit side by side. You might keep a
`saline_group` and a `cocaine_group` and visualize both at once.

## Before you start

Group analysis averages results that already exist on disk. Run **Step 1** through **Step 4** on
every session you want to include first; see [Your First Analysis](../tutorials/first_analysis.md)
for that baseline workflow. A run folder can join a group once it has its own PSTH output.

Every member run also needs the same **recording-site labels**, meaning the names you gave the
channels in Step 1's Label Stores GUI, such as `DMS` or `DLS`. GuPPy checks this when the step
runs and stops with an error naming the mismatched run if the sites disagree.

Behavioral **event** labels are free to differ between members. Averaging a `novelobject` session
alongside a `novelfemale1` session works, and GuPPy averages each event over just the members that
recorded it.

## Defining a group

Groups follow the same pattern as individual runs. **Label Groups** defines one, and the **Group
Output Folder Selection** card picks which defined groups you are working with, once, for every
step that touches them.

1. Click **Label Groups** in the sidebar. A new browser tab opens with the Label Groups page.
2. Leave the mode at **create_new_group**.
3. Under **1 · What goes into the group?**, pick the member runs. These are the
   `<session>_output_<run>` directories found inside each session. They may live under different
   parent directories.
4. Under **2 · Where does the group go?**, type a **Group name** and pick the destination
   directory. The name may not contain path separators, `..`, `_output_`, or `_group`.
5. Click **Save group definition**.

   ```{image} ../_static/images/label_groups_page.png
   :alt: The Label Groups page, with the member-runs browser on the left and the group name and destination browser on the right
   :width: 100%
   ```

This writes `<destination>/<name>_group/` containing a single file, `group_members.json`, which
records the group's membership. The group has no results at this point, much like a run folder
that holds `storesList.csv` before Step 2 fills it.

To change a group later, open Label Groups again and switch the mode to **edit_existing_group**.
Section 2 becomes a browser for the group itself. Navigate to the `<name>_group` directory you
want to change and its recorded members load into the member browser, ready to adjust. Editing
works whether or not anything is selected on the homepage.

## Selecting and averaging

1. Open the **Group Output Folder Selection** card and tick the `<name>_group` directories you
   want to work with. You choose once: the same selection serves both averaging and
   visualization.
2. Click **Group Analysis** in the sidebar.

Each selected group is averaged from its recorded members into its own directory. Re-running
rebuilds a group's results from scratch, so dropping a member also drops its columns. The group's
definition survives the rebuild.

## Visualizing a group

A group directory behaves like any other output directory in the visualizer, so there is no mode
to switch on and nothing to select a second time. Click **Open Visualization GUI** in the sidebar.

You can visualize a group on its own. Step 5 accepts an empty individual selection as long as a
group is ticked, so you can leave Output Folder Selection alone and open only the groups.

One dashboard opens per selected group, with one line per member run where a session's dashboard
has one line per trial. Any selected session runs open their own dashboards at the same time, so a
group and an individual session can be compared in one click.

```{image} ../_static/images/group_psth_plot.png
:alt: The Visualization dashboard's PSTH plot for a group, showing the averaged trace with a shaded error band across member runs
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

See [Output data model](../reference/outputs.md#group-analysis-group-directories) for the full
file layout.

## Re-running

Group averaging recomputes a group's results from its members' current outputs every time, so
nothing accumulates across runs. After re-analyzing a member, click **Group Analysis** again. To
add or drop a member, edit the group in **Label Groups** first, then re-run.

# Glossary

GuPPy uses one canonical vocabulary for the entities that flow through the
pipeline. This page defines those terms and records how they map onto the
files and keys GuPPy writes to disk.

## Core terms

```{glossary}
store
  The umbrella term for a single data stream — one continuous signal (e.g. a
  demodulated calcium channel) or one discrete event stream (e.g. TTL
  timestamps). "store" was chosen over *channel* (which collides with the
  physical *optical* channel — several stores can come from one optical
  channel), over *stream* (which denotes a group of channels in the
  neo/SpikeInterface ecosystem, the wrong granularity), and over *trace*.

store_id
  The source-file identifier for a store, exactly as it appears in the
  acquisition system's files (e.g. `Dv1A` for TDT, a column name for CSV).
  Row 0 of `storesList.csv`.

store_label
  The user-supplied analytic label for a store (e.g. `signal_DMS`,
  `control_DMS`, `reward`). Assigned in Step 1 (Label Stores). Row 1 of
  `storesList.csv`.

session
  One recording — a single mouse, day, and rig session. The atomic input to
  analysis.

session_folder
  The directory on disk that holds one session's raw data files.

run
  One analysis pass over a session, defined by its session, its `storesList`,
  its parameter set, and its output destination. A single session can produce
  many runs.

run_name
  The user-supplied label for a run.

run_folder
  The directory on disk holding one run's outputs, named
  `<session_basename>_output_<run_name>`.
```

## Lists vs maps

A recurring source of confusion is conflating a *collection* of stores with a
*mapping* between store identifiers. GuPPy's naming convention keeps them
distinct:

- **Lists are plural**: `store_ids`, `store_labels`.
- **Maps are directional**, named `<key>_to_<value>` (matching the codebase's
  existing `X_to_Y` dict idiom). The map from source identifiers to user
  labels is `store_id_to_store_label`; dicts keyed by a store's label are
  `store_label_to_<value>` (e.g. `store_label_to_data`).

## Persisted contracts

Some of these terms are baked into GuPPy's on-disk formats. The **format field
names below are intentionally kept** even where the in-memory variable uses the
newer term, so existing outputs remain readable:

| On disk | Holds | In-code term |
|---|---|---|
| `storesList.csv` — row 0 | store identifiers | `store_ids` |
| `storesList.csv` — row 1 | store labels | `store_labels` |
| `<store_id>.hdf5` files | per-store signal/timestamps | one file per `store_id` |
| `storename` dataset inside each `.hdf5` | the store id (provenance) | value is a `store_id` |
| `~/.storesList.json` | cross-session label cache | `store_id_to_store_labels` |
| run folder `<session>_output_<run_name>` | one run's outputs | `run_folder` |

`GuPPyParamtersUsed.json` keys for these entities use the snake_case glossary
terms: `session_folders`, `group_session_folders`, `run_name`,
`run_name_policy`, `selected_runs`, `group_selected_runs`, and the headless
Step-1 mapping `store_id_to_store_label`.

Step 1 of the pipeline — where the user assigns a `store_label` to each
`store_id` — is called **Label Stores** in the GUI.

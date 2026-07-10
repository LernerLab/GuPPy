# Glossary

The canonical terms GuPPy uses for the entities that flow through the pipeline.

```{glossary}
store
  A single data stream — one continuous signal (e.g. a demodulated calcium
  channel) or one discrete event stream (e.g. TTL timestamps).

store_id
  The source-file identifier for a store, exactly as it appears in the
  acquisition system's files (e.g. `Dv1A` for TDT, a column name for CSV).

store_label
  The user-supplied analytic label for a store (e.g. `signal_DMS`,
  `control_DMS`, `reward`), assigned in Step 1 (Label Stores).

session
  One recording — a single mouse, day, and rig session. The atomic input to
  analysis.

session_folder
  The directory on disk that holds one session's raw data files.

run
  One analysis pass over a session, defined by its session, its stores, its
  parameter set, and its output destination. A single session can produce many
  runs.

run_name
  The user-supplied label for a run.

run_folder
  The directory on disk holding one run's outputs.
```

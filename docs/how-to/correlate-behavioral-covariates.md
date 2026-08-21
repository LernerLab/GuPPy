# Correlate a behavioral covariate with the photometry signal

Some experiments score a continuous behavioral variable alongside the recording —
akinesia severity rated every couple of minutes, say, or a manually scored
engagement level. GuPPy can take that variable in as a **behavioral covariate**,
average it over the same time bins it reduces the photometry signal to, and report
how the two relate across the session.

This is an **optional** step. If you only care about signal aligned to discrete
events, use event TTLs instead and skip this guide.

**To try it without your own data**, the repository ships a sample session at
`stubbed_testing_data/csv/sample_data_csv_covariate_1/`. It holds a control and a
signal channel, a TTL channel, and two covariates: `akinesia`, which tracks the
signal, and `grooming`, which does not. Follow this guide against that folder with a
**Bin Width (s)** of `50`. The CSVs are stored with [Git LFS](https://git-lfs.com),
so pull them before use:

```bash
git lfs pull --include="stubbed_testing_data/csv/sample_data_csv_covariate_1/*"
```

## 1. Prepare the CSV

A covariate is an ordinary GuPPy CSV: three columns, one row per score.

```text
timestamps,data,sampling_rate
0.0,2,0.00833
120.0,3,0.00833
240.0,3,0.00833
360.0,4,0.00833
```

- **`timestamps`** — when the score was made, in **seconds**. See step 2.
- **`data`** — the score itself. Any numeric value; the scale is yours.
- **`sampling_rate`** — how often you scored, in Hz. Scoring every 120 seconds is
  `1/120`, or `0.00833`. GuPPy requires the column because it is part of the CSV
  format, but nothing in the covariate analysis reads it, so an approximate value
  is fine.

**One variable per file.** To bring in two variables, write two CSVs and label each
one separately.

You do not need to score on a regular schedule, and your scoring interval does not
need to match the bin width — GuPPy averages whatever scores fall inside each bin.

## 2. Choose your timestamps

Covariate timestamps must be on **the same clock GuPPy reports for your recording**.
This is the single easiest thing to get wrong, and getting it wrong produces either
an error or a silently misaligned result.

- **TDT** — seconds from the start of the recording. A session that begins at `0`
  and runs an hour takes timestamps in `0`–`3600`. This is usually what you would
  write anyway.
- **CSV, Doric and Neurophotometrics** — the acquisition clock exactly as your
  system emits it, which often does **not** start at zero. A Neurophotometrics
  session might report its first sample at `24106.9` seconds, in which case your
  first score belongs near `24106.9`, not near `0`.

If you are unsure, run Steps 1–4 once without the covariate and open
`binned_metrics_<site>.csv`. Its `bin_start` column is the clock your timestamps
must match.

## 3. Drop the file into the session folder

Put the CSV alongside your other data files, in the session folder itself. No
configuration is needed — GuPPy finds it automatically, and it appears as a store
the next time you open the Label Stores GUI.

## 4. Label it

Open **Step 1: Label Stores**. Your file appears as a store named after the file
(so `akinesia.csv` shows up as `akinesia`). Set its **Type** to
**behavioral covariate** and type a **Name** for the variable — this is the name
that appears in the outputs and in the plots.

```{image} ../_static/images/covariate_label_stores.png
:alt: The Label Stores GUI for a session with two covariates: the control, signal and TTL stores labeled as usual, then rows for akinesia and grooming whose Type is set to behavioral covariate with the variable name typed alongside
:width: 90%
```

Names cannot contain spaces, and each one must be unique within the session.

## 5. Run Steps 2 to 4

Run Step 2 and Step 3 as usual. In the parameters form, enable
**Compute Binned Metrics?** and set a **Bin Width (s)**, then run Step 4.

The bin width — not your scoring interval — sets the resolution of the result.
Wider bins average more of both series together and give you fewer, steadier
points; narrower bins give you more points, but bins your scoring never reached
will be empty.

Two combinations are rejected, because the covariate timestamps could not be
interpreted against them: **Combine Data?**, which re-times several sessions onto
one synthetic clock, and the `concatenate` artifact-removal method, which
compresses the time axis where artifacts were cut. Use `replace with NaN` instead.

## 6. Read the results

Two files per recording site land in the run folder:

- **`binned_covariates_<site>.csv`** — your scores averaged into the same bins as
  the photometry, with a count of how many scores landed in each.
- **`covariate_correlations_<site>.csv`** — one row per metric–covariate pair, with
  `pearson_r`, `spearman_rho` and `n_bins`.

`pearson_r` measures a straight-line relationship; `spearman_rho` measures whether
the two move together in rank order, and so tolerates a curved relationship.
`n_bins` is the number of bins where both values were present — the sample size
behind the two coefficients.

## 7. View the results

Open **Step 5: Visualize** and select the **Covariates** tab. Pick a recording site,
a metric and a covariate.

```{image} ../_static/images/covariate_correlations.png
:alt: The Covariates tab: Recording site, Metric and Covariate selectors above four stacked panels - the photometry trace, its per-bin metric, the raw akinesia scores, and their per-bin means - then a scatter of the metric against akinesia with a least-squares line through it, titled with the Pearson r and bin count, and the table of every metric-covariate pair
```

Four panels stack above the scatter, sharing one time axis: the photometry trace,
the per-bin metric it was reduced to, your covariate as you scored it, and your
scores averaged into the same bins. Zooming or panning any one of them moves all
four together. This is the view to check first — it is where a covariate that landed
on the wrong timebase, or a bin width that is far coarser or finer than your scoring
cadence, is obvious at a glance, and where you can see whether a high coefficient
comes from the two series tracking each other bin by bin or from both drifting the
same way across the session.

Below them, the scatter puts one point per bin, with the least-squares line through
it — the line is `pearson_r` drawn, so it is exactly as steep as the coefficient in
the title says. The correlations table lists every metric-covariate pair at once,
`spearman_rho` included.

## No p-value, and why

GuPPy deliberately reports no p-value for these correlations, and one you compute
yourself from `pearson_r` and `n_bins` would not be valid either.

The usual significance tests for a correlation assume every sample is independent.
Per-bin photometry values are not: the signal drifts, so each bin resembles the one
before it. A slowly changing behavioral score has the same property. When two such
series are compared, the standard tests report very small p-values for pairs of
signals that have no relationship at all — a well-documented failure with
false-positive rates reaching 100% in simulation (Harris, *Nonsense correlations in
neuroscience*, 2020). The usual repairs, including shuffling the series or
randomizing its phase, do not fix it.

What you can do with the coefficients:

- **Compare them across recording sites within a session.** All the sites share the
  same behavior, so a much stronger relationship at one site is informative.
- **Compare the stacked panels** to tell a bin-by-bin relationship apart from two
  series that merely drift the same way across the session.
- **Treat any claim of significance as needing a comparison across independent
  sessions**, which GuPPy does not yet perform.

Two more things worth keeping in mind. Bin width is a choice, and trying several and
reporting the best-looking one will find a relationship whether or not one exists —
pick a width from your scoring schedule and stay with it. And a coarse scoring
cadence means few bins, so `n_bins` is often small; check it before reading much
into a coefficient.

## Troubleshooting

**"Covariate ... lands in only N bin(s)"** — nearly always a clock mismatch. The
error prints both the span of your timestamps and the span of the session; if they
do not overlap, revisit step 2. It can also mean your bin width is much narrower
than your scoring interval.

**The store does not appear in Label Stores** — check the file is in the session
folder itself rather than a subfolder, and that its header row is exactly
`timestamps,data,sampling_rate`.

**Coefficients read `NaN`** — either the covariate or the metric is constant across
the bins, or fewer than three bins have both values present.

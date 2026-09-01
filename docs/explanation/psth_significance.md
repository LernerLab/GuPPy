# PSTH significance

A [PSTH](psth.md) with an SEM band leaves the reader to judge by eye whether a response is real, and two PSTHs plotted together are worse: overlapping bands are not a test, and how much they overlap depends on how many trials were collected. PSTH significance answers that question for every timepoint in the window.

Two kinds of comparison are computed:

- **Against zero.** Is this event's response different from baseline? Computed for every event automatically.
- **Between two events.** Is the response to event A different from the response to event B? Computed for the event pairs named in the parameter form.

The result is not a single p-value for the window. It is a mask over the time axis marking which stretches are significant, since the useful question is usually *when* a response begins and how long it lasts.

## The bootstrap

At each timepoint, GuPPy computes a confidence interval on the mean across trials — or, for a comparison between two events, on the difference between the two means. Timepoints whose interval excludes zero are candidates for significance.

The interval is built by resampling rather than from a formula:

1. Draw a resample of *n* trials, with replacement, from the *n* real trials.
2. Average them into one synthetic mean waveform.
3. Repeat — 1000 times by default, set by **Bootstrap Resamples**.
4. Read the interval off the spread of those values at each timepoint.

Whole trials are drawn, so the structure within a trial is preserved, and NaN entries left by artifact removal are ignored when averaging. Resampling makes no assumption about the shape of the trial-to-trial distribution, which matters because photometry trial amplitudes are typically skewed and heavy-tailed.

The interval itself is the **bias-corrected and accelerated (BCa)** interval, computed by [`scipy.stats.bootstrap`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) with `method="BCa"`. BCa applies two corrections to the raw spread of resampled means: a bias correction, for when the resampling distribution sits systematically to one side of the observed mean, and an acceleration term, for when its spread changes with the value of the mean. Both matter most at the small trial counts photometry usually has. The two-sided width is set by **Significance Level (alpha)**, so the default `0.05` gives a 95% interval.

![A mean PSTH plotted as a blue line with its 95% bootstrap confidence interval shaded around it and a dashed line at zero. A red bar along the top of the plot spans the stretch where the interval excludes zero, running from shortly before the event to about two seconds after it.](../_static/images/psth_significance_explainer/fig1_bootstrap_interval.svg)

## Minimum duration

The interval is computed independently at each timepoint, so the band is pointwise rather than simultaneous. Across a window of several thousand samples, some timepoints clear zero without anything happening there.

GuPPy keeps a stretch only if it runs longer than **twice the moving-average filter window** set in Step 3. The filter cannot produce features shorter than its own window, so a briefer excursion did not come from the signal. A stretch just under the threshold is discarded even when it is real; a longer window discards more.

![The same kind of plot with two bars along the top. The upper grey bar marks every timepoint whose interval excludes zero, showing scattered isolated marks across the whole window alongside a solid block over the response. The lower red bar shows only the stretches longer than the filter window, which removes every isolated mark and leaves the block over the response.](../_static/images/psth_significance_explainer/fig2_minimum_duration.svg)

This is why the test requires a non-zero filter window: with filtering disabled there is no shortest meaningful duration to compare against, and every isolated timepoint would survive.

## The resampling unit

Photometry data is nested — trials sit inside sessions, and sessions sit inside groups — and which level gets resampled decides what the result is a claim about.

GuPPy resamples whichever unit the folder it runs in holds:

| Where it runs | What a column holds | What is resampled |
|---|---|---|
| A session run folder | one trial | trials |
| A group folder | one session's average | sessions |

That falls out of the file layout: Group Analysis averages each member session down to a single column before the group PSTH is written, so a group's columns are session means and resampling them resamples sessions.

![Five sessions shown as rows. Each row has that session's individual trials as small grey dots scattered around a larger blue diamond marking the session mean, and the session means differ noticeably from one another. An arrow points at one session's row of trials, labeled as the unit a run folder resamples. A vertical blue bracket spans the column of session means, labeled as the unit a group folder resamples.](../_static/images/psth_significance_explainer/fig3_resampling_unit.svg)

The unit has to match the question being asked. Comparing two events within one session is a claim about that session, and its trials are the right unit. A comparison across a group is a claim about the population those sessions came from, where the session is the unit — the trials inside each one have already been averaged away. Trials from the same session are not independent samples of that wider population, so treating them as if they were produces an interval far narrower than the data supports. This is a well-documented failure mode for nested data in neuroscience; see [Aarts et al. (2014)](https://www.nature.com/articles/nn.3648), which found nested designs in 53% of the papers it reviewed, and [Saravanan, Berman and Sober (2020)](https://nbdt.scholasticahq.com/article/13927-application-of-the-hierarchical-bootstrap-to-multi-level-data-in-neuroscience) for the same point applied to bootstrap methods. Averaging within each session before comparing, which is what a group folder does, is equivalent to a multilevel model when every session contributes the same number of trials ([McNabb and Murayama, 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC9559079/)).

## Limitations

**Sessions in a group are treated as independent.** If several come from the same animal they are not, and the interval will be too narrow in the same way. GuPPy has no concept of a subject, so it cannot detect this.

**A significant stretch does not pin down when the response started.** The result says the effect lies somewhere in that stretch, not that it begins at its left edge. Reading a precise onset off a boundary is the most common way this kind of analysis is over-read.

**Small samples give unreliable intervals.** Below three trials or sessions the bootstrap cannot run and the comparison is skipped. Between three and five it runs but is driven by a handful of distinct resamples; GuPPy logs a warning, and the sample count is recorded in the output.

**Some timepoints have no interval.** Trials near the start or end of a recording have their PSTH windows padded, and artifact removal blanks stretches of others. Where fewer than three trials remain at a timepoint, no interval can be computed and that timepoint is reported as not significant. GuPPy warns when this affects a large fraction of a window.

## Outputs

Each comparison is written as its own table under `psth_significance_output/`, giving the time axis, the estimate, both interval bounds, the significance flag, the alpha it was computed at, and the sample count. The bounds are kept alongside the flag so a reader can see how close a non-significant stretch came and how wide the interval is where the sample size is small.

See [Outputs](../reference/outputs.md) for the file layout and [Parameters](../reference/parameters.md) for the settings that control the test.

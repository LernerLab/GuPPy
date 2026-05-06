# Z-score normalization

## Background

Even after dF/F has corrected for motion and bleaching, the resulting traces are still in session-specific units. Two recordings of the same brain region in the same animal on different days can produce dF/F traces with very different typical magnitudes purely because of differences in fiber coupling, LED intensity, or accumulated photobleaching from previous sessions. A transient that reaches 5% dF/F in one session and 2% dF/F in another can represent the same underlying event scaled by these incidental factors. Pooling responses across sessions, animals, or experimental conditions therefore requires a further normalization that strips the session-specific scale away and puts every recording on a common, comparable footing.

Z-score takes a trace and rescales it so that each value is expressed as a number of standard deviations of the recording's noise distribution. The standard formula subtracts the recording's mean from every sample and divides by its standard deviation:

$$
z_{\text{std}} = \frac{\Delta F/F - \mu}{\sigma}
$$

where $\mu$ and $\sigma$ are the mean and standard deviation of the full $\Delta F/F$ trace.

The result is a unit-free trace centered at zero. The unit is *standard deviations of the noise distribution*: a value of `z = 0` means the sample sits at the typical noise level, `z = 2` means two standard deviations above noise, and `z = 3` is conventionally taken as a real event clearly above background. This is the same statistical convention used throughout signal processing, neuroscience, and statistics more generally. Because the unit is defined per-recording (each session's mean and std produce its own z-score), z-score is best thought of not as a transformation of the data but as the *common scale* on which downstream analyses are done. Two sessions with different absolute fluorescence levels become directly comparable once both are expressed in standard-deviations-of-their-own-noise. This is illustrated in the next figure:


![Top row: dF/F traces from two synthetic sessions in their own panels, with a shared y-axis. Session A (blue, left) has large transients in low noise and fills the y-range; session B (orange, right) has smaller transients in moderate noise and sits as a smaller ripple in the same range. The shared y-axis is what makes the absolute-scale difference visible directly. Bottom row: the same two sessions after standard z-scoring, overlaid on a single shared session-relative scale; the two traces have peaks at different times but their event magnitudes are now in the same units and directly comparable.](../_static/images/zscore_explainer/fig1_zscore_cross_session.svg)

Two synthetic recordings with their own event structures and very different absolute scales (one with large transients in low noise, the other with smaller transients in moderate noise) look essentially incomparable in dF/F units. After standard z-scoring, both recordings sit on a shared session-relative scale and event magnitudes become comparable in the same units, even though the events themselves happen at different times.

## Downstream uses of z-score

Z-score appears in two places in the GuPPy pipeline:

- **[PSTH](psth.md) y-axes.** When the PSTH step computes event-aligned trial averages, it operates on the z-scored trace by default. The resulting PSTHs are in z-score units, so that group analysis can pool trials across sessions and animals on a common scale.
- **Transient detection.** GuPPy's transient detector identifies peaks that exceed a threshold expressed in noise units (effectively `z > 3`, with refinements described below). The threshold's meaning depends on z-score being a faithful unit of noise.

Outside GuPPy, z-score is also the standard reporting unit in fiber photometry papers, so converting to it makes downstream comparisons with the published literature straightforward.

## Improvements: estimating the noise more cleanly

The standard formula uses the full recording's mean and standard deviation as estimates of where the noise sits and how broadly it spreads. That estimator can fail when the recording contains things that are not noise, namely real events of interest. The events get included in the mean and std calculations, contaminating them, and the resulting z-score values have a unit that is not actually a faithful measure of noise. GuPPy offers two alternatives, each fixing this contamination in a different way. They are not mutually exclusive: both address the same underlying question (how do we get a clean estimate of the noise distribution?), with different threat models in mind.

### Baseline z-score: a quieter reference window

The natural use case is event-locked PSTH analysis with a structured task. When each trial has a clean pre-stimulus baseline window by experimental design, baseline z-score uses the statistics of that window as the reference distribution. The result is exactly the language most behavioral hypotheses are stated in: deviation from pre-event baseline activity, in pre-event standard deviations. For event-locked workflows where a clean baseline is available, this is more interpretable than the whole-recording version because the unit is anchored to the same baseline period the reader is comparing event responses against.

The formula keeps the same shape as standard z-score; only the reference distribution differs:

$$
z_{\text{baseline}} = \frac{\Delta F/F - \mu_{\text{baseline}}}{\sigma_{\text{baseline}}}
$$

where $\mu_{\text{baseline}}$ and $\sigma_{\text{baseline}}$ are the mean and standard deviation computed only over the user-selected baseline window.

The window boundaries are set as `baselineWindowStart` and `baselineWindowEnd` in the Input Parameters GUI, in seconds. The figure below shows the effect. The dF/F trace has a quiet baseline window (shaded) followed by transients spread across the rest of the recording. Standard z-score divides everything by the full-recording standard deviation, which the transients have inflated. Baseline z-score divides by the baseline-window standard deviation, which contains only noise and is therefore much smaller. Same trace, same shape; the transients tower in baseline z-score units because the denominator is smaller and is a more honest estimate of the noise.

![A dF/F trace (top) where the shaded baseline window contains only noise and the transients fall outside it. Below: the same trace as standard z-score and as baseline z-score, with the y-axis shared between the two panels. The transients reach much higher values in baseline z-score units because its denominator (the baseline-window std) is much smaller than the full-recording std.](../_static/images/zscore_explainer/fig2_baseline_zscore_contrast.svg)

The catch GuPPy cannot enforce is that the chosen window has to genuinely be baseline. If it is contaminated by event activity, anesthesia, equipment settling, or any other atypical epoch, baseline z-score is normalizing against the wrong reference and downstream interpretations drift accordingly.

The same logic applies more strongly when several sessions are pooled into a group analysis. Standard z-score divides each session by *its own* full-recording std, which mixes pre-event and post-event activity together. Two sessions with identical event responses but different overall activity (one busy, one quiet) end up with different denominators, so their PSTHs come out at different heights even though the underlying response is the same. Baseline z-score sidesteps this by normalizing each session against its own pre-event window, which is the same epoch the experimenter is comparing the event response to. The figure below shows two synthetic sessions with identical event-locked responses; session B has additional non-event-locked transients scattered through the recording. In standard z-score the PSTHs disagree on response magnitude, even though the response itself is identical; in baseline z-score the two PSTHs overlap.

![Two PSTH panels side by side. Left: PSTHs in standard z-score for session A (blue, quiet recording) and session B (orange, active recording with extra non-event-locked transients). Session A's curve peaks higher than session B's even though their event responses are identical, because session B's full-recording std is inflated by the background transients. Right: the same two sessions in baseline z-score; the two curves overlap almost exactly, because each session is normalized against its own pre-event window rather than against its full recording.](../_static/images/zscore_explainer/fig5_psth_baseline_vs_standard.svg)

### Modified z-score: a robust noise estimator

Use case: transient detection. The threshold's meaning has to survive the events being detected, which standard z-score's mean and std cannot do because the events themselves contaminate the estimate. Modified z-score swaps to the median in place of the mean and the median absolute deviation ([MAD](https://en.wikipedia.org/wiki/Median_absolute_deviation)) in place of the standard deviation. Both are insensitive to a small number of large outliers in the strict statistical sense.

The formula is:

$$
z_{\text{modified}} = 0.6745 \cdot \frac{\Delta F/F - \mathrm{median}(\Delta F/F)}{\mathrm{MAD}(\Delta F/F)}
$$

The factor $0.6745$ rescales the result so that, for Gaussian data, modified z-score numerically matches standard z-score; for heavy-tailed non-Gaussian data this calibration shifts and the unit interpretation degrades. The median and MAD are robust for a mathematical reason: standard deviation is the square root of an average of *squared* deviations, so a few large outliers contribute disproportionately and inflate the estimate, while the median is a *rank* statistic and adding a few extreme values to a sorted list does not move the middle of the list. The median and MAD therefore stay anchored to typical noise no matter how large the outliers are.

![Three panels stacked vertically with a shared x-axis. Top: a dF/F trace with three small real events plus two large outliers (40% at t = 5 s, 80% at t = 11 s). Middle: standard z-score of the same trace; the outliers contaminate the std estimate, the real events fall into the shaded below-threshold region, and a peak detector would miss them. An arrow marks the first missed real event. Bottom: modified z-score; the MAD denominator is robust to the outliers, the same real events cross the z = 3 threshold at z ≈ 8, and a peak detector correctly identifies them. The outliers themselves go off-screen at z ≫ 30 (annotated arrows) because the MAD does not see them as part of the typical spread.](../_static/images/zscore_explainer/fig3_modified_zscore_robustness.svg)

GuPPy's transient detector itself uses MAD-based statistics with an additional moving-window two-stage filter, documented in the GuPPy paper. For PSTH and group-analysis workflows the standard or baseline variant is more common, because the events of interest are sparse enough that the std is not dramatically contaminated.

## Limitations

The main caveat cuts across all three z-score variants. Z-score is computed per-recording, so it puts each session on its own session-relative scale. This is what makes cross-session comparison work for the shape and relative magnitude of events, but it is also what hides absolute differences in activity. Two recordings can produce visually identical z-scored traces while their absolute dF/F responses differ by a substantial factor.

The figure below makes the failure mode concrete. Animal A and Animal B both show stereotyped responses to the same event, with the same signal-to-noise ratio. Animal B's recording, however, has 3× larger absolute responses and 3× higher noise (perhaps from looser fiber coupling, lower indicator expression, or a different photobleaching trajectory). In dF/F units, the two animals look very different in absolute scale (top row, shared y-axis). After standard z-scoring (bottom row, overlaid), the two traces become visually indistinguishable. A claim like "Animal B has stronger responses than Animal A" is not answerable from the z-scored traces; you would need to go back to dF/F, and even then the cross-animal comparison is only as good as the cross-animal comparability of the dF/F itself.

![Top row: dF/F traces from two synthetic animals in their own panels with a shared y-axis. Animal A (blue, left) has small absolute responses in low noise; Animal B (orange, right) has 3x larger absolute responses but proportionally 3x higher noise, so the SNR is the same. Bottom row: the same two animals after standard z-scoring, overlaid; the two traces are visually indistinguishable because z-score is invariant to the uniform scaling that distinguishes them in dF/F.](../_static/images/zscore_explainer/fig4_cross_session_limitation.svg)

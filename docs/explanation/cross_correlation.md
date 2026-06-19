# Cross-correlation

## Background

In experiments that record from two brain regions or two cell populations simultaneously, a natural follow-up to computing each region's [PSTH](psth.md) is: *do these two signals move together around the event?* And if they do, *does one lead the other?*

The first half is answered by an ordinary correlation between the two traces. The second half is not. An ordinary correlation compares region A at time `t` with region B at the same time `t`, so a consistent 200 ms offset between them simply registers as a weaker correlation, with no information about which region led. Two regions tightly coupled with a 200 ms lag can look much less correlated than they actually are, and you cannot tell from the result whether A preceded B or the reverse.

Cross-correlation recovers the missing direction by reporting how well the two signals align at *every* possible time offset, not just zero. In fiber photometry it is typically used to compare two subregions of the same structure (for example DMS vs DLS in striatum) or two simultaneous sensors (a dopamine and an acetylcholine sensor) and ask which one consistently leads.

Formally, cross-correlation is a similarity-versus-lag curve. Given two time series `x(t)` and `y(t)`, you slide one across the other and at each offset `tau` compute their inner product. The resulting curve, written `R_xy(tau)`, encodes two distinct pieces of information:

- **Where the peak sits.** A peak at `tau = 0` means the two signals covary instantaneously. A peak at `tau = -200 ms` means `x` leads `y` by 200 ms. Reading the lag at which the peak occurs is how you decide whether one region's activity precedes another.
- **How sharp and tall the peak is.** A narrow, tall peak says the alignment is precise and consistent. A broad, low peak says the two signals share slow drift but not fast structure. A flat curve says no systematic relationship survived the integration window.

![Two event-aligned traces and their cross-correlogram. The lower panel's peak sits at the lag at which the upper traces best align.](../_static/images/cross_correlation_explainer/fig1_what_is_cross_correlation.svg)

## Event-locked cross-correlation

The version of this analysis paired with PSTH is event-locked rather than continuous. Instead of cross-correlating the full [z-scored](zscore.md) traces of the two regions, the **per-event PSTH window** of each region is taken, both already aligned to the same event timestamp, and cross-correlated within that window. The question becomes "do these regions co-fluctuate around the behavioural event?" rather than "are these regions coupled in general?". For event-driven analyses the two are different, because long-timescale baseline coupling between two photometry channels is often dominated by shared physiological drift (movement, photobleaching, breathing artefacts) and is usually not the thing being measured. Restricting to the event window suppresses that contribution. The trade-off is that anything happening outside the PSTH window is invisible; long-range or oscillatory coupling beyond the event has to be computed on the continuous z-scored trace separately.

The procedure is:

1. For each event, take the traces from regions A and B over the same event-aligned window.
2. Compute the full cross-correlation of the two windowed traces.
3. Normalise the resulting curve by dividing it by its own peak absolute value, so the maximum becomes `±1`.
4. Repeat across events, producing one normalised correlogram per event along a shared lag axis in seconds.

The output of this procedure is a set of normalised per-event cross-correlograms paired with a shared lag axis. These are the curves that the rest of this page reads, and that group analysis averages across sessions.

## Reading the correlogram

For each combination of event, signal type, and region pair, the analysis produces a correlogram with lag in seconds on the x-axis and normalised correlation on the y-axis. The curve drawn on it is either the across-event average or a single selected event. Everything below describes how to read **that plot**.

### Lag axis

The lag runs from `-W` to `+W` where `W` is the PSTH window length. The sign convention follows `scipy.signal.correlate`: a positive lag means the second region trails the first. So if the pair is ordered `(A, B)` and the peak sits at `+50 ms`, B trails A by 50 ms on that event, and A leads.

![Three scenarios showing how the peak of the cross-correlogram tracks the relative timing of the two traces. Top: signals aligned, peak at zero. Middle: B trails A by 0.3 s, peak at positive lag. Bottom: B leads A by 0.3 s, peak at negative lag.](../_static/images/cross_correlation_explainer/fig2_lag_mapping.svg)

### Peak height

Every per-event correlogram is normalised independently to its own peak, so its maximum is `±1` by construction. Peak height is therefore not comparable across events: two events with completely different underlying coupling magnitudes will both end up at peak = 1 after normalisation. The figure below makes this concrete. Three events with the same lag relationship but very different absolute amplitudes (both regions large, both regions tiny, asymmetric) produce visually indistinguishable normalised correlograms. Per-event peak normalisation discards absolute coupling magnitude, and that information cannot be recovered from the normalised correlograms after the fact.

![Three events whose normalised correlograms (left) are visually indistinguishable, despite the underlying traces (right) having very different amplitudes: both regions large, both regions tiny, and asymmetric (one large, one small). Per-event peak normalisation collapses these scenarios so neither absolute scale nor amplitude asymmetry between the two regions survives in the saved output.](../_static/images/cross_correlation_explainer/fig3_normalisation_discards_amplitude.svg)

### Peak spread (timing consistency)

It helps to be explicit about two related quantities at this point. A **per-event correlogram** is the cross-correlogram computed from one event's pair of event-aligned traces. The **across-event average** is the simple mean across events of those per-event correlograms, lag by lag. In the figures below, per-event correlograms appear as the pale gray curves and the across-event average is the bold dark line.

The width of the across-event average peak is mostly telling you about *timing consistency across events*, not about the shape of any single event's correlogram. If every event has the same lead/lag, the per-event peaks all stack at the same lag and the across-event average peak is narrow and tall. If the lead/lag varies event-by-event, the per-event peaks spread across lag values and the average smears into a broad peak with proportionally lower height. Each per-event correlogram is normalised to `±1` by its own peak, but misaligned peaks average to less than 1.

A separate factor that affects peak width is signal bandwidth: the cross-correlogram inherits the broader of the two signal widths, so cross-correlating a sharp transient in one region with a slow drift in another produces a broad peak even when timing is perfectly locked, because the slow signal cannot resolve fast structure.

![Top row: consistent lag across events produces a narrow, tall average peak. Bottom row: large event-by-event jitter in the lag produces a broad, lower-amplitude average peak even though each individual-event correlogram is itself sharp. Left column: eight overlaid example events. Right column: per-event correlograms (pale) and across-event average (bold).](../_static/images/cross_correlation_explainer/fig5_timing_consistency.svg)

### Across events

The across-event average is a useful summary, but it can hide structure that a per-event view would expose. The most striking case is **bimodality**: when half the events peak at one lag and the other half peak at a different lag, the average can blend the two clusters into a single broad peak that looks indistinguishable from "events with uniformly jittered timing", but reflects neither population of events honestly.

The figure below shows two scenarios whose across-event averages look broadly similar but whose underlying per-event peak distributions are very different. Top row: events with peak lag uniformly jittered across a moderate range. Bottom row: events whose peak lag is either +0.2 s or -0.2 s, with nothing in between. Both averages are broad single-peaked curves; only the per-event peak-lag histograms (right column) reveal that the two scenarios are unrelated.

![Two scenarios viewed two ways, in 80-event simulations. Left column: across-event average correlogram. Right column: histogram of per-event peak lags. Top: uniformly jittered timing. Bottom: bimodal timing with no events at lag zero. The averages look similar; the histograms do not.](../_static/images/cross_correlation_explainer/fig8_across_trials.svg)

The across-event average does not surface bimodality on its own. Distinguishing it requires reading off each per-event peak lag and binning those values across events, which is what the right column shows.

### No isolated peak (no systematic relationship)

Two regions can each have visible event-locked structure on an event-by-event basis without their structure tracking each other. If the timing of A's response and the timing of B's response are independent across events, every per-event correlogram still has a peak (per-event normalisation forces one), but the peak location is different on each event. The across-event average is no longer dominated by any single lag, so it has no isolated peak rising clearly above the baseline. The meaningful signature of "no systematic relationship" is this absence of a localised peak in the average, not a perfectly flat curve.

![Each event-aligned trace has a clearly visible bump in each region but the bump positions are independent across regions, so the per-event correlograms peak at random lags. The across-event average has no isolated peak.](../_static/images/cross_correlation_explainer/fig7_no_systematic_relationship.svg)

## Limitations and alternatives

Two cautions before reading too much into a peak. First, a leading peak does not imply causation: both regions could be driven by a common upstream input, and apparent leads can come from indicator kinetics or filter delays rather than from neural timing, with the additional caveat that the lead/lag you see is conditional on the alignment event and window length you chose. Second, the result is sensitive to the PSTH window itself: too short a window squeezes the lag range so that real lags hit the edge, while too long a window lets the longest-timescale fluctuation in the window dominate the normalisation and broadens the peak in ways that have nothing to do with the timing relationship of interest.

Cross-correlation is also one of several tools for studying how two simultaneously recorded channels relate. A Pearson correlation at zero lag is simpler and gives a single number for instantaneous similarity; it is often used as a quick check before reaching for cross-correlation. Coherence is the frequency-domain analog, useful when the coupling is band-specific (for instance, theta-band coordination across regions). Granger causality tests whether one signal helps predict the other beyond its own past, hinting at directional influence, but in fiber photometry it has well-known confounds from indicator kinetics and slow drift. None of these are computed by GuPPy. Cross-correlation is the standard tool when the question is "does one signal lead the other around this event," and is best read as one piece of evidence in a larger argument rather than a self-contained answer.

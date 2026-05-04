# Isosbestic correction

A bare fluorescence trace is the sum of calcium-dependent GCaMP fluorescence and several confounds: the fiber tip moving relative to tissue, blood vessels dilating and changing scattering, the indicator photobleaching over the session. The standard correction is to record a second channel simultaneously through the same fiber at the indicator's isosbestic wavelength, where calcium changes produce no fluorescence change, and use it to estimate and subtract those confounds. The rest of this page is about how GuPPy implements that correction, the assumptions behind the linear-fit-and-subtract procedure, and how to read the result.

:::{note} Scope
This page covers isosbestic correction for the GCaMP family (GCaMP6, GCaMP7, jGCaMP8) and similar two-state calcium indicators like RCaMP.
:::

## What is an isosbestic point?

Fiber photometry uses GCaMP because its fluorescence tracks calcium concentration: more calcium, more emission. What is less obvious is that *how much* the fluorescence changes depends on the laser wavelength used to excite the indicator. At some wavelengths a calcium event produces a large change, at others a smaller one, and at one specific wavelength, the **isosbestic point**, the response is zero (calcium concentration changes produce no fluorescence change at all):

![Empirical observation: total fluorescence response during a single calcium event at four different excitation wavelengths, plus the underlying ΔF-versus-wavelength curve. Top row of panels: in each column the same calcium event is overlaid as a black curve, with the fluorescence response plotted in the wavelength-specific colour. From left to right: at 470 nm (green) the response is large and positive; at 420 nm (amber) it is smaller and positive; at 405 nm (purple) it is essentially flat, just baseline noise; at 390 nm (red) it is inverted (a calcium-driven population shift here *decreases* total fluorescence). Bottom panel: ΔF plotted continuously as a function of excitation wavelength, with the four sampled wavelengths marked as coloured dots; the curve is negative below the isosbestic, crosses zero at it, and rises to a positive peak at longer wavelengths.](../_static/images/isosbestic_explainer/fig1_response_columns.svg)

The molecular reason this wavelength exists is that GCaMP has two conformational states whose fluorescence contributions cancel at one specific wavelength; the [mechanism section at the end of this page](#why-the-isosbestic-point-exists) walks through this in detail. For practical artifact correction, what matters is the *consequence*: a recording at the isosbestic wavelength is sensitive to motion, photobleaching, and optical-path drift, but invariant to calcium itself. That makes it a same-tissue, same-fiber estimate of the non-calcium contamination, which can be subtracted from the calcium-sensitive trace to recover the calcium signal alone. In practice the recording wavelength is 405 nm (the standard violet LED), which sits a few nm short of the GCaMP isosbestic (~410 nm); the gap is small enough that the cancellation still works.

## The isosbestic channel as a control signal

Almost anything that fluctuates at 405 nm cannot be calcium-driven. It must come from a process that affects all GCaMP proteins in the field of view equally: the fiber moving, the optical path changing, the bulk amount of indicator going down. Those are exactly the artifacts contaminating the 470 nm trace, recorded through the same physical setup, so the 405 nm channel is a same-tissue, same-fiber estimate of the non-neural component of the signal. That asymmetry is what makes the correction possible: the calcium signal lives only at the calcium-sensitive wavelength, while artifacts live at both.

![Three scenarios contrasting what shows up at 405 nm versus 470 nm. Left: a fiber-motion artifact at t = 15 s shows in both channels. Centre: slow photobleaching decays both channels proportionally over 30 s. Right: a calcium event at t = 15 s appears only at 470 nm; the 405 trace stays flat. The empirical content of the isosbestic property is that the first two are visible at 405 nm and the third is not.](../_static/images/isosbestic_explainer/fig4_what_405_captures.svg)

The 405 nm control mirrors the *shape* of the artifacts in the 470 nm signal but not their *magnitude*: each channel has its own absolute scale, set by hardware and indicator differences. Subtracting raw from raw would mix scales rather than cancel the artifact. The fix is a linear rescaling that brings the control onto the signal's scale:

$$
\widehat{C}(t) = m \cdot C(t) + b
$$

where $m$ and $b$ are chosen by ordinary least-squares fit of the smoothed control $C(t)$ to the smoothed signal.

Both traces are first smoothed with a moving-average filter to suppress per-sample noise. Subtracting the fitted control from the smoothed signal then cancels the shared artifact: the residual is the part that the rescaled control cannot explain. Going one step further and dividing that residual by the fitted control converts the result from raw fluorescence units into a fractional change, comparable across recordings with different absolute brightness (which varies with fiber insertion depth, indicator expression, and LED power). Multiplied by 100, this is dF/F: a fractional change expressed in percent.

$$
\mathrm{dF/F}(t) = \frac{S(t) - \widehat{C}(t)}{\widehat{C}(t)} \times 100
$$

where $S(t)$ is the smoothed signal.

A note on notation: this dF/F is different from the ΔF used in the figures above. The figures use ΔF (the raw fluorescence change in arbitrary units) because that is the natural quantity for comparing wavelengths to each other. dF/F is the residual after artifact removal divided by the fitted control: a fractional change expressed in percent. Both are stages of the same trace: ΔF is the change at the indicator, dF/F is the calibrated, motion-corrected reading downstream. From dF/F the pipeline goes on to [z-score normalisation](zscore.md) and event-aligned analyses such as the [PSTH](psth.md).

![The full GuPPy isosbestic correction on a 60 s synthetic trace. Top: the raw 470 nm signal (green) and 405 nm control (purple) are recorded simultaneously and share a slow bleach plus two motion artifacts; calcium events at t = 10, 30, 50 s appear only on the 470 channel. Middle: the linear fit (dashed grey) is overlaid on the signal; rescaling the control onto the signal's baseline is what makes subtraction sensible despite the two channels being on different absolute scales. Bottom: the dF/F residual after subtracting the fit. The shared slow drift and motion bumps cancel; the three calcium events remain.](../_static/images/isosbestic_explainer/fig5_linear_fit_and_correction.svg)

A linear transformation of the control signal is a good approximation because the dominant artifacts (fiber-bending motion, slow photobleaching) attenuate both wavelengths by approximately the same multiplicative factor, so a single linear fit captures most of their contribution. However, consider that the approximation has limits: bleaching tends toward exponential rather than linear, and the relationship between 405 nm and 470 nm artifacts can drift over a long recording (changing brain state, slow temperature changes). The correction is exact only when the artifacts at the two wavelengths really do scale by the same constant throughout.

Not every artifact is removed cleanly. Real recordings contain confounds whose ratio between 405 nm and 470 nm depends on time or context: a head movement that briefly tilts the fiber differently at the two wavelengths, hemodynamic events with strong wavelength dependence, electrical noise that contaminates only one channel. The single global linear fit cannot rescale these correctly, so the corrected trace will still show some residual artifact at those moments, usually visible as transient deflections that look "too symmetric" or that line up with obvious behavioral events such as grooming bouts.

![A wavelength-dependent artifact illustrated on a 20 s synthetic recording. Left: raw traces, with a motion bump that is large on the 470 nm signal (green) and only weakly visible on the 405 nm control (purple). Right: the corrected dF/F. No single rescaling of the 405 trace can match the 470 bump, so a reduced version of the artifact remains in the residual.](../_static/images/isosbestic_explainer/fig6_what_survives.svg)

## The isosbestic channel as a quality diagnostic

The isosbestic trace is worth looking at on its own, not just as input to the correction. An isosbestic channel that is largely flat with slow exponential decay says the recording is well-behaved: the optical path is stable and the only artifact is bleaching. An isosbestic channel with visible step changes, large transients, or rhythmic fluctuations on the timescale of the animal's movement is telling you that motion or hemodynamic artifacts are large in this recording, and that some fraction of those artifacts will leak through into the corrected trace. Looking at the control before trusting the corrected signal is good practice.

![Two example isosbestic traces, both 60 s long. Left: a clean recording showing only smooth exponential bleaching, indicating the optical path was stable. Right: a recording with several motion-artifact bumps and a slow drift, indicating that some fraction of those artifacts will leak through the linear-fit correction into the dF/F.](../_static/images/isosbestic_explainer/fig7_control_diagnostic.svg)

## Synthetic fallback

Sometimes the isosbestic channel is not available, for example in older datasets recorded before dual-LED rigs were standard, or in sessions where the isosbestic LED was disabled or failed. In those cases an artificial control signal can still be constructed by fitting a simple model to the smoothed signal itself and using that fit as if it were a control trace. GuPPy uses a single exponential for this fit. The purpose is to remove the slow bleaching trend: the exponential captures the dominant low-frequency component of most photometry recordings, so the resulting dF/F sits on a flat baseline and is easier to read than the raw signal.

![Same 60 s synthetic recording corrected two ways. Top row: the input signal (green) is identical in both columns; the control used for the fit is either the real isosbestic trace (left, purple) or the synthetic exponential built from the signal itself (right, dashed red). Bottom row: the resulting dF/F. With the real isosbestic control, both the bleach and the motion artifact at t = 30 s cancel and the calcium events stand out cleanly. With the synthetic exponential, the slow bleach is removed but the motion artifact passes through unchanged: the right-column panels shade the motion-artifact window in amber so the preserved feature is easy to see. In the dF/F it is hard to distinguish from a real calcium event.](../_static/images/isosbestic_explainer/fig8_synthetic_fallback.svg)

This has obvious limitations. A model inferred from the signal itself is not an independent measurement, so it can only remove components that match its model shape. Motion artifacts, hemodynamic events, and any non-monotonic confound pass through into the dF/F trace untouched. A trace produced this way is best read as "bleaching-corrected" rather than "artifact-corrected," and movement-locked responses cannot be cleanly distinguished from movement artifacts.

## Why the isosbestic point exists

The previous sections used the isosbestic property as given. This section walks through the molecular mechanism behind it. There are two pieces: (1) why fluorescence reacts to calcium at all, and (2) why the response is zero at one specific wavelength.

A GCaMP molecule does not always emit the same way. Each individual protein sits in one of two conformational states, and the state determines how much fluorescence the protein produces when illuminated:

* **apo** state: the calcium-binding domain has no calcium bound. In this conformation more of the energy from absorbed light is lost as heat rather than re-emitted as fluorescence, so each protein is dim.
* **calcium-bound** state: calcium has bound to the calcium-binding domain and rearranged the protein around its chromophore. The new conformation suppresses the heat-loss pathway, so much more of the absorbed light comes back out as fluorescence.

The critical fact is that the fraction of GCaMP molecules in each state at any instant depends on the local calcium concentration: more calcium binds more molecules, less calcium lets them relax back (panel A in the next figure). This means that at higher calcium concentrations the indicator population is more biased toward the bright bound state, and the tissue emits more fluorescence overall (panel B). When neural activity opens voltage-gated calcium channels, calcium concentration spikes (panel C, black). The rise shifts the binding equilibrium toward the bound state: molecules transition from apo to bound (panel D), the per-molecule emission rises, and the detector reads an increase in fluorescence (panel C, green).


![Calcium drives the apo/bound population mix, and the population mix drives the fluorescence response. Left column (functions of calcium concentration): (A) apo and bound population fractions as a function of calcium (Hill saturation curve); (B) fluorescence as a function of calcium. Right column (dynamics over time): (C) a calcium event (black) overlaid with the resulting fluorescence response (green, rescaled to share the y-axis); (D) the apo (coral) and bound (teal) population fractions tracking the event with a small kinetic delay. The left column shows the equilibrium relationships; the right column shows the same chain unfolding dynamically.](../_static/images/isosbestic_explainer/fig2_calcium_chain.svg)

The fluorescence response that results from this population shift depends on the recording wavelength, because each state has its own emission spectrum (see top left of the next figure). The calcium-bound state emits more strongly overall and absorbs best at longer wavelengths, while the apo state is dimmer and absorbs best at shorter wavelengths. The per-molecule contribution from each state therefore depends on the recording wavelength: above the spectral crossing (for example, 470 nm) bound dominates and a calcium-driven population shift produces a positive fluorescence response; below the crossing (for example, 390 nm) apo dominates and the same population shift produces a negative response (the signal inverts); at the crossing wavelength itself, the apo loss and the bound gain are equal in magnitude and opposite in sign. This crossing wavelength is the isosbestic point.


![The isosbestic point as a spectral-crossing argument. Top: emission spectra of the apo (coral) and bound (teal) states as a function of wavelength, with their crossing point marked as the isosbestic. Bottom: at four sampled wavelengths, the calcium event decomposed into the apo contribution (coral, the loss as the apo population shrinks), the bound contribution (teal, the gain as the bound population grows), and their sum (the wavelength-coloured total). At the column whose wavelength matches the spectral crossing, the apo loss exactly equals the bound gain and the total stays flat.](../_static/images/isosbestic_explainer/fig3_decomposition.svg)

Note that at the isosbestic the chemistry continues (calcium binds, the apo-to-bound conversion proceeds); the two contributions cancel optically, so the total fluorescence does not change. This is what makes the isosbestic recording a calcium-invariant control channel.

The two-state framework described here is shared by the GCaMP family (GCaMP6, GCaMP7, jGCaMP8) and a few related indicators like RCaMP. Many neuromodulator indicators (dLight, GRAB-DA, GRAB-NE) do not have a clean isosbestic point that current photometry hardware can reach; the correction described above does not apply to them in the same form.

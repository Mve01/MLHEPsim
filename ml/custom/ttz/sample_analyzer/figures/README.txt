TTZ Sample Analyzer Figures - Interpretation Guide

This directory stores the diagnostic plots produced by the TTZ sample analyzer.
The plots compare:
- Real MC reference data (blue)
- Generated samples from the trained flow model (red)

The goal is not only to check single-feature agreement, but also to catch
correlation mismatches that can strongly affect derived physics observables.

======================================================================
1) MAIN FILES YOU WILL SEE
======================================================================

1. feature_comparison.png
	Multi-panel 1D comparison for base and derived features with ratio sub-panels.

2. feature_comparison_log.png
	Same content as above, but y-axis in log scale for tail diagnostics.

3. correlation_diagnostics_1d.png
	Focused angular/correlation 1D checks:
	- DeltaEta_ll
	- DeltaPhi_ll
	- DeltaPhi_l,MET

4. correlation_diagnostics_2d.png
	2D density and ratio maps for (DeltaEta_ll, DeltaPhi_ll):
	- Real density
	- Generated density
	- Generated/Real ratio

5. correlation_diagnostics_wmt_2d.png
	2D density and ratio maps for (W_MT, DeltaPhi_l,MET):
	- Real density
	- Generated density
	- Generated/Real ratio

6. correlation_diagnostics_dphi_binned_wmt.png
	Four-panel 1D overlays of DeltaPhi_l,MET in W_MT quantile bins.

7. correlation_diagnostics_dphi_binned_wpt.png
	Four-panel 1D overlays of DeltaPhi_l,MET in W_Pt quantile bins.


======================================================================
2) HOW TO READ feature_comparison(.png/.log)
======================================================================

Each panel has:
- Top: density overlay (blue real, red generated)
- Bottom: ratio = Generated / Real

Interpretation:
- Good: red and blue overlap well; ratio fluctuates around 1.0 with no strong trend.
- Concerning: broad coherent ratio slope, displaced peaks, or missing tails.
- Tail checks: use feature_comparison_log.png to inspect high-value regions.

Important:
- Good 1D agreement does NOT guarantee good correlations.
- Higher-order variables can fail even if most base 1D marginals look excellent.


======================================================================
3) WHY correlation_diagnostics_1d EXISTS
======================================================================

These observables are sensitive to angular structure that controls many derived
kinematic quantities:

- DeltaEta_ll:
  Difference in pseudo-rapidity between the two Z leptons.

- DeltaPhi_ll:
  Wrapped azimuthal separation between the two Z leptons.

- DeltaPhi_l,MET:
  Wrapped azimuthal separation between W lepton and MET.

Typical failure mode this catches:
- Real shows two shoulders away from 0 in DeltaPhi_l,MET,
- Generated collapses to a strong central peak near 0.

That often indicates overproduction of nearly aligned lepton-MET topology.


======================================================================
4) HOW TO READ correlation_diagnostics_2d (DeltaEta_ll vs DeltaPhi_ll)
======================================================================

Panel A (left): real density map.
Panel B (middle): generated density map.
Panel C (right): generated/real ratio map.

Interpretation of ratio panel:
- Ratio near 1.0 (neutral color): good local agreement.
- Ratio > 1 (warm/red): model overproduces this region.
- Ratio < 1 (cool/blue): model underproduces this region.

Common pattern:
- A central red blob near (0, 0) means too many events with both
  DeltaEta_ll and DeltaPhi_ll close to zero.

Practical note:
- Edge bins with low statistics can look noisy; prioritize coherent structures
  in populated regions.


======================================================================
5) HOW TO READ correlation_diagnostics_wmt_2d
======================================================================

This figure localizes the DeltaPhi_l,MET issue against W_MT.

If generated samples show central excess in DeltaPhi_l,MET, this 2D map tells you:
- whether that excess is concentrated at low W_MT,
- spread across all W_MT,
- or strongest in a specific W_MT band.

This is useful when W_MT-related observables look biased despite acceptable
single-feature overlays.


======================================================================
6) HOW TO READ BINNED DeltaPhi_l,MET PLOTS
======================================================================

Files:
- correlation_diagnostics_dphi_binned_wmt.png
- correlation_diagnostics_dphi_binned_wpt.png

Each figure has 4 panels, each panel a quantile bin in the conditioning variable
(W_MT or W_Pt).

Important details about the panel ranges:
- These are quantile bins (about 25% of events per panel), not equal-width GeV bins.
- Bin edges are computed from pooled real+generated values so both are compared in
	exactly the same kinematic ranges.
- Titles like "W_Pt in [a, b]" are the actual numeric range used for that panel.
- First three bins are [a, b), last bin is [a, b] so no events are dropped on the
	upper edge.

Why DeltaPhi is plotted against W_MT/W_Pt:
- The plotted quantity is always DeltaPhi(l, MET); W_MT or W_Pt is only the
	conditioning axis used to split events into regimes.
- This asks: "Does the DeltaPhi shape mismatch happen everywhere, or only for
	certain W kinematics?"

Physical connection:
- W_MT depends directly on DeltaPhi(l, MET):
	W_MT = sqrt(2 * pT_l * MET * (1 - cos(DeltaPhi(l, MET))))
- W_Pt is a vector sum in the transverse plane:
	W_Pt = | pT_l + MET |, which is also sensitive to DeltaPhi(l, MET).

So if DeltaPhi collapses toward 0 in generated samples, it can bias both W_MT and
W_Pt-derived observables. These binned plots show where that bias is strongest.

Use them to answer:
- Is mismatch global or only in specific kinematic regimes?
- Does the central DeltaPhi peak appear only at low W_MT/W_Pt?
- Do shoulders recover at high W_MT/W_Pt?

This helps target model improvements and prevents chasing global fixes for a
regime-local problem.


======================================================================
7) WEIGHTING AND COMPARISON SEMANTICS
======================================================================

When available, real MC can be weighted to a selected component (sm, linear,
quadratic, full, and signed variants). Generated samples are compared against
that weighted target distribution.

Therefore, judge agreement against the displayed real reference in each plot,
not against intuition from unweighted SM shapes.


======================================================================
8) QUICK TRIAGE CHECKLIST
======================================================================

If higher-order variables look wrong:

1) Check correlation_diagnostics_1d.png first.
	- Is DeltaPhi_l,MET collapsing to zero?

2) Check correlation_diagnostics_2d.png.
	- Is there a central ratio excess in (DeltaEta_ll, DeltaPhi_ll)?

3) Check correlation_diagnostics_wmt_2d.png.
	- Is DeltaPhi_l,MET mismatch tied to W_MT bands?

4) Check binned DeltaPhi_l,MET plots.
	- Identify whether mismatch is low-, mid-, or high-kinematics dominant.

5) Confirm feature ordering and preprocessing consistency for the run.
	- Ordering issues can mimic physics mismatch.


======================================================================
9) OUTPUT NAMING NOTES
======================================================================

Base output path set by analyzer is expanded into related files by suffix:
- _log.png
- _1d.png
- _2d.png
- _wmt_2d.png
- _dphi_binned_wmt.png
- _dphi_binned_wpt.png

Temporary or smoke-test files may start with _tmp_ and can be removed safely.


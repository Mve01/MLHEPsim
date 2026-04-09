SMEFT Reweighting Figures - Legend Semantics

This directory contains per-c_Ht comparison figures produced by
ml/custom/ttz/smeft_reweighting.py.

Important update (April 2026):
- The blue baseline curve (SM ROOT) is eventWeight-weighted.
- It is no longer an unweighted ROOT count baseline.

Per-panel curves in reweighted_c*.png:
1) Blue: SM ROOT eventWeight
   - Histogrammed with ROOT eventWeight as weights.
   - Normalized as density via sum(eventWeight) * bin_width.

2) Red: Generated c_Ht
   - Flow-generated sample reweighted to the chosen c_Ht.

3) Green: ROOT c_Ht
   - ROOT decomposition baseline: sm + c*linear + c^2*quadratic,
     normalized to unit total weight.

Ratio panel:
- Red ratio: Generated(c_Ht) / ROOT(SM eventWeight)
- Green ratio: ROOT(c_Ht) / ROOT(SM eventWeight)

File naming:
- reweighted_cplusXpY.png      : linear y-scale
- reweighted_cplusXpY_log.png  : log y-scale
- neff_vs_c.png                : effective sample size diagnostic

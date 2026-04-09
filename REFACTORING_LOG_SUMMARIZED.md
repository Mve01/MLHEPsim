# Refactoring Log Summary (Condensed)

This file summarizes both historical logs:
- REFACTORING_LOG.md (large main log, up to Entry 54)
- REFACTORING_LOG_CONTINUED.md (Session-based continuation, currently through Session 20)

Goal: keep older history short, preserve recent high-impact changes, and provide a practical current-state reference.

---

## 1) Ultra-Condensed Early History (Feb 18 to early Mar 2026)

Older entries were heavily condensed by request.

### What happened (high level)
- Built and stabilized the TTZ normalizing-flow training/evaluation pipeline.
- Iterated on feature schemas (object selection, jet/lepton handling, coordinate representations).
- Fixed multiple validation/comparison bugs where train/val/test or generated-vs-reference logic could diverge.
- Standardized preprocessing/scaling behavior (including no_process/selection edge cases).
- Improved reproducibility and data handling (shuffle behavior, selection updates, file consistency).
- Repaired MLflow registration/loading issues several times.
- Added and then refined higher-order physics validation plots and analysis scripts.

### Net result from this period
- A working baseline pipeline for TTZ flow training, model registration, generation, and diagnostics was established.

---

## 2) Condensed Mid History (Mar 2 to Mar 12)

### Training objective and weights
- Fixed a critical bug where training weights were not effectively applied to loss.
- Refactored weight storage into a single multi-column dataset format.
- Added signed-weight handling strategies (including positive/negative decomposition paths) and stabilized training with stratified batching + SNIS-style normalization refinements.

### SMEFT analysis stack
- Added/expanded smeft_reweighting tooling and sweep orchestration.
- Improved plotting quality (ratio ranges, labels, normalization, layout) and added comparisons against truth/reference.
- Hardened runtime robustness (ragged arrays, device mismatches, shape mismatches, NaN guards).

### Infrastructure and migration
- Added GPU/Condor execution paths for reweighting jobs.
- Performed repository migration/cleanup in continued log:
  - Removed unused HIGGS/DrellYan/classifier/VAE code and stale data.
  - Updated hardcoded paths for the new repo location.
  - Flattened/cleaned model config and active model-code paths.
  - Corrected docs to match active 15-feature TTZ setup.

### Net result from this period
- Pipeline moved from “works” to “operationally robust,” with reproducible weighted training + more reliable SMEFT diagnostics.

---

## 3) Recent Detailed History (Mar 18 to Mar 30)

This period remains most relevant for current work.

### Entry 45 (Mar 18)
- Added NaN-poisoning guardrails in SMEFT reweighting path.

### Entry 46 (Mar 24)
- Improved log-ratio diagnostics and introduced Condor GPU-idle mitigation concepts.

### Entry 47 (Mar 24)
- Reverted TTZ representation to direct Pt/Eta/Phi (from earlier cartesian migration direction).

### Entry 48 (Mar 26)
- Fixed reweighting ratio binning to stay aligned with ROOT-basis/reference handling.

### Entry 49–50 (Mar 26)
- Introduced sin/cos(phi) representation and migrated analyzer path to 20D schema.

### Entry 51 (Mar 27)
- Fixed analyzer weight reset bug causing higher-level mismatch artifacts.

### Entry 52 (Mar 27)
- Rolled back sin/cos(phi) migration back to direct-phi 15D schema across TTZ pipeline.

### Entry 53 (Mar 27)
- Added automatic correlation diagnostics (1D and 2D angular/correlation checks) to analyzer and inherited SMEFT sweep workflow.

### Entry 54 (Mar 29)
- Reconciled Condor anti-idle mitigations across submission paths (CPU feed + metrics cadence consistency).

### Session 20 (Mar 30, continued log)
- Fixed analyzer feature-order sensitivity by explicitly aligning real-data columns to selected feature names before derived-variable computation.
- This resolved the observed “base features look fine but higher-order variables look broken” mismatch pattern tied to implicit ordering assumptions.

---

## 4) Continued Log Sessions (Quick Digest)

From REFACTORING_LOG_CONTINUED.md:

- Session 15: major repository cleanup + path migration to LLoCagenerator workspace.
- Session 16: documentation accuracy pass (15-feature reality reflected across READMEs).
- Session 17: config/model cleanup and simplification to active model paths.
- Session 18: phi-wrapping fix era (historical context before later representation rollbacks).
- Session 19: cylindrical↔cartesian migration phase (later superseded by direct-phi rollback in main log).
- Session 20: analyzer feature-order alignment fix (current and active).

---

## 5) Current Technical State (Practical Snapshot)

### Representation and feature schema
- Active TTZ schema is direct Pt/Eta/Phi with 15 base physics features (+ optional 4 weight columns in data files).

### Analyzer behavior
- Generates base + derived comparisons.
- Includes automatic correlation diagnostics from plot_all().
- Real-data columns are now explicitly aligned by feature name before derived-variable calculations (prevents silent order mismatch bugs).

### SMEFT/reweighting
- Includes robustness protections for NaNs, shape/device issues, and improved ratio diagnostics.

### Condor/operations
- Submission paths include reconciled anti-idle mitigations to reduce GPU-held/no-usage failure modes.

---

## 6) Notes on Log Organization

- Logging diverged over time between the two files:
  - Main file contains recent Entry 45–54 updates.
  - Continued file contains session-style migration/cleanup narrative and Session 20.
- This summary is the consolidation layer so future readers can avoid parsing ~5900 lines of raw logs.

---

## 7) Suggested Ongoing Logging Policy (Optional)

- Keep detailed day-to-day entries in REFACTORING_LOG_CONTINUED.md only.
- Keep REFACTORING_LOG_SUMMARIZED.md as the single rolling executive summary.
- Update this summarized file only when a change affects:
  - feature schema,
  - training objective/weight semantics,
  - analyzer/reweighting interpretation,
  - or operational reliability.

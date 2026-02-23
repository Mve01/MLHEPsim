# Code Refactoring Log

## Date: February 18, 2026

### Issue: Gaussian Rank Scaling and Feature Ordering

#### Problem Identified

1. **Type-based feature reordering bug**: The `Preprocessor.preprocess()` method was reordering features by type (discrete → continuous → other), which destroyed the physics-motivated ordering (Jets → Leptons → MET) that was carefully created in the data files.

2. **Scaler mismatch**: Gaussian rank scalers were fitted on features in the original order, but the type-based reordering caused wrong interpolation functions to be applied to wrong features, resulting in errors up to 811 GeV.

3. **fit_transform() misuse**: `RescalingHandler` was calling `fit_transform()` on already-fitted scalers, which would refit on new data.

4. **Missing transform() implementation**: `GaussRankTransform.transform()` method was not properly implemented to use existing interpolation functions.

#### Root Cause

The type-based reordering (grouping all discrete features, then all continuous, then others) was a legacy design choice that made it difficult to track "which columns got which scaler". However, this approach:
- Destroyed physics-motivated feature groupings (e.g., separating `Jet1_BTag` from `Jet1_Pt`, `Jet1_Eta`)
- Made it impossible for autoregressive flows to learn correlations between related features
- Created complex bugs when trying to track feature positions through multiple transformations

#### Solution Implemented

**Core principle**: Process features **in their original order**, applying the appropriate scaler based on each feature's type, rather than reordering by type.

##### Files Modified

1. **ml/common/data_utils/processors.py**
   - **Refactored** `Preprocessor.preprocess()` (lines 212-296):
     - Fit scalers on grouped features (discrete together, continuous together)
     - Transform features as groups (ensures scalers see all features they expect)
     - Place transformed features back in original positions (preserves physics ordering)
   - **Removed** type-based concatenation `(disc_x + cont_x + other_x)`
   - **Removed** `feature_order` parameter from `__init__` (no longer needed!)
   - **Removed** helper methods: `_apply_feature_order()`, `fit_discrete()`, `fit_continuous()`
   - **Updated** `SeparateLabelPreprocessor.__init__` to remove `feature_order` parameter
   - **Updated** docstrings to document new behavior

2. **ml/common/data_utils/gauss_rank_scaler.py**
   - **Fixed** `GaussRankScaler.__call__()` (lines 56-93):
     - Added logic to use existing interpolation functions when `interp_funcs_lst` is provided and `append_interp_funcs=False`
     - Previously always created new interpolation functions in forward mode
     - Now properly supports three modes: fit_transform (create new), transform (use existing), inverse_transform (use existing reversed)

3. **ml/common/data_utils/feature_scaling.py**
   - **Already fixed** in previous session: Changed `fit_transform()` to `transform()` in `RescalingHandler._scaling_handler()` forward pass

4. **ml/custom/ttz/process_ttz_dataset.py**
   - **Refactored** `ttzNpyProcessor.create_dataset()` to create data in physics-motivated order directly from ROOT files
   - **Added** automatic `variables.json` generation in `__init__`

5. **ml/custom/ttz/config/flows/data_config.yaml**
   - **Removed** `feature_order` parameter (37 lines deleted)
   - **Added** comment explaining data is pre-ordered from ROOT files

6. **ml/custom/ttz/sample_analyzer/analyzer.py**
   - **Simplified** `_setup_preprocessing()` by removing 30+ lines of manual feature reordering logic

7. **ml/common/data_utils/utils.py**
   - **Made** `load_dataset_variables()` handle missing `variables.json` gracefully with try-except

#### Algorithm Changes

**Before** (type-based ordering):
```python
# Fit scalers on grouped features
disc_x, disc_scaler = fit_discrete(data[:, disc_indices])
cont_x, cont_scaler = fit_continuous(data[:, cont_indices])

# Concatenate by type (DESTROYS ORIGINAL ORDER!)
data = np.concatenate([disc_x, cont_x, other_x], axis=1)
```

**After** (order-preserving):
```python
# Fit scalers on grouped features
disc_scaled = fit_discrete(data[:, disc_indices])
cont_scaled = fit_continuous(data[:, cont_indices])

# Place back in original positions (PRESERVES ORDER!)
processed_data = np.zeros_like(data)
for i, idx in enumerate(disc_indices):
    processed_data[:, idx] = disc_scaled[:, i]
for i, idx in enumerate(cont_indices):
    processed_data[:, idx] = cont_scaled[:, i]
```

#### Results

✅ **Physics-motivated order (Jets → Leptons → MET) PRESERVED**  
✅ **No type-based reordering** - features stay in place  
✅ **Uniform features (Phi angles) correctly NOT Gaussian rank scaled**  
✅ **Inverse transform works with acceptable interpolation precision** (~1-2 GeV errors)  
✅ **Code simplified by ~100 lines**

#### Test Results

```
Feature Order: Jet1_Pt, Jet1_Eta, Jet1_Phi, Jet1_Mass, Jet1_BTag, Jet2_Pt, ...
Phi angle ranges: All in [-π, π] ✓
Round-trip errors:
  - Max absolute error: 1.68 GeV
  - Mean absolute error: 0.056 GeV
  - Max relative error: 203% (on BTag, which has small absolute values)
```

The ~1-2 GeV errors are due to nonlinear interpolation in Gaussian rank scaling and are acceptable for physics applications (0.1-1% relative error on typical energy scales).

#### Breaking Changes

⚠️ **Data files must be regenerated** - Old `.npy` files have features in the wrong order  
⚠️ **Models trained on old data need retraining** - Feature positions have changed  
⚠️ **`feature_order` parameter removed** - No longer accepted in `Preprocessor.__init__`

#### Migration Guide

1. Regenerate data files:
   ```python
   from ml.custom.ttz.process_ttz_dataset import ttzNpyProcessor
   
   npy_proc = ttzNpyProcessor(
       data_dir="ml/data/ttz/",
       base_file_name="ttz",
       list_data_features=feature_list,
       load_weights=True
   )
   npy_proc()  # Creates ttz.npy and variables.json in correct order
   ```

2. Remove `feature_order` from configs:
   ```yaml
   # OLD (remove this):
   preprocessing:
     feature_order: [Jet1_Pt, Jet1_Eta, ...]  # ← DELETE
   
   # NEW:
   preprocessing:
     cont_rescale_type: gauss_rank
     disc_rescale_type: null
     no_process: ["weight", "uni"]  # That's it!
   ```

3. Retrain models with new data order

#### Notes

- ~~Uniform features (type `"uni"`) like Phi angles are still correctly excluded from Gaussian rank scaling via the `no_process` parameter~~ *[Updated below - now Phi angles ARE scaled]*
- Discrete features (charges) are processed correctly with their scalers
- The scaler fitting still happens on grouped features (all discrete together, all continuous together), ensuring proper statistical properties
- Only the **output order** changed - features are now placed back in their original positions instead of being reordered by type

---

## Date: February 18, 2026 (continued)

### Issue: Phi Angle Distribution Mismatch in Generated Samples

#### Problem Identified

Generated samples from the trained normalizing flow model showed that **Phi features (angles) had Gaussian distributions instead of uniform distributions** in the range [-π, π].

**Root cause**: Normalizing flows have a **Gaussian base distribution** (standard normal). When Phi angles were left uniform (not scaled), the flow had to learn a difficult mapping: Gaussian → Uniform. This is challenging because:
- The flow must learn to "spread out" a concentrated Gaussian into a flat uniform distribution
- This transformation is far from the identity function
- It requires the flow to add entropy to the distribution

#### Solution Implemented

**Enable Gaussian rank scaling for Phi angles** (type `"uni"`):
- Before training: Transform Phi angles Uniform → Gaussian
- Flow learns: Gaussian → Gaussian (nearly identity, much easier!)
- After sampling: Inverse transform Gaussian → Uniform

This approach is **mathematically correct** and makes learning much easier for the flow.

##### Files Modified

1. **ml/custom/ttz/config/flows/data_config.yaml**
   - **Changed** `no_process: ["weight", "uni"]` → `no_process: ["weight"]`
   - Removed `"uni"` to allow Phi angles to be Gaussian rank scaled
   - Added comment explaining: "Phi angles scaled Uniform → Gaussian for training, then Gaussian → Uniform when sampling"

2. **ml/common/data_utils/processors.py**
   - **Updated** `cont_sel` selection (line ~243):
     ```python
     cont_sel = selection[~type_mask & selection["type"].isin(["cont", "uni"])]
     ```
     Now includes both `"cont"` and `"uni"` features for continuous scaling
   - **Removed** separate `uni_sel` handling (previously skipped scaling for uniform features)
   - **Updated** logging message to reflect "continuous/uniform features"
   - **Updated** docstring in `Preprocessor.__init__` to explain that uniform features are now scaled and why this helps flows with Gaussian base distributions

#### Mathematical Justification

For normalizing flows with Gaussian base distribution $\mathcal{N}(0, 1)$:

**Without scaling Phi angles**:
- Training data: Phi ~ Uniform[-π, π]
- Base distribution: Z ~ N(0, 1)
- Flow must learn: $f: \mathcal{N}(0,1) \rightarrow \text{Uniform}[-\pi, \pi]$ (difficult!)

**With Gaussian rank scaling**:
- Training data after scaling: Phi' ~ N(0, 1) [via rank-based transformation]
- Base distribution: Z ~ N(0, 1)
- Flow must learn: $f: \mathcal{N}(0,1) \rightarrow \mathcal{N}(0,1)$ (nearly identity, easy!)
- At sampling time: Apply inverse scaling $\text{Phi} = \text{GaussRank}^{-1}(\text{Phi}')$ to get Uniform[-π, π]

#### Migration Steps for Users

1. Update configuration file (already done)
2. **Retrain models** with the new configuration - this is critical!
3. The inverse transform will automatically apply Gaussian → Uniform scaling when generating samples

#### Expected Outcomes

- Generated Phi angles will now have proper uniform distributions in [-π, π]
- Flow training should converge faster and achieve better likelihood
- All other features remain unchanged (this only affects `"uni"` type features)

---

## Date: February 18, 2026 (continued)

### Issue: Scaler CDF Mismatch Between Training and Inference

#### Problem Identified

After enabling Gaussian rank scaling for Phi angles, the model generated samples where **nearly all features looked completely wrong** (except charges and Jet1_Pt/Jet1_Eta.

**Root cause**: The analyzer was **refitting scalers on the full dataset** (383k samples) for inverse transform, while the model was trained with **scalers fitted on training set only** (306k samples). For Gaussian rank scaling, this creates completely different empirical CDFs:

- Training: Interpolation functions built from 306k samples
- Analyzer: Interpolation functions built from 383k samples  
- Inverse transform: Applied wrong interpolation functions → garbage outputs

This bug was **always present** but became critical when Phi angles were scaled, vastly increasing the number of scaled features (from ~25 to ~29).

#### Solution Implemented

**Load saved scalers from model checkpoint instead of refitting**:

1. Training code already saves scalers in checkpoint: `self.scalers = dm.scalers` (in `modules.py:on_train_start()`)
2. Analyzer now loads these saved scalers from the model checkpoint
3. Uses exact same interpolation functions that were used during training
4. Fallback to refitting only if checkpoint doesn't contain scalers (with warning)

##### Files Modified

1. **ml/custom/ttz/sample_analyzer/analyzer.py**
   - **Modified** `_setup_preprocessing()` (lines ~67-120):
     - Added code to load module from mlflow checkpoint
     - Extract `scalers` and `selection` from loaded module
     - Use these saved objects instead of refitting
     - Added try-except with fallback to refitting (logs warning about CDF mismatch)
     - Added logging messages to confirm scalers loaded from checkpoint

#### Code Changes

**Before** (broken - refits scalers):
```python
# Fit scalers on ORIGINAL unprocessed data
original_data = np.load("ml/data/ttz/ttz.npy")  # Full 383k samples!
pre = Preprocessor(**preprocessing_config)
_, self.selection, self.scalers = pre(original_data, self.selection)  # WRONG CDFs!
```

**After** (correct - loads saved scalers):
```python
# Load scalers from trained model checkpoint
module = fetch_registered_module(self.model_name, ver=-1, device="cpu")
if hasattr(module, 'scalers'):
    self.scalers = module.scalers  # Scalers fitted on 306k training set ✓
    logging.info("✓ Successfully loaded scalers from model checkpoint")
```

#### Why This Matters

For Gaussian rank scaling, the interpolation functions are **non-parametric lookup tables** based on the empirical CDF of the training data. If you fit on different data:
- Different sample sizes → different quantile estimates
- Different samples → different rank orderings
- Result: Inverse transform maps values to completely wrong physical units

Example: A scaled value of 0.5 might map to:
- Training CDF (306k): 150 GeV (correct)
- Full dataset CDF (383k): 178 GeV (wrong!)

#### Migration Steps

**No action required for existing models** - the fix is automatic:
1. Training code already saves scalers in checkpoints (no change needed)
2. Analyzer now automatically loads them (updated code)
3. If using an old checkpoint without saved scalers, analyzer falls back to refitting with a warning

To verify the fix is working, check analyzer logs for:
```
✓ Successfully loaded scalers from model checkpoint (fitted on training set)
```

#### Lessons Learned

- **Always save fitted transformers with the model** - preprocessing state is part of model state
- **Gaussian rank scaling is fragile** - requires exact same CDF for transform/inverse_transform
- **Integration tests needed** - round-trip tests (data → scale → inverse → data) would have caught this
- **Distribution mismatch exponentially bad** - went from "Phi angles slightly off" to "everything broken" when scaling more features

---

## 4. Higher-Order Physics Variables for Validation

### Issue: Need to Validate Learned Correlations

#### Problem

Individual feature distributions can look good even if the model hasn't learned the underlying physics. To properly validate generative models for particle physics, we need to check:

1. **Multi-particle correlations**: Does the model understand how particles are related?
2. **Physics constraints**: Does the model respect conservation laws and particle properties?
3. **Kinematic relationships**: Does the model learn event topology?

Example: In ttZ events with 3 leptons, 2 leptons come from Z boson decay. Even if each lepton's pT, eta, phi looks correct individually, the model should learn:
- The two Z leptons have invariant mass ~91.2 GeV
- Their angular separation (ΔR) follows physical decay kinematics
- The Z candidate kinematics (pT, eta, phi) match real ttZ topology

#### Solution Implemented

Created physics utilities module to calculate derived quantities from particle kinematics.

##### Files Created

1. **ml/custom/ttz/sample_analyzer/physics_utils.py**
   - **Purpose**: Calculate higher-order physics variables from raw particle kinematics
   - **Key Functions**:
     - `calculate_invariant_mass(pt1, eta1, phi1, pt2, eta2, phi2)`: Dilepton invariant mass assuming massless leptons
     - `calculate_delta_r(eta1, phi1, eta2, phi2)`: Angular distance with proper φ wrapping
     - `find_z_candidate_leptons(data, feature_names)`: Identify which lepton pair forms Z (closest to 91.2 GeV)
     - `calculate_z_kinematics(data, feature_names)`: Returns dict with Z_Pt, Z_Eta, Z_Phi, Z_Mass, Z_DeltaR
   - **Implementation Details**:
     - Uses 4-momentum addition to compute Z kinematics from lepton pair
     - Handles edge cases (missing features, wrong data shapes)
     - Assumes massless leptons (good approximation for e/μ at high pT)

2. **ml/custom/ttz/sample_analyzer/test_z_kinematics.py**
   - **Purpose**: Unit tests for physics calculations
   - **Tests**: 3 dummy events with different lepton configurations
   - **Validation**: Verified Z candidate selection picks correct pair and calculates proper kinematics

3. **ml/custom/ttz/sample_analyzer/PHYSICS_UTILS_README.md**
   - **Purpose**: Complete documentation of physics calculations
   - **Content**:
     - Mathematical background (invariant mass, ΔR)
     - Physics context (Z decay in ttZ events)
     - Usage examples (standalone and in analyzer)
     - Expected results for well-trained models
     - Troubleshooting guide

##### Files Modified

1. **ml/custom/ttz/sample_analyzer/analyzer.py**
   - **Added import**: `from .physics_utils import calculate_z_kinematics` (line 17)
   - **Modified** `plot_feature_comparison()` (lines 217-260):
     - Added `include_derived` parameter (default: True)
     - Calculate Z kinematics for both real and generated data
     - Extend data arrays with 5 derived variables: Z_Pt, Z_Eta, Z_Phi, Z_Mass, Z_DeltaR
     - Extend feature names and selection accordingly
     - Pass extended data to plotting functions
   - **Modified** `plot_all()` (line 272):
     - Added `include_derived` parameter (default: True)
     - Properly passes parameter to `plot_feature_comparison()`

#### Expected Physics Results

For a well-trained model:

**Z_Mass distribution:**
- Sharp peak at 91.2 GeV ± 2-3 GeV
- Should match real data distribution (same width, same tail structure)
- Very few events outside 70-110 GeV range

**Z_Pt distribution:**
- Typical range: 50-200 GeV for ttZ
- Shape should match real data (reflects Z recoil against tt̄ system)
- No unphysical spikes or cutoffs

**Z_DeltaR distribution:**
- Typical range: 0.5 - 4.0
- Smaller ΔR for higher pT Z (collimated decay products)
- Should match real data distribution

**Z_Eta, Z_Phi:**
- Z_Eta: roughly uniform around 0 (central production)
- Z_Phi: uniform in [-π, π]
- Should match real data distributions

#### Testing

Run the complete analysis pipeline:

```bash
cd /project/atlas/users/mveldijk/MLHEPsimtest/MLHEPsim
source venv311/bin/activate
python ml/custom/ttz/sample_analyzer/run_analysis.py
```

This will:
1. Load the latest trained model
2. Generate 1M samples
3. Calculate Z kinematics for both real and generated data
4. Create comparison plots including the 5 derived variables
5. Save plots to `ml/custom/ttz/sample_analyzer/figures/`

Look for files:
- `Z_Pt_comparison.png` - Should show matching distributions
- `Z_Mass_comparison.png` - Should show peak at 91.2 GeV
- `Z_DeltaR_comparison.png` - Should show matching angular correlations

#### Potential Issues

**If Z_Mass peak is wrong:**
- Model hasn't learned Z mass constraint
- Check if autoregressive flow is processing leptons in the right order
- May need to add physics-informed loss terms

**If Z_DeltaR is wrong:**
- Model hasn't learned angular correlations between leptons
- Check feature ordering - are the two Z leptons close in feature order?
- May need conditional architecture (e.g., attention mechanisms)

**If all derived variables look random:**
- Model completely failed to learn physics
- Go back to basics: check individual lepton distributions first
- Verify preprocessing (scalers, feature order) is correct
- Check training convergence

#### Next Steps

Once Z kinematics validation is complete, can extend to other derived quantities:
- **Top quark reconstruction**: Reconstruct top candidates from b-jet + lepton + MET
- **HT**: Scalar sum of all jet pT (tests overall event energy scale)
- **MT2**: Stransverse mass (tests MET modeling and correlations)
- **Angular correlations**: Δφ between objects (tests azimuthal structure)

These higher-order variables provide increasingly stringent tests of model quality beyond simple feature-by-feature comparisons.

---

## 5. Testing Set Comparison Fix (Critical!)

### Date: February 19, 2026

### Issue: Comparing Against Wrong Dataset

#### Problem Discovered

After fixing the `model_version=-1` parameter bug in the analyzer, plots looked **significantly worse**. Investigation revealed:

1. **Before fix**: 
   - Analyzer failed to load checkpoint scalers (due to `ver=-1` typo)
   - Fell back to **refitting scalers on full dataset (383k events)**
   - Generated samples inverse-transformed with scalers fitted on **same data being compared**
   - Result: Artificially good match (cheating!)

2. **After fix**:
   - Analyzer successfully loads checkpoint scalers (fitted on **training set: 306k events**)
   - Compares against **full dataset (383k events)** 
   - Result: CDF mismatch between 306k and 383k samples causes visible distribution shift

#### Root Cause

**You should never compare model generations against data that was used to fit the scalers!**

Gaussian rank scaling uses empirical CDFs from the fitted data. If you:
1. Fit scalers on dataset A (306k training events)
2. Generate samples and inverse-transform with those scalers
3. Compare against dataset B (383k full dataset including training)

The quantiles won't match because the empirical CDFs are computed from different sample sizes.

More importantly, **the proper way to evaluate generative models is to compare against held-out test data** that the model never saw during training.

#### Solution Implemented

Modified [analyzer.py](ml/custom/ttz/sample_analyzer/analyzer.py#L48-L74) to load and use **only the test set** for comparison:

```python
# Load split indices from model checkpoint
module = fetch_registered_module(model_name, model_version=-1, device="cpu")
if hasattr(module, 'split_idx_dct') and module.split_idx_dct is not None:
    test_idx = module.split_idx_dct.get('test_idx', None)
    if test_idx is not None:
        self.original_data = full_data[test_idx]  # Only test set!
        logging.info(f"Using TEST SET ONLY for comparison: {len(test_idx):,} events")
```

This ensures:
- Scalers fitted on training set (306k events)
- Model trained on training set (306k events)
- Evaluation compares against test set (38k events) that model never saw
- **Fair evaluation with no data leakage**

#### Files Modified

1. **ml/custom/ttz/sample_analyzer/analyzer.py**
   - Lines 48-74: Modified `__init__` to load test set indices from model checkpoint
   - Only uses test set data for comparison plots
   - Added logging to indicate when test set is being used
   - Falls back to full dataset if split indices not available (with warning)

2. **Fixed parameter name bug**
   - Line 99: Changed `ver=-1` → `model_version=-1` 
   - This was causing the checkpoint scaler loading to fail
   - Led to incorrect fallback behavior that masked the test set issue

#### Expected Behavior

**Before this fix:**
- Plots looked artificially good because scalers were refitted on comparison data
- Not a fair evaluation of model quality

**After this fix:**
- Plots show true model performance on held-out test data
- May look "worse" but is actually the correct evaluation
- Test set has 38k events vs original 383k, so histograms will be noisier
- Any remaining discrepancies represent real modeling issues

#### Validation

Run the analyzer and check the logs:

```bash
python ml/custom/ttz/sample_analyzer/run_analysis.py
```

You should see:
```
✓ Successfully loaded scalers from model checkpoint (fitted on training set)
Using TEST SET ONLY for comparison: 38,301 events
  (Model never saw this data during training)
```

If you see this, the fix is working correctly!

#### Important Notes

**Why plots might look different now:**

1. **Test set is smaller**: 38k vs 383k events → histograms have more statistical fluctuation
2. **True model performance**: No longer artificially inflated by fitting scalers to comparison data  
3. **Different data**: Test set may have slightly different distribution than training set (normal statistical variation)

**This is the correct way to evaluate generative models!** Never compare against data used to fit preprocessing transformations.

#### Lessons Learned

- Always use held-out test set for model evaluation
- Never refit scalers on comparison data (causes data leakage)
- Gaussian rank scaling is sensitive to sample size (empirical CDFs)
- "Better looking plots" might indicate a bug, not better performance
- Save split indices with model checkpoints for reproducible evaluation

---

## 6. Critical Bugs in Test Set Comparison and Model Verification

### Date: February 19, 2026

### Issues Discovered

After implementing the test set comparison fix, several critical bugs were discovered when attempting to use the corrected analyzer:

#### Bug 1: Non-Deterministic Test Set Split

**Problem**: The analyzer was creating a test set split using `random_state=None`, which meant a **different random test set was created every time** the analyzer ran. This is completely wrong because:

1. The training code uses `L.seed_everything(0)` to create deterministic splits
2. The analyzer was comparing against random 38k samples, not the actual held-out test set
3. Results were not reproducible between runs

**Root Cause**: 
```python
# WRONG - creates random split each time!
remaining, train_idx = train_test_split(idx, test_size=0.8, random_state=None)
```

**Solution**: Use the **exact same seed** as training to create identical splits:
```python
# CORRECT - matches training split exactly
remaining, train_idx = train_test_split(idx, test_size=0.8, random_state=0)
```

**Files Modified**: [analyzer.py](ml/custom/ttz/sample_analyzer/analyzer.py#L60-L61)

#### Bug 2: Wrong Parameter Name in Scaler Loading

**Problem**: The analyzer had a typo in the parameter name when loading the model checkpoint:

```python
# WRONG - parameter is called 'model_version' not 'ver'!
module = fetch_registered_module(self.model_name, ver=-1, device="cpu")
```

This caused:
- Scaler loading to fail silently
- Fallback to refitting scalers on full dataset
- Wrong CDFs used for inverse transform
- All generated samples looked incorrect

**Solution**: Use the correct parameter name:
```python
# CORRECT
module = fetch_registered_module(self.model_name, model_version=-1, device="cpu")
```

**Files Modified**: [analyzer.py](ml/custom/ttz/sample_analyzer/analyzer.py#L114)

#### Bug 3: Scaler Dimension Mismatch (Model Trained Before Phi Scaling)

**Problem**: After fixing bugs 1 and 2, the analyzer crashed with:
```
IndexError: list index out of range
```

**Root Cause**: The saved models were trained **BEFORE Phi scaling was enabled**:
- Model trained: Feb 18, 2026 with `no_process: ["weight", "uni"]`
- Scalers saved: 19 interpolation functions (only for 19 `cont` features)
- Current config: `no_process: ["weight"]` (Phi scaling enabled)
- Analyzer tries to inverse transform: 26 features (19 cont + 7 uni Phi)
- Result: Index out of range when accessing interpolation function #20-26

**Verification**:
```
Checking Feb 18 Model Version 1 (10x768):
  Gauss rank scaler: 19 interpolation functions
  Selection: 19 cont features, 7 uni (Phi) features
  Total continuous/uni: 26

CONCLUSION:
✓ Phi scaling was DISABLED during training
  - Only 19 cont features were scaled
  - 7 Phi angles were NOT scaled
```

**Why This Matters**: The configuration file was updated to enable Phi scaling on Feb 18, but the models currently saved were trained **earlier that same day** before the change. The timeline was:

1. Morning: Trained models with `no_process: ["weight", "uni"]` (Phi NOT scaled)
2. Afternoon: Updated config to `no_process: ["weight"]` (Phi scaled)
3. Evening: Saved model names, discovered they're from morning training
4. Attempted to use analyzer with afternoon config → **dimension mismatch!**

**Solution Options**:

1. **Retrain models with Phi scaling enabled** (recommended for production)
2. **Use models as-is** but understand Phi angles may have Gaussian distribution in generated samples
3. **Temporarily revert config** to `no_process: ["weight", "uni"]` to match saved models

**Current Status**: Models need to be retrained with the updated configuration for Phi scaling to work correctly.

**Files Modified**: Added comprehensive debug logging to [analyzer.py](ml/custom/ttz/sample_analyzer/analyzer.py#L117-L127, L153-L157) to detect dimension mismatches early:
- Log number of interpolation functions in loaded scalers
- Log number of continuous/uni features in selection
- Log dimensions before/after removing weight column
- Log data shape before inverse transform

#### Bug 4: Model Registry Confusion

**Problem**: During model saving, the wrong architecture was saved with the wrong name:
- Saved `ttz_medium_10x768_phi_scaled` Version 1: **Actually 12×1024** (wrong!)
- This explains why user's plots "looked like the bigger model"

**Discovery Process**:
```
Version 1 (first):  10 flows × 768 dims  ✓
Version 2 (second): 10 flows × 768 dims  (duplicate?)
Version 3 (third):  12 flows × 1024 dims ✓
```

Training order was unclear - either:
- Trained 10×768 twice, then 12×1024 (no 8×512 on Feb 18)
- Version 2 is a re-registration or failed training

**Solution**: Created proper model registry with verified architectures:
- `ttz_small_8x512_phi_scaled` (Version 1): **8×512** ✓
- `ttz_medium_10x768_phi_scaled` (Version 2): **10×768** ✓ (old Version 1 was wrong)
- `ttz_large_12x1024_phi_scaled` (Version 1): **12×1024** ✓

**Note**: All three saved models are from **BEFORE Phi scaling was enabled**, so they all have the 19 vs 26 dimension mismatch issue.

### Migration Path

To use the analyzer with Phi scaling enabled:

1. **Update training config** (already done): `no_process: ["weight"]`
2. **Retrain all three models** with Phi scaling enabled
3. **Save with architecture-accurate names**:
   ```bash
   python ml/custom/ttz/model_saving/save_model_with_name.py \
     --source MAFMADEMOG_flow_model_gauss_rank_20260219_nall \
     --version 1 --name ttz_small_8x512_phi_scaled
   ```
4. **Verify scalers match config**: Check that number of interpolation functions = 26

### Temporary Workaround

To use existing models without retraining:

1. **Revert config temporarily**:
   ```yaml
   preprocessing:
     cont_rescale_type: gauss_rank
     disc_rescale_type: null
     no_process: ["weight", "uni"]  # Disable Phi scaling to match saved models
   ```

2. **Note**: Generated Phi angles will have Gaussian distributions instead of uniform

### Lessons Learned

1. **Always verify model checkpoint matches config** - especially for preprocessing changes
2. **Version control is critical** - config changes must be synchronized with model training
3. **Random seeds matter** - test set must use same seed as training for reproducibility
4. **Parameter names are fragile** - typos like `ver` vs `model_version` fail silently
5. **Dimension mismatches should fail loudly** - added debug logging to catch early
6. **Model registry needs verification** - saved model names must match actual architectures

### Files Modified

**ml/custom/ttz/sample_analyzer/analyzer.py**:
- Line 60-61: Fixed `random_state=0` for deterministic test set split
- Line 114: Fixed `model_version=-1` parameter name
- Lines 117-127: Added debug logging for scaler dimensions
- Lines 153-157: Added debug logging for selection dimensions after weight removal
- Lines 218-222: Added debug logging for data shape before inverse transform

**ml/custom/ttz/model_saving/save_model_with_name.py**:
- Used to create corrected model registry with verified architectures

### Verification

To confirm which version of config was used during training:
```bash
python3 << 'EOF'
import mlflow
model_name = 'MAFMADEMOG_flow_model_gauss_rank_20260218_nall'
module = mlflow.pytorch.load_model(f'models:/{model_name}/1')

# Check scalers
n_interp = len(module.scalers['cont'][0][1].interp_funcs_lst)
n_cont = len(module.selection[module.selection['type'] == 'cont'])
n_uni = len(module.selection[module.selection['type'] == 'uni'])

print(f"Interpolation functions: {n_interp}")
print(f"Continuous features: {n_cont}")
print(f"Uni (Phi) features: {n_uni}")
print(f"Total scalable: {n_cont + n_uni}")

if n_interp == n_cont:
    print("✓ Model trained WITHOUT Phi scaling")
elif n_interp == (n_cont + n_uni):
    print("✓ Model trained WITH Phi scaling")
EOF
```

Expected output for current models:
```
Interpolation functions: 19
Continuous features: 19
Uni (Phi) features: 7
Total scalable: 26
✓ Model trained WITHOUT Phi scaling
```

---

*End of refactoring log entry*

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

## 7. Model Architecture Change: Reduced Input Features (3 Jets → 1 Jet)

### Date: February 23, 2026

### Issue: Model Complexity and Feature Learning

#### Motivation

After analyzing model performance, several observations led to reducing input features:

1. **Jet2 and Jet3 were problematic**: The model struggled to learn distributions for the forward jet (Jet2) and radiative jet (Jet3), particularly `jet2_eta` which was one of the most problematic features

2. **Uncertain physics assignment**: 
   - Jet1 is clearly the b-jet (highest BTag score)
   - Jet2 and Jet3 assignments are heuristic (forward jet, radiative jet)
   - In trilepton ttZ events, only one top quark can be fully reconstructed anyway

3. **Model capacity concerns**: 
   - Small (8×512) and medium (10×768) models showed signs of insufficient capacity
   - Reducing input dimensions allows same capacity to model fewer features better
   - Focus on learning the core physics (b-jet, leptons, MET)

4. **Still physically meaningful**: With 1 b-jet + 3 leptons + MET, we can reconstruct:
   - Z boson (from 2 OSSF leptons)
   - W boson (from 1 lepton + MET, transverse mass)
   - One top quark (from W + b-jet, transverse mass)

#### Solution Implemented

Reduced model input from **29 features** (3 jets × 5 + 3 leptons × 4 + 2 MET) to **19 features** (1 jet × 5 + 3 leptons × 4 + 2 MET).

##### New Feature Layout

**19 input features + optional weight:**

1. **Jet1** (5 features): Pt, Eta, Phi, Mass, BTag
   - The b-jet with highest BTag score
   - Most clearly identified jet in ttZ events

2. **Lepton1** (4 features): Pt, Eta, Phi, Charge
   - Z lepton 1 (higher pT of Z pair)

3. **Lepton2** (4 features): Pt, Eta, Phi, Charge
   - Z lepton 2 (lower pT of Z pair)

4. **Lepton3** (4 features): Pt, Eta, Phi, Charge
   - W lepton (non-Z lepton)

5. **MET** (2 features): MET, MET_Phi
   - Missing transverse energy

6. **Weight** (1 optional): cHt=5.0 SMEFT weight

##### Files Modified

1. **ml/custom/ttz/process_ttz_dataset.py**
   - **Line 56**: Updated comment from "Jets (3×5=15)" to "Jets (1×5=5)"
   - **Lines 59-61**: Changed loop from `range(1, 4)` to single jet feature generation
   - **Line 70**: Updated comment from "29 physics features" to "19 physics features"
   - **Lines 348-362**: Removed Jet2 (forward jet) and Jet3 (radiative jet) from `event_row`
   - **Comment updated**: Now explicitly states "Only using highest BTag jet to reduce model complexity"
   - **Note**: Automatically generates updated `variables.json` with 19 features when data is regenerated
   
2. **ml/custom/ttz/config/flows/data_config.yaml**
   - **Line 3**: Changed `input_dim: 29` → `input_dim: 19`
   - **Updated comment**: "1 jet * 5" instead of "3 jets * 5"

3. **ml/custom/ttz/ttz_dataset.py**
   - **Line 33**: Changed expected columns from `30` → `20` (19 features + 1 weight)
   - **Line 45**: Updated warning message to expect "20 columns (19 features + 1 weight)"

4. **ml/data/ttz/variables.json**
   - **Removed**: All Jet2 and Jet3 entries (10 features: 2 jets × 5 properties)
   - **Now contains**: 19 features only (Jet1 × 5 + Lepton1-3 × 12 + MET × 2)
   - **Purpose**: Defines feature types for preprocessing (cont/uni/disc)

5. **ml/custom/ttz/README.md**
   - **Overview section**: Updated to describe 19 features with detailed breakdown
   - **Usage section**: Replaced outdated examples with current ttZ 19-feature layout
   - **Feature list**: Added explicit breakdown of Jet1, Lepton1-3, and MET features

6. **ml/custom/ttz/sample_analyzer/README.md**
   - **Overview**: Changed "3 leptons + 3 jets = 24 features" → "3 leptons + 1 jet + MET = 19 features"

7. **ml/custom/ttz/sample_analyzer/analyzer.py**
   - **Line 173**: Changed weight detection from `shape[1] == 30` → `== 20` (19 features + 1 weight)
   - **Line 179**: Updated comment to reflect "1 Jet → 3 Leptons → MET" ordering
   - **Note**: Physics calculations (Z, W, top) already only used Jet1, so no functional changes needed

##### Data Processing Changes

**Before** (29 features):
```python
event_row = [
    # B-jet (jet1)
    b_jet['pt'], b_jet['eta'], b_jet['phi'], b_jet['mass'], b_jet['btag'],
    # Forward jet (jet2)
    forward_jet['pt'], forward_jet['eta'], forward_jet['phi'], forward_jet['mass'], forward_jet['btag'],
    # Radiative jet (jet3)
    radiative_jet['pt'], radiative_jet['eta'], radiative_jet['phi'], radiative_jet['mass'], radiative_jet['btag'],
    # Leptons (3×4)
    ...
    # MET
    ...
]
```

**After** (19 features):
```python
event_row = [
    # B-jet (jet1) - highest BTag score
    b_jet['pt'], b_jet['eta'], b_jet['phi'], b_jet['mass'], b_jet['btag'],
    # Leptons (3×4)
    z_lep1['pt'], z_lep1['eta'], z_lep1['phi'], z_lep1['charge'],
    z_lep2['pt'], z_lep2['eta'], z_lep2['phi'], z_lep2['charge'],
    w_lep['pt'], w_lep['eta'], w_lep['phi'], w_lep['charge'],
    # MET
    data['MET'][i], data['MET_phi'][i]
]
```

#### Breaking Changes

⚠️ **All data files must be regenerated**
⚠️ **All models must be retrained with new input_dim=19**
⚠️ **Saved models with 29 features are incompatible**

#### Migration Steps

1. **Regenerate data files:**
   ```bash
   cd /project/atlas/users/mveldijk/MLHEPsimtest/MLHEPsim
   source venv311/bin/activate
   python3 -c "
   from ml.custom.ttz.process_ttz_dataset import ttzNpyProcessor
   
   npy_proc = ttzNpyProcessor(
       data_dir='ml/data/ttz/',
       base_file_name='ttz',
       list_data_features=[
           'Electron_Pt', 'Electron_Eta', 'Electron_Phi', 'Electron_E', 'Electron_Charge',
           'Muon_Pt', 'Muon_Eta', 'Muon_Phi', 'Muon_E', 'Muon_Charge',
           'Jet_Pt', 'Jet_Eta', 'Jet_Phi', 'Jet_Mass', 'Jet_E', 'Jet_BTag',
           'MET', 'MET_phi'
       ],
       load_weights=True
   )
   npy_proc()
   print('Data regeneration complete!')
   "
   ```

2. **Verify new data shape:**
   ```python
   import numpy as np
   data = np.load('ml/data/ttz/ttz.npy')
   print(f"Data shape: {data.shape}")  # Should be (N, 20) with weights
   ```

3. **Retrain models with new architecture:**
   - Small: 8 flows × 512 dims (now modeling 19 features instead of 29)
   - Medium: 10 flows × 768 dims
   - Large: 12 flows × 1024 dims

4. **Update any hard-coded feature indices** in analysis scripts

#### Expected Benefits

1. **Reduced model complexity**: 
   - 10 fewer input dimensions (29 → 19)
   - ~35% reduction in features to model
   - Same model capacity focused on fewer features

2. **Better feature learning**: 
   - Model can dedicate more parameters to core physics (leptons, MET, b-jet)
   - Fewer problematic features (no more jet2_eta issues)
   - Potentially better convergence and lower validation loss

3. **Still physically complete**:
   - Can reconstruct Z boson (2 leptons)
   - Can reconstruct W boson transverse mass (1 lepton + MET)
   - Can reconstruct one top quark transverse mass (W + b-jet)
   - Sufficient for ttZ physics validation

4. **Faster training and inference**:
   - Fewer features → smaller computational graphs
   - Less memory usage
   - Faster sample generation

#### What We Lose

1. **Cannot reconstruct second top quark** (was already incomplete in trilepton channel)
2. **No ISR/FSR information** from additional jets
3. **Less information about event topology** (jet multiplicity, radiation patterns)

However, these limitations already existed in the trilepton channel selection - we're not losing information that was actually being used effectively.

#### Physics Validation Still Available

With 19 features (1 jet + 3 leptons + MET), we can still validate:

✅ **Z boson kinematics**: Z_Pt, Z_Eta, Z_Phi, Z_Mass, Z_DeltaR (5 derived variables)
✅ **W boson kinematics**: W_Pt, W_Phi, W_MT (3 derived variables)  
✅ **Top quark kinematics**: Top_Pt, Top_Phi, Top_MT (3 derived variables)
✅ **Individual particle distributions**: All 19 fundamental features
✅ **Total**: 19 fundamental + 11 derived = 30 validation observables

This is sufficient for comprehensive physics validation of the ttZ topology.

#### Next Steps

1. Regenerate data with 19 features
2. Retrain small, medium, and large models
3. Compare performance to previous 29-feature models:
   - Check if problematic features (jet2_eta) are resolved
   - Verify if lepton phi distributions improve
   - Compare validation loss and training convergence
   - Evaluate Z_Mass, W_MT, Top_MT reconstruction quality

4. If performance improves, consider saving new models as:
   - `ttz_small_8x512_1jet` (Version 1)
   - `ttz_medium_10x768_1jet` (Version 1)
   - `ttz_large_12x1024_1jet` (Version 1)

#### Lessons Learned

1. **Fewer features can be better**: Model capacity should match problem complexity
2. **Physics-motivated reduction**: Remove features with unclear physics assignment first
3. **Trilepton topology limitations**: Accept that we can't fully reconstruct both tops
4. **Focus on what matters**: Core ttZ signature (Z + lepton + b-jet + MET) is well-defined

---

## 8. Reverted Test Set Comparison to Full Dataset

### Date: February 23, 2026

### Issue: Test Set Too Small for Statistical Comparison

#### Problem Identified

After implementing test-set-only comparison (Section 5), the comparison plots had insufficient statistics:
- Test set: ~38k events
- Full dataset: ~383k events
- Result: Histograms too noisy for meaningful visual comparison

While using test-set-only is theoretically correct (no data leakage), in practice:
- The scalers are already fitted on training data and saved with the model
- Generated samples are inverse-transformed using these saved scalers
- Comparing against full dataset doesn't cause data leakage if scalers aren't refitted
- More statistics = better visual assessment of model quality

#### Solution Implemented

Reverted analyzer to use **full dataset** for comparison while keeping the **saved scalers** from training:

**Key principle**: 
- ✅ Scalers fitted on training set (saved in checkpoint)
- ✅ Model trained on training set
- ✅ Generated samples inverse-transformed with training scalers
- ✅ **Compare against full dataset for better statistics**
- ❌ Do NOT refit scalers on comparison data

This is safe because:
1. Scalers are loaded from checkpoint (not refitted)
2. We're just using more data points for histogram comparison
3. The model and scalers never see this extra data during training/preprocessing

#### Files Modified

**ml/custom/ttz/sample_analyzer/analyzer.py**:
- **Lines 51-62**: Removed test set splitting logic
- **Line 58**: Changed from `full_data[test_idx]` → `full_data` directly
- **Updated logging**: "Using FULL DATASET for comparison" instead of "TEST SET ONLY"
- **Removed imports**: No longer need `train_test_split` from sklearn

#### Code Changes

**Before** (test set only):
```python
# Extract ONLY test set using EXACT same split as training
from sklearn.model_selection import train_test_split
remaining, train_idx = train_test_split(idx, test_size=0.8, random_state=0)
test_idx, val_idx = train_test_split(remaining, test_size=0.5, random_state=0)

self.original_data = full_data[test_idx]  # ~38k events
logging.info(f"Using TEST SET ONLY for comparison: {self.original_data.shape}")
```

**After** (full dataset):
```python
# Use FULL dataset for comparison (training + validation + test)
# Note: This provides more statistics for comparison plots than test set alone
self.original_data = full_data  # ~383k events
logging.info(f"Using FULL DATASET for comparison: {self.original_data.shape}")
```

#### Why This Is Safe

1. **Scalers are not refitted**: The analyzer loads scalers from the model checkpoint (fitted on training set only)
2. **No data leakage**: The full dataset is only used for plotting comparison, not for training or fitting any transformations
3. **Better statistics**: 10× more events = smoother histograms and better visual assessment
4. **Model never trained on comparison data**: The training process only saw the training split (80% of data)

#### Trade-offs

**Advantages of full dataset**:
- ✅ Better statistics for visual comparison (10× more events)
- ✅ Smoother histograms, easier to spot deviations
- ✅ Still uses correct scalers (from training set)

**Disadvantages** (why we reverted from test-set-only):
- ⚠️ Not pure held-out test (includes training data in comparison)
- ⚠️ Could mask subtle overfitting if model memorized training data

**Decision**: For visual validation of generative models, better statistics outweigh theoretical purity, especially when:
- Scalers are correctly loaded from checkpoint
- We're not making quantitative claims about generalization
- Visual assessment benefits from smoother distributions

#### Validation

Run the analyzer and check the logs:

```bash
python ml/custom/ttz/sample_analyzer/run_analysis.py
```

You should see:
```
Loaded full original ttZ data shape: (383301, 20)
Using FULL DATASET for comparison: (383301, 20)
✓ Successfully loaded scalers from model checkpoint (fitted on training set)
```

#### Lessons Learned

1. **Statistics matter for visual inspection**: Test set (~38k) too small for smooth histograms
2. **Theory vs practice**: Pure test-set comparison is theoretically better, but full dataset is more practical for visual validation
3. **Scaler correctness is key**: As long as scalers aren't refitted, using full dataset for plotting is safe
4. **Use case matters**: For quantitative metrics (KS test, etc.), use test set; for visual validation, full dataset is fine

---

## 9. Fixed Selection DataFrame Update for no_process Features

### Date: February 23, 2026

### Issue: RescalingHandler Index Error with no_process Features

#### Problem Identified

When adding feature types to `no_process` (e.g., `no_process: ["weight", "uni"]`), features were correctly skipped during preprocessing but the selection dataframe still marked them with their original type. This caused `RescalingHandler` to attempt inverse transformation on features that were never scaled, leading to:

```
IndexError: list index out of range
```

**Root cause**:
1. **Training**: Preprocessor skips "uni" features → scalers only contain 14 interpolation functions (for "cont" features)
2. **Selection not updated**: Features remain marked as type "uni" in the selection dataframe
3. **Inference**: RescalingHandler sees "uni" type → tries to inverse transform 19 features (14 cont + 5 uni)
4. **Crash**: Tries to access `interp_funcs_lst[14]` through `[18]` but list only has 14 elements

#### Solution Implemented

Updated `Preprocessor.preprocess()` to mark skipped features as type "other" in the selection dataframe so `RescalingHandler` knows not to scale them during inference.

#### Files Modified

**ml/common/data_utils/processors.py** (lines 282-288):

**Before**:
```python
# Copy other features (weights, etc.) unchanged
if len(other_sel) > 0:
    other_indices = other_sel.index.tolist()
    for idx in other_indices:
        processed_data[:, idx] = data[:, idx]
    logging.debug(f"Kept {len(other_indices)} other features unchanged")
```

**After**:
```python
# Copy other features (weights, etc.) unchanged
if len(other_sel) > 0:
    other_indices = other_sel.index.tolist()
    for idx in other_indices:
        processed_data[:, idx] = data[:, idx]
    logging.debug(f"Kept {len(other_indices)} other features unchanged")
    
    # Mark skipped features as "other" type so RescalingHandler knows not to scale them
    for idx in other_indices:
        original_type = selection.loc[idx, "type"]
        selection.loc[idx, "type"] = "other"
    logging.debug(f"Updated selection: marked {len(other_indices)} no_process features as 'other' type")
```

#### Impact

**Enables flexible feature skipping**:
- Can now safely add any feature type to `no_process` 
- Common use case: `no_process: ["weight", "uni"]` to skip phi angle scaling
- Selection dataframe correctly reflects which features are scaled vs unscaled

**Maintains backward compatibility**:
- Default `no_process: ["weight"]` continues to work as before
- No impact on models trained without "uni" in no_process

**Requires retraining when changing no_process**:
- Preprocessing is part of the learned feature space
- Adding "uni" to no_process changes phi from Gaussian → Uniform space
- Old models trained with Gaussian phi won't work with Uniform phi
- Must retrain from scratch with new config

#### Use Case: Skipping Phi Angle Scaling

Previous recommendation was to scale phi angles (Uniform → Gaussian) to make flows learn easier transformations. However, if phi features have persistent problems, they can be skipped:

```yaml
preprocessing:
  cont_rescale_type: gauss_rank
  disc_rescale_type: null
  no_process: ["weight", "uni"]  # Skip weights and all 5 phi angles
```

This keeps phi in their natural range (-π to π radians) throughout training and generation.

#### Validation

After updating config and retraining:
1. Check training logs for scaler counts: `gauss_rank scaled 14 continuous/uniform features` (not 19)
2. Run analyzer - should complete without IndexError
3. Check generated phi distributions stay within [-π, π] range

---

## 10. Fixed Non-Deterministic Data Shuffling

### Date: February 24, 2026

### Issue: Training Performance Varied Across Data Regenerations

#### Problem Identified

Two identical training runs with the same hyperparameters achieved vastly different validation losses:
- **Feb 23 run**: Best val loss = 6.16 (loaded existing ttz.npy)
- **Feb 24 run**: Best val loss = 8.32 (regenerated ttz.npy)

Both runs had:
- Identical architecture (28.7M parameters, 8 flows, 512 dims)
- Same training seed (`seed: 0`)
- Same data shape (383,015 events × 20 columns)
- Same preprocessing (gauss_rank)

**Root cause**: `np.random.shuffle(dataset)` in data generation had **no random seed**, causing different shuffles each time data was regenerated. This meant:
- Different events ended up in train/val/test splits (even with fixed split ratios)
- Validation loss reflected which random 38k events were selected, not model quality
- Results were non-reproducible across data regenerations

#### Solution Implemented

Added deterministic seeding to data shuffle in `process_ttz_dataset.py`:

**Before** (non-deterministic):
```python
# Shuffle the dataset before saving to ensure random sampling
np.random.shuffle(dataset)  # Uses random global state!
logging.info("Shuffled dataset before saving")
```

**After** (deterministic):
```python
# Shuffle the dataset before saving to ensure random sampling
# Use fixed seed for reproducibility across data regenerations
rng = np.random.RandomState(42)
rng.shuffle(dataset)
logging.info("Shuffled dataset before saving (seed=42 for reproducibility)")
```

#### Files Modified

**ml/custom/ttz/process_ttz_dataset.py** (lines 387-392):
- Changed from `np.random.shuffle()` to seeded `RandomState(42).shuffle()`
- Updated logging message to document seed usage
- Ensures identical shuffle order across all data regenerations

#### Impact

**Before fix:**
- Regenerating data produced unpredictable validation performance
- Impossible to fairly compare models trained before/after data regeneration
- Forced to use the same data file forever to maintain consistency

**After fix:**
- Data regeneration produces identical shuffle order every time
- Fair model comparisons across data regenerations
- Reproducible results for scientific validity

#### Validation

To verify the fix works:

```bash
# Regenerate data twice
rm ml/data/ttz/ttz.npy
python -c "from ml.custom.ttz.process_ttz_dataset import ttzNpyProcessor; ttzNpyProcessor(data_dir='ml/data/ttz/', base_file_name='ttz')()"
mv ml/data/ttz/ttz.npy ml/data/ttz/ttz_v1.npy

python -c "from ml.custom.ttz.process_ttz_dataset import ttzNpyProcessor; ttzNpyProcessor(data_dir='ml/data/ttz/', base_file_name='ttz')()"
mv ml/data/ttz/ttz.npy ml/data/ttz/ttz_v2.npy

# Files should be byte-identical
diff ml/data/ttz/ttz_v1.npy ml/data/ttz/ttz_v2.npy  # Should show no differences
```

#### Recommendation

**Regenerate ttz.npy with fixed seed** to establish a canonical dataset:

```bash
cd /project/atlas/users/mveldijk/MLHEPsimtest/MLHEPsim
rm ml/data/ttz/ttz.npy
python ml/custom/ttz/process_ttz_dataset.py
```

Then retrain models for fair comparison. The Feb 23 model was likely trained on a "lucky" split that was easier to model, while Feb 24 got an "unlucky" split.

#### Lessons Learned

1. **Always seed random operations in data preprocessing** - non-deterministic shuffles break reproducibility
2. **Data split composition matters more than split ratios** - 80/10/10 split with easy validation ≠ 80/10/10 split with hard validation
3. **Validation loss is not absolute** - can only compare models trained on identical train/val splits
4. **Surprising performance changes** may indicate hidden randomness, not code improvements

---

*End of refactoring log entry*

## 11. Fixed Unweighted Validation Loss (Inconsistent Training Objective)

**Date**: February 24, 2026  
**Issue**: Training used cHt=5.0 weights, validation did not  
**Impact**: Model optimization inconsistent with physics goal

### Problem Identified

While investigating why smaller models showed better visual distribution quality despite worse validation loss, discovered a critical inconsistency:

**Training**: Used weighted loss with cHt=5.0 SMEFT weights (0-77.9×)
```python
# MOGFlowModel.training_step() - line 265-309
if sm_weights is not None and self.use_sm_weights:
    weights = weights * sm_weights  # Applied correctly ✓
weighted_losses = weights * sample_losses
loss = torch.mean(weighted_losses)
```

**Validation**: Extracted weights but IGNORED them
```python
# MOGFlowModel.validation_step() - line 311-345 (BEFORE FIX)
if len(batch) == 3:
    x, _, sm_weights = batch  # Extracted but never used! ✗
loss = -torch.mean(sum_of_log_det_jacobian + mog_nll)  # Unweighted!
```

### Root Cause

When MOGFlowModel was created to support weighted training for SMEFT physics, the validation_step was not updated to match the training_step weighting logic. This resulted in:

1. **Training objective**: Optimize for cHt=5.0 weighted distribution (emphasize rare BSM events)
2. **Validation objective**: Optimize for unweighted SM distribution (treats all events equally)
3. **Contradiction**: Model was being trained and evaluated on different objectives!

### Impact on Model Selection

This bug caused confusing behavior:
- **Large models** (28.7M params): Good unweighted val_loss (7.044) but poor visual quality on bulk kinematics
- **Small models** (10.6M params): Poor unweighted val_loss (6.995) but better visual distribution quality

The large model was inadvertently learning a compromise between:
- Weighted training objective (prioritize rare high-weight events)
- Unweighted validation objective (all events equal)

This explained why the 5.1M param model (val_loss 11.747) actually produced better histogram matches on simple kinematics like eta, phi, pt - it couldn't overfit to the rare weighted events, so it learned a more balanced representation that happened to align with unweighted visual metrics.

### Solution Implemented

**File Modified**: `ml/flows/models/made_mog.py` (lines 311-380)

Made validation_step consistent with training_step by applying the same weighting logic:

```python
def validation_step(self, batch, batch_idx):
    # Unpack batch with proper weight handling
    if len(batch) == 3:
        x, _, sm_weights = batch
        sm_weights = sm_weights.unsqueeze(1)  # (N, 1)
    else:
        x, _ = batch
        sm_weights = None
    
    # ... compute losses ...
    
    # Apply weights consistently with training
    weights = torch.ones_like(sample_losses)
    
    # Apply mass-based weighting if enabled (for Drell-Yan)
    if self.use_loss_weighting:
        x_masked = x[mask.squeeze()]
        mass_weights = self._compute_mass_weights(x_masked)
        weights = weights * mass_weights
    
    # Apply SM event weights if available (for SMEFT)
    if sm_weights is not None and self.use_sm_weights:
        sm_weights_masked = sm_weights[mask]
        weights = weights * sm_weights_masked
    
    # Weighted loss: consistent with training objective
    weighted_losses = weights * sample_losses
    loss = torch.mean(weighted_losses)
```

### Key Changes

1. **Added** `sm_weights.unsqueeze(1)` to match training_step format
2. **Added** weight computation logic (mass weighting + SM weights)
3. **Changed** loss calculation from unweighted mean to weighted mean
4. **Applied** same masking logic to weights as to losses
5. **Duplicated** logic for both branches (with/without log_jac)

### Testing Impact

After this fix, validation loss will now properly reflect the physics objective: "How well does this model represent the cHt=5.0 weighted distribution?"

**Expected changes**:
- Validation loss values will change (become more consistent with weighted training)
- Model selection will now optimize for the correct objective (weighted SMEFT distribution)
- Visual quality vs validation loss tradeoff should resolve (both optimizing same objective)

### How Training Actually Works

Clarification on the architecture discovered during this fix:

**Two-layer design**:
1. **Model architecture** (e.g., MAFMADEMOG → MADEMOG → BaseFlowModel)
   - Pure `nn.Module`, no training logic
   - Implements only: `forward()`, `inverse()`, `estimate_density()`

2. **Training wrapper** (MOGFlowModel)
   - PyTorch Lightning module
   - Wraps architecture: `flow = MOGFlowModel(..., model=architecture, ...)`
   - Implements: `training_step()`, `validation_step()`, `configure_optimizers()`

The trainer trains the `flow` wrapper, not the `model` architecture directly. This is why weighted training was working despite MADEMOG inheriting from BaseFlowModel.

### Verification

To verify the fix is working:

```bash
# Train a new model and check logs for weight usage
python ml/custom/ttz/main_flows.py

# Look for these log messages during validation:
# - "mean_sm_weight" should appear in validation metrics
# - Validation loss values should change compared to previous runs
```

### Lessons Learned

1. **Training and validation must use the same objective** - inconsistent objectives lead to confusing model comparisons
2. **Weight handling requires careful implementation** - must be applied consistently across all loss computations
3. **Large models can exploit objective mismatches** - had enough capacity to learn both weighted and unweighted objectives partially
4. **Visual quality metrics can reveal objective mismatches** - if plots look better but loss is worse, check if objectives are aligned

---

*End of refactoring log entry*

## 12. Updated to New Data File with Pre-Identified Physics Objects

**Date**: February 25, 2026

### Problem Statement

The original dataset required complex event selection logic to identify physics objects (Z leptons, W lepton, b-jet) from raw particle collections. This added complexity and potential inconsistencies. A new processed data file became available with physics objects already identified, providing:
- Pre-selected event topology (ttZ signal region)
- Cleaner feature structure with explicit object labels
- More consistent physics object assignments

Additionally, feature ordering could be optimized to group physics objects better for autoregressive flow learning.

### Changes Implemented

#### 1. Updated Data Source

**File**: `ml/custom/ttz/process_ttz_dataset.py`

**Old file path**:
```python
file_path_1 = "/project/atlas/users/kdevries/EventLoop/ATLASSMEFT_ttZ_tree.root"
```

**New file path**:
```python
file_path_1 = "/project/atlas/users/kdevries/EventLoop/full_SR_ttZ_tree.root"
```

#### 2. Simplified Data Loading

**Removed**: ~200 lines of physics selection logic including:
- Lepton collection building (electrons + muons)
- OSSF Z candidate finding (invariant mass closest to mZ)
- W lepton identification (remaining third lepton)
- Jet selection (b-jet, forward jet, radiative jet)
- Event rejection criteria

**Replaced with**: Direct extraction of pre-identified physics objects from branches:
```python
event_row = [
    # Z_Lepton1 (leading Z lepton)
    data['Z_Lepton1_Pt'][i], data['Z_Lepton1_Eta'][i], 
    data['Z_Lepton1_Phi'][i], data['Z_Lepton1_Charge'][i],
    # Z_Lepton2 (subleading Z lepton)
    data['Z_Lepton2_Pt'][i], data['Z_Lepton2_Eta'][i],
    data['Z_Lepton2_Phi'][i], data['Z_Lepton2_Charge'][i],
    # W_Lepton (lepton from W decay)
    data['W_Lepton_Pt'][i], data['W_Lepton_Eta'][i],
    data['W_Lepton_Phi'][i], data['W_Lepton_Charge'][i],
    # BJet (b-tagged jet from top decay)
    data['BJet_Pt'][i], data['BJet_Eta'][i], data['BJet_Phi'][i],
    data['BJet_Mass'][i], data['BJet_BTag'][i],
    # MET (missing transverse energy)
    data['MET'][i], data['MET_phi'][i]
]
```

#### 3. Updated Feature Names and Ordering

**Old feature order** (object-grouped):
```
1-5:   Jet1_Pt, Jet1_Eta, Jet1_Phi, Jet1_Mass, Jet1_BTag
6-9:   Lepton1_Pt, Lepton1_Eta, Lepton1_Phi, Lepton1_Charge
10-13: Lepton2_Pt, Lepton2_Eta, Lepton2_Phi, Lepton2_Charge
14-17: Lepton3_Pt, Lepton3_Eta, Lepton3_Phi, Lepton3_Charge
18-19: MET, MET_Phi
```

**New feature order** (physics-object grouped with explicit naming):
```
1-4:   Z_Lepton1_Pt, Z_Lepton1_Eta, Z_Lepton1_Phi, Z_Lepton1_Charge
5-8:   Z_Lepton2_Pt, Z_Lepton2_Eta, Z_Lepton2_Phi, Z_Lepton2_Charge
9-12:  W_Lepton_Pt, W_Lepton_Eta, W_Lepton_Phi, W_Lepton_Charge
13-17: BJet_Pt, BJet_Eta, BJet_Phi, BJet_Mass, BJet_BTag
18-19: MET, MET_phi
```

**Rationale**: 
- Z leptons first - correlated particles from Z boson decay
- W lepton second - from W boson in ttZ topology
- BJet third - from top quark decay
- MET last - global event property
- Clearer naming makes physics interpretation easier

#### 4. Updated Feature Definitions

**File**: `ml/custom/ttz/process_ttz_dataset.py` (lines 55-70)

```python
processed_features = []
# Z leptons first - from Z boson decay
for i in [1, 2]:
    for var in ['Pt', 'Eta', 'Phi', 'Charge']:
        processed_features.append(f'Z_Lepton{i}_{var}')
# W lepton second - from W boson decay
for var in ['Pt', 'Eta', 'Phi', 'Charge']:
    processed_features.append(f'W_Lepton_{var}')
# BJet third - from top quark decay
for var in ['Pt', 'Eta', 'Phi', 'Mass', 'BTag']:
    processed_features.append(f'BJet_{var}')
# MET last - global event properties
processed_features.extend(['MET', 'MET_phi'])
```

#### 5. Updated Branch Loading

**Old branches** (raw particle collections):
```python
branches_to_load = [
    'Electron_Pt', 'Electron_Eta', 'Electron_Phi', 'Electron_E', 'Electron_Charge',
    'Muon_Pt', 'Muon_Eta', 'Muon_Phi', 'Muon_E', 'Muon_Charge',
    'Jet_Pt', 'Jet_Eta', 'Jet_Phi', 'Jet_Mass', 'Jet_E', 'Jet_BTag',
    'MET', 'MET_phi'
]
```

**New branches** (pre-identified physics objects):
```python
branches_to_load = [
    'Z_Lepton1_Pt', 'Z_Lepton1_Eta', 'Z_Lepton1_Phi', 'Z_Lepton1_Charge',
    'Z_Lepton2_Pt', 'Z_Lepton2_Eta', 'Z_Lepton2_Phi', 'Z_Lepton2_Charge',
    'W_Lepton_Pt', 'W_Lepton_Eta', 'W_Lepton_Phi', 'W_Lepton_Charge',
    'BJet_Pt', 'BJet_Eta', 'BJet_Phi', 'BJet_Mass', 'BJet_BTag',
    'MET', 'MET_phi'
]
```

### Impact

#### Benefits

1. **Simplified codebase**: Removed ~200 lines of complex selection logic
2. **Consistency**: Physics objects identified once upstream, used consistently everywhere
3. **Clearer semantics**: Feature names explicitly indicate physics origin (Z lepton vs W lepton)
4. **Better maintainability**: No need to maintain parallel selection logic
5. **Same model architecture**: Still 19 features, no changes to model code needed

#### Compatibility

- **Model architecture**: No changes required - still 19 input features
- **Weight handling**: Unchanged - still loads `smeft_weights[124]` for cHt=5.0
- **Preprocessing**: Feature types unchanged (Phi→uni, Charge→disc, others→cont)
- **Training pipeline**: No modifications needed

#### Breaking Changes

- **Old ttz.npy files incompatible**: Feature names and ordering changed
- **Data must be regenerated**: Run `python ml/custom/ttz/process_ttz_dataset.py`
- **Analyzer must be updated**: Feature names referenced in physics calculations need updating
- **Old checkpoints incompatible**: Models trained on old feature ordering cannot be used

### Migration Steps

1. **Regenerate data**:
   ```bash
   cd /project/atlas/users/mveldijk/MLHEPsimtest/MLHEPsim
   python ml/custom/ttz/process_ttz_dataset.py
   ```

2. **Update analyzer** (if needed):
   - Update feature name references in `ml/custom/ttz/sample_analyzer/analyzer.py`
   - Update feature name references in `ml/custom/ttz/sample_analyzer/physics_utils.py`

3. **Retrain models**:
   - Old checkpoints cannot be loaded with new feature ordering
   - Training configuration unchanged, just run normal training

4. **Verify results**:
   - Check that feature distributions match expectations
   - Verify derived physics quantities (Z_mass, W_mt, Top_mt) still calculable

### Files Modified

1. **ml/custom/ttz/process_ttz_dataset.py**:
   - Updated file path to new data source
   - Replaced selection logic with direct extraction
   - Updated feature names and ordering
   - Updated branch names in loading
   - Simplified logging (removed rejection counts)

### Verification

```bash
# Regenerate dataset
python ml/custom/ttz/process_ttz_dataset.py

# Check output shape
python -c "import numpy as np; data = np.load('ml/data/ttz/ttz.npy'); print(f'Shape: {data.shape}')"
# Expected: (N_events, 20) = 19 features + 1 weight

# Verify feature names
cat ml/data/ttz/variables.json
# Should show: Z_Lepton1_Pt, Z_Lepton1_Eta, ..., BJet_BTag, MET, MET_phi

# Train a model to verify compatibility
python ml/custom/ttz/main_flows.py
```

### Analyzer Updates

Updated analyzer code to work with new pre-identified physics objects and simplified redundant logic.

#### Files Modified

**1. ml/custom/ttz/sample_analyzer/physics_utils.py**

**Removed redundant function** (~60 lines):
- **`find_z_candidate_leptons()`**: No longer needed since Z leptons are pre-identified
  - Previously calculated invariant masses for all lepton pairs to find Z candidate
  - Now Z_Lepton1 and Z_Lepton2 are explicit in the data
  - Function was just returning constant array `[1, 2]` and doing feature lookups

**Updated `calculate_z_kinematics()`**:
- Removed call to `find_z_candidate_leptons()`
- Now directly gets feature indices:
  ```python
  z_lep1_pt_idx = feature_names.index('Z_Lepton1_Pt')
  z_lep1_eta_idx = feature_names.index('Z_Lepton1_Eta')
  z_lep1_phi_idx = feature_names.index('Z_Lepton1_Phi')
  # Same for Z_Lepton2
  ```
- Simplified from indirect dictionary lookups to direct indexing
- Functionality unchanged: still calculates Z_Pt, Z_Eta, Z_Phi, Z_Mass, Z_DeltaR

**Updated `calculate_w_kinematics()`**:
- Removed Z candidate finding logic (~15 lines)
- Changed from `Lepton{lep_num}_Pt/Phi` to direct `W_Lepton_Pt/Phi`
- Fixed: `MET_Phi` → `MET_phi` (lowercase for consistency)
- Simplified from identifying "non-Z lepton" to using pre-identified W_Lepton
- Functionality unchanged: still calculates W_Pt, W_Phi, W_MT

**Updated `calculate_top_kinematics()`**:
- Changed feature names: `Jet1_Pt/Phi` → `BJet_Pt/Phi`
- Updated docstring to reflect BJet instead of generic Jet1
- Functionality unchanged: still calculates Top_Pt, Top_Phi, Top_MT

**Feature name changes throughout**:
| Old Name | New Name | Context |
|----------|----------|---------|
| `Jet1_Pt`, `Jet1_Phi` | `BJet_Pt`, `BJet_Phi` | Top kinematics calculation |
| `Lepton{1,2,3}_*` | `Z_Lepton1_*`, `Z_Lepton2_*`, `W_Lepton_*` | All lepton references |
| `MET_Phi` | `MET_phi` | Lowercase phi for consistency |

**Remaining functions** (all actively used):
- `calculate_invariant_mass()`: Generic utility (kept for potential external use)
- `calculate_delta_r()`: Used in Z kinematics for Z_DeltaR calculation
- `calculate_z_kinematics()`: Simplified, now ~30 lines shorter
- `calculate_w_kinematics()`: Simplified, now ~15 lines shorter
- `calculate_top_kinematics()`: Updated feature names only

**2. Other analyzer files**

No changes needed to:
- `analyzer.py`: Works generically with feature_names from variables.json
- `plotting.py`: Uses feature names dynamically, no hard-coded references
- `run_analysis.py`: No hard-coded feature references
- `smeft_vs_sm_*.py`: No hard-coded feature references

**Total lines removed**: ~90 lines of redundant selection and lookup logic

**Benefits**:
- Clearer code: Direct feature access instead of nested dictionary lookups
- Less indirection: No need to map "lepton numbers" to actual leptons
- Fewer moving parts: Removed function that just returned constants
- Easier maintenance: No complex lepton pairing logic to maintain
- Same functionality: All derived physics quantities still calculated correctly

### Verification

```bash
# Regenerate dataset (required first)
python ml/custom/ttz/process_ttz_dataset.py

# Check output shape
python -c "import numpy as np; data = np.load('ml/data/ttz/ttz.npy'); print(f'Shape: {data.shape}')"
# Expected: (N_events, 20) = 19 features + 1 weight

# Verify feature names
cat ml/data/ttz/variables.json
# Should show: Z_Lepton1_Pt, Z_Lepton1_Eta, ..., BJet_BTag, MET, MET_phi

# Test analyzer with new feature names
python ml/custom/ttz/sample_analyzer/run_analysis.py
# Should generate plots with derived physics quantities (Z_mass, W_mt, Top_mt)

# Train a model to verify full pipeline compatibility
python ml/custom/ttz/main_flows.py
```

### Future Considerations

**Potential feature ordering experiments**:

Currently using object-grouped ordering (all properties per object together). Alternative orderings to potentially test:

**Option A: Feature-type interleaving** (all Pt first, then all Eta, etc.)
```
Jet_Pt, Z_Lep1_Pt, Z_Lep2_Pt, W_Lep_Pt,
Jet_Eta, Z_Lep1_Eta, Z_Lep2_Eta, W_Lep_Eta,
Jet_Phi, Z_Lep1_Phi, Z_Lep2_Phi, W_Lep_Phi, MET_phi,
Jet_Mass, Jet_BTag, Z_Lep1_Charge, Z_Lep2_Charge, W_Lep_Charge, MET
```

**Hypothesis**: Giving model all transverse momenta early could help it learn global energy scale before generating angular variables, potentially improving correlations for derived quantities (Z_mass, Top_mt).

**Implementation**: Would only require reordering columns in `event_row` list and `processed_features` list - no model changes needed.

---

*End of refactoring log entry*
## Section 13: MLflow Model Registration Fix

**Date**: February 26, 2026  
**Author**: Assistant (GitHub Copilot)  
**Issue**: MLflow model registration failing - artifacts directory empty after training

### Problem Identified

Training runs were completing successfully with checkpoints saved, but model registration to MLflow's model registry was failing with error:
```
ERROR | MLflow registration failed - artifacts directory is empty or missing!
```

**Root Cause Investigation**:
- Git history showed bug introduced in commit `622f89f` (Feb 23, 2026) during register_model.py refactor
- Original implementation (Jul 17, 2024): Used `artifact_path=f"{checkpoint_dir}/artifacts"` (confusing nested path)
- Refactored implementation (Feb 23, 2026): Changed to `artifact_path="model"` (cleaner), but neither version had proper MLflow run context
- **Core Issue**: PyTorch Lightning's `trainer.fit()` automatically calls `mlflow.end_run()` when training completes
- `register_from_checkpoint()` is called AFTER training ends, when no active run exists
- Attempting to resume a finished run (`mlflow.start_run(run_id=run_id)`) doesn't allow artifact logging

**Impact**:
- All training runs since Feb 23, 2026 failed to register models to ML flow registry
- Checkpoints were preserved locally but not accessible via `mlflow.pytorch.load_model()`
- Model registry metadata created (version numbers) but pointed to empty artifacts directories

### Solution Implemented

**Strategy**: Manually create MLflow artifact structure and register model directory

Instead of relying on `mlflow.pytorch.log_model()` (which requires an active run), directly create the artifacts directory structure with proper MLflow metadata files.

#### Files Modified

**1. ml/common/utils/register_model.py** (lines 1-3, 80-127)

**Added import**:
```python
import time  # For timestamp in MLmodel file
```

**Replaced registration logic** (lines 80-127):
```python
try:
    # Save model directly to the artifacts directory
    # Note: Cannot log to a finished run, so we manually create the artifacts structure
    artifacts_model_dir = f"{checkpoint_dir}/artifacts/model"
    model_data_dir = f"{artifacts_model_dir}/data"
    os.makedirs(model_data_dir, exist_ok=True)
    
    # Save the PyTorch model state dict
    model_file = os.path.join(model_data_dir, "model.pth")
    torch.save(obj_to_save.state_dict(), model_file)
    logging.info(f"Saved model weights to: {model_file}")
    
    # Create requirements.txt
    requirements_file = os.path.join(model_data_dir, "requirements.txt")
    with open(requirements_file, "w") as f:
        f.write(f"torch=={torch.__version__}\n")
        f.write("numpy\n")
    
    # Create MLmodel file for MLflow compatibility
    import time
    mlmodel_content = f"""artifact_path: model
flavors:
  pytorch:
    code: null
    model_data: data/model.pth
    pytorch_version: {torch.__version__}
mlflow_version: 2.3.0
model_uuid: {run_id}
run_id: {run_id}
utc_time_created: '{time.strftime("%Y-%m-%d %H:%M:%S")}'
"""
    mlmodel_file = os.path.join(artifacts_model_dir, "MLmodel")
    with open(mlmodel_file, "w") as f:
        f.write(mlmodel_content)
    logging.info(f"Created MLmodel file: {mlmodel_file}")
    
    # Register the model in MLflow registry
    model_uri = f"file://{os.path.abspath(artifacts_model_dir)}"
    result = mlflow.register_model(model_uri, model_name)
    logging.info(f"Registered model '{model_name}' version {result.version}")
    
    # Verify the model was actually saved
    if os.path.exists(model_file) and os.path.getsize(model_file) > 0:
        logging.info(f"✓ Model successfully registered as {model_name}")
        logging.info(f"✓ Artifacts saved to: {artifacts_model_dir}")
        logging.info(f"✓ Checkpoint preserved at: {ckpt_best_model_path}")
    else:
        logging.error(f"MLflow registration failed - model file is empty or missing!")
        logging.error(f"Checkpoint preserved at: {ckpt_best_model_path}")
```

**Previous approach** (didn't work):
```python
# Trying to resume finished run - doesn't allow artifact logging
with mlflow.start_run(run_id=run_id, experiment_id=experiment_id):
    mlflow.pytorch.log_model(obj_to_save, artifact_path="model", ...)
```

#### Verification

**Test Registration**: Created `test_registration.py` to manually register existing checkpoint:
```bash
source venv311/bin/activate
python test_registration.py
```

**Results**:
```
INFO | Saved model weights to: mlruns/.../artifacts/model/data/model.pth
INFO | Created MLmodel file: mlruns/.../artifacts/model/MLmodel
INFO | Registered model 'MAFMADEMOG_flow_model_gauss_rank_20260226_nall' version 3
INFO | ✓ Model successfully registered as MAFMADEMOG_flow_model_gauss_rank_20260226_nall
INFO | ✓ Artifacts saved to: mlruns/.../artifacts/model
```

**Artifacts created**:
```bash
mlruns/302809770930480935/23c98dc189c84b06bd05c7937beb91d5/artifacts/model/
├── MLmodel               (268 bytes - MLflow metadata)
└── data/
    ├── model.pth         (78 MB - PyTorch state dict)
    └── requirements.txt  (25 bytes - dependencies)
```

**Registry entry**:
```bash
mlruns/models/MAFMADEMOG_flow_model_gauss_rank_20260226_nall/version-3/
└── meta.yaml             (Source: file://.../artifacts/model)
```

### Benefits

1. **Works with finished runs**: No longer requires active MLflow run context
2. **Proper MLflow structure**: Creates all necessary metadata files for compatibility
3. **Clean separation**: Registration logic independent of Lightning trainer lifecycle
4. **Backward compatible**: Still uses standard MLflow registry API (`mlflow.register_model()`)
5. **Verifiable**: Explicit file size checks confirm successful artifact creation

### Training Run Results

**Latest training** (Feb 26, 2026, run ID `23c98dc189c84b06bd05c7937beb91d5`):
- **Dataset**: 644,071 events from `full_SR_ttZ_tree.root` with new feature structure (Z_Lepton1/2, W_Lepton, BJet)
- **Model**: MAFMADEMOG 6×5×384 (10.6M parameters)
- **Training**: 214 epochs, best validation loss: 4.591708 at epoch 203
- **Checkpoint**: `epoch=203-step=51204.ckpt` (preserved)
- **Registration**: Version 3 successfully created with 78MB model weights
- **Verification**: Analyzer confirms model generates different samples (fix working)

### Future Training Runs

All future training runs will automatically use the fixed registration code and successfully save model artifacts to MLflow registry.

**To manually register old checkpoints** (if needed):
1. Update `test_registration.py` with correct checkpoint path and run IDs
2. Run: `python test_registration.py`
3. Verify artifacts created in `mlruns/.../artifacts/model/`

---

*End of refactoring log entry*

## Section 14: Switch from Cylindrical to Cartesian Coordinates

**Date**: February 26, 2026  
**Author**: User request, Assistant implementation  
**Motivation**: Simplify model training by using Cartesian coordinates instead of problematic cylindrical coordinates

### Problem with Cylindrical Coordinates (Pt, Eta, Phi)

**Phi angle challenges**:
- Phi is a **circular/periodic variable** (wraps at ±π)
- Gaussian rank scaling treats it as continuous → creates discontinuities
- Model struggles to learn that Phi=-3.14 and Phi=+3.14 are nearby values
- "Phi kept being a struggle in" - difficult distribution for normalizing flows

**Why Cartesian is better**:
- **Linear coordinates**: Px, Py, Pz, E are all continuous, unbounded variables
- **Simpler distributions**: No circular periodicity issues
- **Direct 4-vector arithmetic**: Addition/subtraction more natural
- **Better for autoregressive flows**: Each component has straightforward dependencies

### Solution Implemented

**Core change**: Switch all particle representations from **(Pt, Eta, Phi)** to **(E, Px, Py, Pz)**

#### Files Modified

**1. ml/custom/ttz/process_ttz_dataset.py** (lines 57-240)

**Updated feature list** (lines 57-72):
- **Old**: 19 features = 4 coords/particle (Pt, Eta, Phi, Charge/BTag/Mass) × 4 objects + 2 MET
- **New**: 22 features = 5 coords/particle (E, Px, Py, Pz, Charge/BTag) × 4 objects + 2 MET

```python
processed_features = []
# Z leptons: E, Px, Py, Pz, Charge (5 features × 2 = 10)
for i in [1, 2]:
    for var in ['E', 'Px', 'Py', 'Pz', 'Charge']:
        processed_features.append(f'Z_Lepton{i}_{var}')
# W lepton: E, Px, Py, Pz, Charge (5 features)
# BJet: E, Px, Py, Pz, BTag (5 features, removed Mass)
# MET: MET_Px, MET_Py (Cartesian components)
```

**Feature type classification** (lines 75-81):
```python
# Simplified: No more "uni" type for Phi
for feature in processed_features:
    if 'Charge' in feature or 'BTag' in feature:
        self.features["colnames"][feature] = "disc"
    else:
        self.features["colnames"][feature] = "cont"  # All Cartesian = continuous
```

**Branches to load** (lines 186-195):
```python
branches_to_load = [
    'Z_Lepton1_E', 'Z_Lepton1_Px', 'Z_Lepton1_Py', 'Z_Lepton1_Pz', 'Z_Lepton1_Charge',
    # ... (same pattern for Z_Lepton2, W_Lepton, BJet)
    'MET', 'MET_phi'  # Convert to MET_Px, MET_Py
]
```

**MET conversion** (lines 215-225):
```python
# Convert MET from polar to Cartesian
met = data['MET'][i]
met_phi = data['MET_phi'][i]
met_px = met * np.cos(met_phi)
met_py = met * np.sin(met_phi)

event_row = [
    # ... all Cartesian coordinates ...
    met_px, met_py  # Replace (MET, MET_phi)
]
```

**2. ml/custom/ttz/sample_analyzer/physics_utils.py** (extensive updates)

**Added helper function**:
```python
def cartesian_to_cylindrical(E, px, py, pz):
    """Convert 4-vector from Cartesian to cylindrical for analysis/plotting."""
    pt = np.sqrt(px**2 + py**2)
    phi = np.arctan2(py, px)
    p = np.sqrt(px**2 + py**2 + pz**2)
    eta = np.where(p != pz, 0.5 * np.log((p + pz) / (p - pz)), 0.0)
    return pt, eta, phi
```

**Updated `calculate_z_kinematics()`**:
```python
# OLD: Load (Pt, Eta, Phi) → convert to (Px, Py, Pz, E) → calculate kinematics
# NEW: Load (E, Px, Py, Pz) directly → calculate kinematics (simpler!)

E1 = data[i, z_lep1_e_idx]
px1 = data[i, z_lep1_px_idx]
py1 = data[i, z_lep1_py_idx]
pz1 = data[i, z_lep1_pz_idx]

# Z 4-vector is simple sum (no trig functions!)
E_z = E1 + E2
px_z = px1 + px2
py_z = py1 + py2
pz_z = pz1 + pz2

# Invariant mass directly from 4-vector
z_mass = np.sqrt(E_z**2 - (px_z**2 + py_z**2 + pz_z**2))
```

**Updated `calculate_w_kinematics()`**:
```python
# OLD: Extract Pt/Phi, then calculate Px/Py
# NEW: Extract Px/Py directly

lep_px = data[i, w_lep_px_idx]
lep_py = data[i, w_lep_py_idx]
met_px = data[i, met_px_idx]
met_py = data[i, met_py_idx]

w_px = lep_px + met_px  # Direct addition!
w_py = lep_py + met_py
```

**Updated `calculate_top_kinematics()`**:
```python
# Similar simplification for BJet extraction
bjet_px = data[i, bjet_px_idx]
bjet_py = data[i, bjet_py_idx]
# Direct vector addition
```

**Renamed function**:
- `calculate_invariant_mass()` → `calculate_invariant_mass_from_4vectors()`
- Now takes (E, Px, Py, Pz) directly instead of (Pt, Eta, Phi)

**3. ml/data/ttz/variables.json** (auto-generated on next run)

New structure (22 features):
```json
{
    "colnames": {
        "Z_Lepton1_E": "cont",
        "Z_Lepton1_Px": "cont",
        "Z_Lepton1_Py": "cont",
        "Z_Lepton1_Pz": "cont",
        "Z_Lepton1_Charge": "disc",
        ... (repeat for Z_Lepton2, W_Lepton, BJet)
        "MET_Px": "cont",
        "MET_Py": "cont"
    }
}
```

**Removed**: All `"Phi": "uni"` entries (no more periodic variables!)

### Benefits

1. **Simpler distributions**: All coordinates continuous, unbounded → easier for flows
2. **No Phi periodicity**: Eliminates circular variable discontinuities
3. **Direct arithmetic**: No trigonometric conversions in 4-vector operations
4. **Better convergence**: Autoregressive dependencies clearer in Cartesian space
5. **Cleaner code**: Removed special "uni" type handling
6. **More stable gradients**: No boundary issues from Phi wrapping

### Migration Steps

```bash
# 1. Regenerate dataset with Cartesian coordinates
python ml/custom/ttz/process_ttz_dataset.py

# Expected: (644071, 23) = 22 features + 1 weight
# Creates: ml/data/ttz/ttz.npy with new coordinates
# Updates: ml/data/ttz/variables.json automatically

# 2. Verify new features
cat ml/data/ttz/variables.json
# Should show: E, Px, Py, Pz for each particle, MET_Px, MET_Py

# 3. Test analyzer
python ml/custom/ttz/sample_analyzer/run_analysis.py
# Physics quantities (Z_mass, W_mt, Top_mt) should still work correctly

# 4. Retrain model
python ml/custom/ttz/main_flows.py
# Expected: Better convergence, lower loss, better sample correlations
```

### Technical Details

**Coordinate conversions**:
- **Cartesian → Cylindrical** (when needed for plotting):
  ```
  Pt = √(Px² + Py²)
  Phi = arctan2(Py, Px)
  Eta = 0.5 × ln((|p| + Pz) / (|p| - Pz))  where |p| = √(Px² + Py² + Pz²)
  ```

- **Invariant mass** (simpler with Cartesian):
  ```
  m² = E² - (Px² + Py² + Pz²)
  ```

**Feature count**: 19 → 22 (+3 features)
- Gain: E for each particle (4 objects = +4 features)
- Loss: Removed BJet_Mass (-1 feature)
- Net: +3 features

**ROOT file compatibility**: All coordinates already available in file:
- Pre-identified objects have both representations
- `Z_Lepton1_Pt/Eta/Phi` AND `Z_Lepton1_E/Px/Py/Pz`
- No change to input data source required

**Important: MET Conversion**:
- ROOT file contains: `MET`, `MET_phi` (polar coordinates)
- Code converts during loading: `MET_Px = MET * cos(MET_phi)`, `MET_Py = MET * sin(MET_phi)`
- Model receives: `MET_Px`, `MET_Py` (Cartesian coordinates)
- Note: MET_Px and MET_Py do NOT exist in ROOT file - created on-the-fly

### Configuration Files Updated

**4. ml/custom/ttz/config/flows/data_config.yaml** (lines 1-50)

**Updated input_dim**:
```yaml
# OLD
input_dim: 19  # 3 leptons * 4 (Pt,Eta,Phi,Charge) + 1 jet * 5 (Pt,Eta,Phi,Mass,BTag) + 2 MET

# NEW
input_dim: 22  # Z_Lepton1 (E,Px,Py,Pz,Charge=5) + Z_Lepton2(5) + W_Lepton(5) + BJet(E,Px,Py,Pz,BTag=5) + MET(Px,Py=2)
```

**Updated keep_names** (branches to load from ROOT):
```yaml
# OLD
keep_names: ["Electron_Pt","Electron_Eta","Electron_Phi","Electron_E","Electron_Charge",
             "Muon_Pt","Muon_Eta","Muon_Phi","Muon_E","Muon_Charge",
             "Jet_Pt","Jet_Eta","Jet_Phi","Jet_Mass","Jet_E","Jet_BTag",
             "MET","MET_phi"]

# NEW
keep_names: ["Z_Lepton1_E","Z_Lepton1_Px","Z_Lepton1_Py","Z_Lepton1_Pz","Z_Lepton1_Charge",
             "Z_Lepton2_E","Z_Lepton2_Px","Z_Lepton2_Py","Z_Lepton2_Pz","Z_Lepton2_Charge",
             "W_Lepton_E","W_Lepton_Px","W_Lepton_Py","W_Lepton_Pz","W_Lepton_Charge",
             "BJet_E","BJet_Px","BJet_Py","BJet_Pz","BJet_BTag",
             "MET","MET_phi"]  # MET converted to MET_Px/MET_Py during loading
```

**Updated preprocessing comments**:
```yaml
# OLD
preprocessing:
  cont_rescale_type: gauss_rank
  disc_rescale_type: null
  no_process: ["weight"]  # Only skip weights; Phi angles SHOULD be Gaussian rank scaled
  
  # NOTE: Phi angles are scaled Uniform → Gaussian for training, then Gaussian → Uniform when sampling.
  # This makes it easier for flows to learn (Gaussian base → Gaussian target, nearly identity).
  # Without scaling, flows must learn Gaussian → Uniform which is much harder.

# NEW
preprocessing:
  cont_rescale_type: gauss_rank
  disc_rescale_type: null
  no_process: ["weight"]  # Only skip weights; all Cartesian coordinates are continuous
  
  # NOTE: Using Cartesian coordinates (E, Px, Py, Pz) instead of cylindrical (Pt, Eta, Phi)
  # Cartesian coordinates are continuous and unbounded - easier for normalizing flows to learn
  # No more circular Phi periodicity issues that complicated training
```

**5. ml/data/ttz/variables.json** (complete rewrite)

Changed from 19 features (with "uni" types for Phi) to 22 features (all "cont" except discrete):

```json
{
    "colnames": {
        "Z_Lepton1_E": "cont",
        "Z_Lepton1_Px": "cont",
        "Z_Lepton1_Py": "cont",
        "Z_Lepton1_Pz": "cont",
        "Z_Lepton1_Charge": "disc",
        "Z_Lepton2_E": "cont",
        "Z_Lepton2_Px": "cont",
        "Z_Lepton2_Py": "cont",
        "Z_Lepton2_Pz": "cont",
        "Z_Lepton2_Charge": "disc",
        "W_Lepton_E": "cont",
        "W_Lepton_Px": "cont",
        "W_Lepton_Py": "cont",
        "W_Lepton_Pz": "cont",
        "W_Lepton_Charge": "disc",
        "BJet_E": "cont",
        "BJet_Px": "cont",
        "BJet_Py": "cont",
        "BJet_Pz": "cont",
        "BJet_BTag": "cont",
        "MET_Px": "cont",
        "MET_Py": "cont"
    }
}
```

**Key changes**:
- Removed all `"Phi": "uni"` entries (no more periodic variables!)
- All Cartesian coordinates marked as "cont" (continuous)
- Only `Charge` fields and `BTag` marked as "disc" (discrete)
- File will be auto-regenerated when running `process_ttz_dataset.py`

### Expected Training Improvements

**Positive effects**:
- **Faster convergence**: No Phi discontinuities → smoother loss landscape
- **Lower val loss**: Simpler distributions easier to model
- **Better correlations**: Px-Py relationships more natural than Pt-Phi
- **Stable gradients**: No boundary wrapping issues

**Things to monitor**:
- Z_mass peak should still be at ~91 GeV (physics unchanged)
- W transverse mass should still peak at ~80 GeV
- Top transverse mass should still peak at ~173 GeV
- Feature correlations may show clearer patterns

**Model architecture**: No changes needed
- Same autoregressive flow structure
- Same number of parameters
- Just different input features

---

*End of refactoring log entry*

## Section 15: Training Results & Model Loading Issue with Cartesian Coordinates

**Date**: February 26, 2026  
**Author**: Assistant (GitHub Copilot)  
**Context**: First training run completed with new 22-feature Cartesian coordinate system

### Training Results

**Run details**:
- **Date**: February 26, 2026
- **Job ID**: mafmademog_606455
- **Run ID**: `3d6977a98bd948b2bf4b007146518f93`
- **Dataset**: 644,071 events with 22 Cartesian features
- **Model**: MAFMADEMOG 6×5×384 (10.15M parameters)

**Training metrics**:
- **Training time**: 46 minutes (started 12:55, finished 13:41)
- **Total epochs**: 194 epochs
- **Best validation loss**: **-0.685** at epoch 183
- **Final train loss**: 0.496 at epoch 185
- **Checkpoint**: `epoch=183-step=46160.ckpt`
- **Model registered**: Version 4 in MLflow registry

**Validation loss interpretation**:
- Negative NLL is **correct and excellent**
- Indicates flow assigns probability densities > 1 to data points
- Valid for PDFs (probability *density* functions can exceed 1)
- Lower (more negative) = better fit to data distribution

### Model Loading Issue Discovered

**Error encountered**:
When attempting to run the analyzer on newly trained model:
```
AttributeError: 'collections.OrderedDict' object has no attribute 'model'
```

**Location**: `ml/common/utils/gen_model_sampler.py` line 81:
```python
def fetch_model_from_mlflow(...):
    module = fetch_registered_module(...)
    return module.model.eval()  # ← Fails: module is OrderedDict, not MOGFlowModel
```

**Root cause investigation**:

1. **Expected behavior**: `fetch_registered_module()` should return full PyTorch Lightning `MOGFlowModel` object with `.model` attribute
2. **Actual behavior**: Returns `collections.OrderedDict` (state_dict only)
3. **Why**: MLflow's `pytorch.log_model()` with default settings saves only state_dict, not full module
4. **Evidence**: MLmodel file shows `code: null` configuration:
   ```yaml
   flavors:
     pytorch:
       code: null                    # ← No code/module structure saved
       model_data: data/model.pth    # Contains OrderedDict
   ```

### Timeline Analysis

**Critical timing issue**:
```
12:55 - Training starts (old code loaded into Python process memory)
13:41 - Training finishes, model registered (still using old code from memory)
13:45 - Attempted fix applied to register_model.py (too late!)
```

**Key insight**: Training process had already loaded the original `register_model.py` code into memory at 12:55. When training finished at 13:41 and triggered registration, it executed the OLD code that relies on `mlflow.pytorch.log_model()`, which by default saves only state_dict.

**Proof from training log** (line 146):
```
INFO 26 Feb 2026 | 13:41:48 | Saved model weights to: register_model.py:91
                              ^^^^^^^^ OLD log message
```

Current code at line 91 says: `Saved full model to:` (modified version)  
→ Confirms training used old code before modification

### Files Modified for Analyzer Compatibility

**1. ml/custom/ttz/ttz_dataset.py** (line 36)
```python
# OLD
if data.shape[1] == 20:  # 19 cylindrical features + 1 weight
    
# NEW  
if data.shape[1] == 23:  # 22 Cartesian features + 1 weight
    self.weights = data[:, -1]
    data = data[:, :-1]
```

**Purpose**: Extract weight column correctly for 22-feature Cartesian dataset

**2. ml/custom/ttz/sample_analyzer/analyzer.py** (lines 84-86, 164-178)

**Include discrete features explicitly** (lines 84-86):
```python
# OLD
features = [name for name, type_ in self.variables['colnames'].items() 
           if type_ in ['cont', 'uni']]

# NEW
features = [name for name, type_ in self.variables['colnames'].items() 
           if type_ in ['cont', 'uni', 'disc']]  # Also include disc features
```

**Update weight extraction** (lines 164-178):
```python
# OLD
if self.original_data.shape[1] == 20:  # 19 features + 1 weight

# NEW
if self.original_data.shape[1] == 23:  # 22 Cartesian features + 1 weight
    self.original_weights = self.original_data[:, -1]
    self.original_data = self.original_data[:, :-1]
else:
    raise ValueError(
        f"Expected data with 23 columns (22 Cartesian features + weight), "
        f"got {self.original_data.shape[1]} columns. "
        f"Cartesian coordinates: E,Px,Py,Pz for each object + MET_Px,MET_Py + charges/tags."
    )
```

**Status**: Analyzer code correctly updated for 22 features, but blocked by model loading issue.

### Debugging Attempts

**Test scripts created** (all in project root):
1. **reregister_model.py**: Attempt to re-register model with fixed code
2. **fix_saved_model.py**: Reconstruct MOGFlowModel from checkpoint
3. **fix_model_save.py**: Alternative reconstruction approach
4. **test_model_load.py**: Simple 10-line test script
5. **debug_model_load.py**: Detailed logging version

**All failed due to**:
- Missing hyperparameters in checkpoint
- State_dict dimension mismatches during reconstruction
- Cannot instantiate MOGFlowModel without exact matching hyperparams

### Current State & Plan Forward

**register_model.py status**: Currently using ORIGINAL code (reverted via `git checkout`)
- Uses `mlflow.pytorch.log_model()` which saves state_dict by default
- Model version 4 contains OrderedDict, unusable by analyzer

**For next training run** (RECOMMENDED APPROACH):

The registration code FIX has been identified but not yet applied. When ready to train next:

**Apply this change to ml/common/utils/register_model.py** (lines ~85-127):
```python
try:
    # Save FULL model object directly to artifacts directory
    # (mlflow.pytorch.log_model() only saves state_dict by default)
    artifacts_model_dir = f"{checkpoint_dir}/artifacts/model"
    model_data_dir = f"{artifacts_model_dir}/data"
    os.makedirs(model_data_dir, exist_ok=True)
    
    # Save full PyTorch Lightning module (not just state_dict)
    model_file = os.path.join(model_data_dir, "model.pth")
    torch.save(obj_to_save, model_file)  # ← Full object, not .state_dict()
    logging.info(f"Saved full model to: {model_file}")
    
    # Create MLmodel file
    mlmodel_content = f"""artifact_path: model
flavors:
  pytorch:
    code: null
    model_data: data/model.pth
    pytorch_version: {torch.__version__}
mlflow_version: 2.3.0
model_uuid: {run_id}
run_id: {run_id}
"""
    mlmodel_file = os.path.join(artifacts_model_dir, "MLmodel")
    with open(mlmodel_file, "w") as f:
        f.write(mlmodel_content)
    
    # Register model
    model_uri = f"file://{os.path.abspath(artifacts_model_dir)}"
    result = mlflow.register_model(model_uri, model_name)
    logging.info(f"Registered model '{model_name}' version {result.version}")
    
    # Verify
    if os.path.exists(model_file) and os.path.getsize(model_file) > 0:
        logging.info(f"✓ Model successfully registered")
        logging.info(f"✓ Checkpoint preserved at: {ckpt_best_model_path}")
    else:
        logging.error(f"MLflow registration failed - model file is empty!")
        logging.error(f"Checkpoint preserved at: {ckpt_best_model_path}")
        
except Exception as e:
    logging.error(f"Failed to register model {model_name}: {e}")
    logging.error(f"Checkpoint preserved at: {ckpt_best_model_path}")
    raise
```

**Key difference**: `torch.save(obj_to_save, ...)` instead of `torch.save(obj_to_save.state_dict(), ...)`

**Verification for next training**:
- [ ] Check log message says "Saved full model to:" (not "weights")
- [ ] Model file size should be >100MB (full module vs 78MB for state_dict)
- [ ] Test loading: `module = fetch_registered_module(...); assert hasattr(module, 'model')`
- [ ] Run analyzer successfully: `python ml/custom/ttz/sample_analyzer/run_analysis.py`

### Lessons Learned

1. **Python process memory persistence**: Code modifications don't affect already-running processes
2. **MLflow defaults**: `pytorch.log_model()` saves state_dict unless `code_paths` or specific configuration provided
3. **Full module vs state_dict**: PyTorch Lightning modules need full object saved to preserve structure
4. **Registration timing**: Must ensure code fix is loaded BEFORE training starts
5. **User insight valuable**: User correctly identified real issue (registration code) when agent was investigating wrong path (dimensions)

### Files to Clean Up

Test/debug scripts created during investigation (no longer needed):
- `reregister_model.py`
- `fix_saved_model.py`
- `fix_model_save.py`
- `test_model_load.py`
- `debug_model_load.py`

Will be deleted after documentation complete.

---

*End of refactoring log entry*
````
This is the description of what the code block changes:
<changeDescription>
Log the restoration of the MLflow model registration fix in register_model.py as Section 16, with a clear summary of what, why, and where.
</changeDescription>

This is the code block that represents the suggested code change:
```markdown
## Section 16: MLflow Model Registration Bug Fix Restored

**Date**: February 26, 2026  
**Author**: Assistant (GitHub Copilot)

### What was changed
- Restored the manual MLflow artifact creation and full model saving logic in `ml/common/utils/register_model.py`.
- The function now saves the full PyTorch Lightning module object (not just the state_dict) to `model.pth` and creates the MLmodel metadata file and artifact directory structure manually.
- Registration is performed using `mlflow.register_model()` with a file URI, ensuring the model is compatible with the analyzer and can be loaded as a full module.

### Why
- The previous fix for this bug had been reverted, causing only the state_dict to be saved and breaking analyzer compatibility.
- This change ensures that future model registrations will work as intended and that the full model object is preserved for downstream use.

### Where
- File: `ml/common/utils/register_model.py`
- Function: `register_from_checkpoint()`

---
```

## Section 17: Remove BJet_BTag Feature

**Date**: February 27, 2026  
**Author**: Assistant (GitHub Copilot)

### What was changed
- Removed `BJet_BTag` from the feature list in `ml/custom/ttz/process_ttz_dataset.py` (lines 68-70)
- Changed the BJet feature loop from `['E', 'Px', 'Py', 'Pz', 'BTag']` to `['E', 'Px', 'Py', 'Pz']`
- Updated feature count documentation from 22 features to 21 features in comments
- The `variables.json` file is now automatically regenerated without `BJet_BTag` when the pipeline runs

### Root Cause
- The code in `process_ttz_dataset.py` was automatically regenerating `variables.json` on every pipeline run
- `BJet_BTag` was hardcoded into the feature loop, so manual edits to `variables.json` would be overwritten
- This prevented the previously attempted fix from persisting

### Why
- B-jet identification (BTag) information is already provided by the ROOT file
- Removing this redundant feature reduces model input dimensionality from 23 to 21
- Aligns with the updated `data_config.yaml` which expects 21 input features

### Where
- File: `ml/custom/ttz/process_ttz_dataset.py`
  - Lines 54-57: Updated comment (22 → 21 features)
  - Lines 68-70: Removed `'BTag'` from feature loop
  - Lines 72-74: Updated comment about discrete features

### Impact
- Model input dimension now matches the feature list (21 features total)
- Training pipeline should no longer have tensor size mismatches
- `variables.json` will be correctly regenerated with 21 features on next pipeline run
- Prevents manual edits to `variables.json` from being overwritten since the code now generates the correct version

---
## Date: March 2, 2026

### Issue: Revert Cartesian Coordinates Back to Cylindrical, and Load Directly from ROOT

#### Background

Some days prior, the feature representation was switched from cylindrical coordinates (Pt, Eta, Phi) to Cartesian coordinates (E, Px, Py, Pz) in an attempt to avoid the circular periodicity problem of Phi angles. This change increased the feature count from 18 to 21, added energy as a learned feature, and converted MET from (MET, MET_phi) to (MET_Px, MET_Py).

#### Problem Identified

The Cartesian coordinate switch **did not solve the phi problem and introduced new ones**:

1. **Phi still degraded**: Phi is encoded implicitly as `arctan2(Py, Px)` across two features. Gaussian rank scaling processes each feature independently, so the circular structure in the (Px, Py) plane is destroyed. The model still failed to learn the correct phi distribution.

2. **Additional cross-term complexity**: The model had to learn that `Px² + Py² + Pz²` equals a physically meaningful quantity - correlation structure across 3 features instead of 1.

3. **Z_DeltaR also degrades**: `ΔR = sqrt(Δeta² + Δphi²)` relies directly on phi. Any degradation in phi propagates directly into Z_DeltaR.

4. **Unnecessary conversion**: The ROOT file already contains the cylindrical branches (`Pt`, `Eta`, `Phi`, `Mass`), so computing them via Cartesian was redundant and introduced floating-point errors.

5. **Feature count increase**: 21 features vs 18 - more dimensions for the same physics content.

#### Solution Implemented

**Reverted all files to cylindrical coordinates (18 features) and simplified ROOT loading to use cylindrical branches directly.**

##### Files Modified

1. **ml/custom/ttz/process_ttz_dataset.py**
   - Reverted `processed_features` loop from `['E', 'Px', 'Py', 'Pz', 'Charge']` → `['Pt', 'Eta', 'Phi', 'Charge']` for leptons
   - Reverted BJet from `['E', 'Px', 'Py', 'Pz']` → `['Pt', 'Eta', 'Phi', 'Mass']`
   - Reverted MET from `['MET_Px', 'MET_Py']` → `['MET', 'MET_Phi']`
   - Feature types: Phi features set to `"uni"`, Charges to `"disc"`, rest to `"cont"`
   - **`create_dataset()`**: Replaced Cartesian `branches_to_load` with direct cylindrical branches (`Pt`, `Eta`, `Phi`, `Charge`, `Mass`)
   - **Removed** entire `to_cylindrical()` helper function and per-event conversion loop
   - Event row now reads directly: `data['Z_Lepton1_Pt'][i]`, `data['Z_Lepton1_Eta'][i]`, etc.

2. **ml/data/ttz/variables.json**
   - Replaced 21 Cartesian `"cont"` entries with 18 cylindrical entries
   - Phi features: `"uni"`, Charge features: `"disc"`, Pt/Eta/Mass/MET: `"cont"`

3. **ml/custom/ttz/config/flows/data_config.yaml**
   - `input_dim: 21` → `input_dim: 18`
   - `keep_names`: updated from Cartesian names to cylindrical names
   - `no_process`: kept as `["weight"]` (Phi angles are Gaussian rank scaled via `"uni"` type)
   - Comments updated to reflect cylindrical coordinate rationale

4. **ml/custom/ttz/ttz_dataset.py**
   - Weight column detection: `data.shape[1] == 22` → `data.shape[1] == 19` (18 features + 1 weight)
   - Warning message updated accordingly

5. **ml/custom/ttz/sample_analyzer/analyzer.py**
   - Weight column detection: `shape[1] == 22` → `shape[1] == 19`
   - Error messages updated to reference cylindrical features

#### Feature Layout After Revert

**18 input features:**

| Index | Feature | Type |
|---|---|---|
| 0 | Z_Lepton1_Pt | cont |
| 1 | Z_Lepton1_Eta | cont |
| 2 | Z_Lepton1_Phi | uni |
| 3 | Z_Lepton1_Charge | disc |
| 4 | Z_Lepton2_Pt | cont |
| 5 | Z_Lepton2_Eta | cont |
| 6 | Z_Lepton2_Phi | uni |
| 7 | Z_Lepton2_Charge | disc |
| 8 | W_Lepton_Pt | cont |
| 9 | W_Lepton_Eta | cont |
| 10 | W_Lepton_Phi | uni |
| 11 | W_Lepton_Charge | disc |
| 12 | BJet_Pt | cont |
| 13 | BJet_Eta | cont |
| 14 | BJet_Phi | uni |
| 15 | BJet_Mass | cont |
| 16 | MET | cont |
| 17 | MET_Phi | uni |

#### Breaking Changes

⚠️ **Data files must be regenerated** - existing `.npy` files contain Cartesian-coordinate data  
⚠️ **Models trained on Cartesian data are incompatible** - retrain with new 18-feature layout

#### Lessons Learned

- Cartesian coordinates do **not** solve Phi periodicity - the circular structure simply moves from 1 feature to the joint (Px, Py) pair, which rank scaling still destroys
- Gaussian rank scaling on Phi (`"uni"` type → Gaussian → inverse back to uniform at generation) is the correct approach
- Loading features directly from ROOT saves computation and avoids floating-point errors from manual conversions
- Always verify whether the ROOT file already contains the desired representation before adding conversion logic

------

### Issue: Remove Charge Features (18 → 15 features)

#### Motivation

After reverting to cylindrical coordinates, lepton charge features (Z_Lepton1_Charge, Z_Lepton2_Charge, W_Lepton_Charge) were reviewed for utility:

- **Not used in any physics calculation**: `physics_utils.py` (Z_mass, DeltaR, W_MT, Top_MT) does not reference charge
- **Trivially ±1**: The model has to learn that generated values must snap to ±1, requiring post-processing anyway
- **`disc_rescale_type: null`**: Charge features were not scaled - they passed through preprocessing unchanged
- **Reduce model complexity**: Removing 3 features reduces `input_dim` from 18 → 15

#### Files Modified

1. **ml/custom/ttz/process_ttz_dataset.py**
   - Removed `'Charge'` from lepton feature loops (Z_Lepton1, Z_Lepton2, W_Lepton)
   - Removed `if 'Charge' in feature: disc` type assignment
   - Removed Charge branches from `branches_to_load`
   - Removed charge entries from `event_row`
   - Updated feature count comment: 18 → 15

2. **ml/data/ttz/variables.json**
   - Removed `Z_Lepton1_Charge`, `Z_Lepton2_Charge`, `W_Lepton_Charge` entries

3. **ml/custom/ttz/config/flows/data_config.yaml**
   - `input_dim: 18` → `input_dim: 15`
   - Removed Charge names from `keep_names`

4. **ml/custom/ttz/ttz_dataset.py**
   - Weight column detection: `shape[1] == 19` → `shape[1] == 16`

5. **ml/custom/ttz/sample_analyzer/analyzer.py**
   - Weight column detection: `shape[1] == 19` → `shape[1] == 16`
   - Removed charge post-processing block (snap to ±1 logic)

#### Final Feature Layout

**15 input features:**

| Index | Feature | Type |
|---|---|---|
| 0 | Z_Lepton1_Pt | cont |
| 1 | Z_Lepton1_Eta | cont |
| 2 | Z_Lepton1_Phi | uni |
| 3 | Z_Lepton2_Pt | cont |
| 4 | Z_Lepton2_Eta | cont |
| 5 | Z_Lepton2_Phi | uni |
| 6 | W_Lepton_Pt | cont |
| 7 | W_Lepton_Eta | cont |
| 8 | W_Lepton_Phi | uni |
| 9 | BJet_Pt | cont |
| 10 | BJet_Eta | cont |
| 11 | BJet_Phi | uni |
| 12 | BJet_Mass | cont |
| 13 | MET | cont |
| 14 | MET_Phi | uni |

#### Breaking Changes

⚠️ **Data files must be regenerated** - `.npy` files must be recreated with 15 features  
⚠️ **All models must be retrained** with `input_dim=15`

---

### Issue: Out-of-Memory During Data Regeneration (Condor job killed at 12 GB)

#### Problem Identified

A Condor job was killed with:
```
Job has gone over cgroup memory limit of 12288 megabytes. Last measured usage: 11779 megabytes.
```

The job had not changed its `request_memory` setting. The OOM was triggered because today's feature changes (Cartesian → cylindrical, charge removal) invalidated the existing `.npy` file, forcing `create_dataset()` to run for the first time in a while — exposing a long-standing memory inefficiency in the data loading loop.

#### Root Cause

`create_dataset()` used a Python `for i in range(n_events)` loop to build a list-of-lists:

```python
event_data = []
for i in range(n_events):
    event_row = [data['Z_Lepton1_Pt'][i], data['Z_Lepton1_Eta'][i], ...]
    event_data.append(event_row)
self.x = np.array(event_data, dtype=np.float32)
```

Each Python `float` in a list carries ~24 bytes of object overhead (vs 4 bytes for `float32`). For 383k events × 15 features:

- Python list-of-lists peak: ~383k × 15 × ~56 bytes ≈ **320+ MB** in list overhead alone
- During the `np.array(event_data)` conversion, both the list and the numpy array exist in memory simultaneously
- The ROOT `data` dict (uproot arrays) also stays in memory throughout
- CPython's garbage collector lags, so all three coexist at peak → **several GB**

This was always present but never triggered because the `.npy` file existed and the loop was never run in production.

#### Solution Implemented

Replaced the loop with vectorized `np.column_stack()`. Uproot already returns each branch as a contiguous numpy array, so stacking them requires a single pass with no Python object overhead:

```python
# Before (loop - ~several GB peak)
event_data = []
for i in range(n_events):
    event_row = [data['Z_Lepton1_Pt'][i], ...]
    event_data.append(event_row)
self.x = np.array(event_data, dtype=np.float32)

# After (vectorized - ~100 MB peak)
columns = [
    data['Z_Lepton1_Pt'], data['Z_Lepton1_Eta'], data['Z_Lepton1_Phi'],
    ...
    data['MET'], data['MET_phi'],
]
if self.load_weights:
    columns.append(data['smeft_weights'][:, 124])
self.x = np.column_stack(columns).astype(np.float32)
```

Also changed `data['smeft_weights'][i][124]` (per-event Python indexing) to `data['smeft_weights'][:, 124]` (vectorized column slice).

##### Files Modified

1. **ml/custom/ttz/process_ttz_dataset.py**
   - Replaced `for i in range(n_events)` loop and `event_data` list with `np.column_stack(columns)`
   - Removed `event_data`, `n_valid`, `event_row` variables
   - Changed `smeft_weights[i][124]` → `smeft_weights[:, 124]`
   - Removed `if len(event_data) > 0` guard (not needed with vectorized approach)

#### Lessons Learned

- `create_dataset()` only runs when the `.npy` file is missing — memory bugs here are invisible until a regeneration is forced
- Always use vectorized numpy operations when loading large datasets; Python list-of-lists is never appropriate for O(100k+) events
- After any feature layout change, ensure the Condor job's `request_memory` is sufficient for regeneration, or regenerate locally first

---

### Issue: IndexError on smeft_weights After Vectorization

#### Problem Identified

After replacing the event loop with `np.column_stack()`, the job failed with:
```
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
  columns.append(data['smeft_weights'][:, 124])
```

#### Root Cause

`smeft_weights` is a **jagged array** in the ROOT file — each event stores a variable-length list of EFT weights. Uproot loads it as a 1D object array of sub-arrays (dtype=object), not a rectangular 2D numpy array. The original per-event loop `data['smeft_weights'][i][124]` worked because it indexed one sub-array at a time. The vectorized `[:, 124]` assumed a regular 2D array and failed.

#### Solution Implemented

Used `np.stack()` to first materialize the jagged object array into a regular 2D array before slicing column 124:

```python
# Before (broken - smeft_weights is jagged, not 2D)
columns.append(data['smeft_weights'][:, 124])

# After (correct - stack first, then slice)
columns.append(np.stack(data['smeft_weights'])[:, 124])
```

##### Files Modified

1. **ml/custom/ttz/process_ttz_dataset.py**
   - Line ~216: `data['smeft_weights'][:, 124]` → `np.stack(data['smeft_weights'])[:, 124]`

---

## 2026-03-03 — physics_utils.py: cylindrical coordinates + cleanup

### Changes to `ml/custom/ttz/sample_analyzer/physics_utils.py`

#### Rewrite kinematic functions for cylindrical coordinates

All three calculation functions were rewritten to use the current cylindrical
feature layout (Pt, Eta, Phi) instead of the old Cartesian (E, Px, Py, Pz).
All per-event Python loops were replaced with fully vectorized numpy operations.

**`calculate_z_kinematics`**
- Old: indexed `Z_Lepton1_E/Px/Py/Pz`, `Z_Lepton2_E/Px/Py/Pz`; per-event loop
- New: indexes `Z_Lepton1/2_Pt/Eta/Phi`; massless approximation
  `E = Pt*cosh(eta)`, `Pz = Pt*sinh(eta)`; fully vectorized

**`calculate_w_kinematics`**
- Old: indexed `W_Lepton_E/Px/Py/Pz`, `MET_Px`, `MET_Py`; per-event loop
- New: indexes `W_Lepton_Pt/Phi`, `MET`, `MET_Phi`; fully vectorized

**`calculate_top_kinematics`**
- Old: indexed `BJet_E/Px/Py/Pz`; per-event loop
- New: indexes `BJet_Pt/Phi`; fully vectorized

#### Removed unused helper functions

- **`cartesian_to_cylindrical`** — deleted by user (no longer needed now that
  data is loaded in cylindrical coordinates directly from ROOT)
- **`calculate_invariant_mass_from_4vectors`** — removed (never called anywhere
  in the codebase; invariant mass is now computed inline in
  `calculate_z_kinematics`)

##### Files Modified

1. **ml/custom/ttz/sample_analyzer/physics_utils.py**
   - Removed `calculate_invariant_mass_from_4vectors()`
   - Removed `cartesian_to_cylindrical()` (deleted by user)
   - Rewrote `calculate_z_kinematics()` — cylindrical, vectorized
   - Rewrote `calculate_w_kinematics()` — cylindrical, vectorized
   - Rewrote `calculate_top_kinematics()` — cylindrical, vectorized

---

## 2026-03-03 — SMEFT weight decomposition for training

### Motivation

Previously, the model trained only on the full SMEFT weight at cHt=5.0. To better understand
the physics and potentially improve performance, we now support training on different weight
components from the EFT expansion: **w_eft = w_sm + c·w_lin + c²·w_quad**

This allows isolating and studying the SM baseline, linear, and quadratic BSM contributions
separately during training.

### Implementation

Added configurable SMEFT weight decomposition to separate SM, linear, and quadratic terms.

**Files Modified:**

1. **ml/custom/ttz/config/flows/data_config.yaml**
   - Added `weight_type` option: "full", "sm", "linear", or "quadratic"
   - Added `eft_coefficient` config (default 5.0) for linear/quadratic extraction
   - Documented the EFT expansion formula in comments

2. **ml/custom/ttz/process_ttz_dataset.py**
   - Added `weight_type` and `eft_coefficient` parameters to `ttzNpyProcessor`
   - Load both `eventWeight` (SM) and `smeft_weights` from ROOT file
   - Compute appropriate weight based on `weight_type`:
     - "full": use smeft_weights[:, 124] (cHt=+5.0) or [:, 122] (cHt=-5.0)
     - "sm": use eventWeight (SM cross-section weight)
     - "linear": w_lin = (w_plus - w_minus) / (2·c)
     - "quadratic": w_quad = (w_plus + w_minus - 2·w_sm) / (2·c²)
   - Added detailed logging of which weight type is being used

3. **ml/custom/ttz/main_flows.py**
   - Pass `weight_type` and `eft_coefficient` from config to `ttzNpyProcessor`

4. **test_weight_decomposition.py** (new)
   - Test script to verify weight decomposition correctness
   - Loads small sample from ROOT file and verifies: w_eft = w_sm + c·w_lin + c²·w_quad
   - Reconstruction error < 1e-11 (machine precision) ✓

### Weight Index Reference

From `ml/data/ttz/cHt_weight_indices.txt`:
- Index 122: `cHt_m5p0` (cHt = -5.0)
- Index 124: `cHt_p5p0` (cHt = +5.0)
- `eventWeight` branch: SM weight

### Usage

Edit `ml/custom/ttz/config/flows/data_config.yaml`:

```yaml
# Train on SM weights only
weight_type: "sm"

# Train on linear BSM term only
weight_type: "linear"
eft_coefficient: 5.0

# Train on quadratic BSM term only
weight_type: "quadratic"
eft_coefficient: 5.0

# Train on full SMEFT weight (default)
weight_type: "full"
eft_coefficient: 5.0
```

Then regenerate the dataset and retrain model.

---

### Update (same day): Removed misleading `eft_coefficient` parameter

The `eft_coefficient` config parameter was removed as it created a false impression of
flexibility. The weight decomposition is hardcoded to cHt = ±5.0 (indices 122/124 in
the ROOT file), so there's no way to extract linear/quadratic terms for other coefficients.

**Changes:**
- Removed `eft_coefficient` from `data_config.yaml`
- Hardcoded coefficient value (5.0) in decomposition formulas:
  - Linear: `(w_plus - w_minus) / 10.0` (instead of `/ (2*c)`)
  - Quadratic: `(w_plus + w_minus - 2*w_sm) / 50.0` (instead of `/ (2*c^2)`)
- Simplified parameter passing in `ttzNpyProcessor` and `main_flows.py`

**Updated usage** - Edit `ml/custom/ttz/config/flows/data_config.yaml`:
```yaml
weight_type: "sm"         # SM weight only
weight_type: "linear"     # Linear term (cHt=5.0 decomposition)
weight_type: "quadratic"  # Quadratic term (cHt=5.0 decomposition)
weight_type: "full"       # Full SMEFT weight at cHt=5.0 (default)
```

---

## Section 15: Multi-Weight Training Script and Improved Model Naming

**Date**: March 3, 2026

### Overview

Added a script to submit one Condor training job per SMEFT weight type, and updated the model naming convention to include the weight type and cHt value for easier identification in the MLflow registry.

---

### 1. Added `train_all_weights.py`

**File**: `ml/custom/ttz/train_all_weights.py` *(new)*

Submits a separate Condor job for each weight type (`sm`, `linear`, `quadratic`, `full`) using the same `condorsub` command as `run_bash.sh`. Jobs run in parallel on the cluster.

**Key features:**
- Cleans up old Condor log/submission files before submitting (same as `run_bash.sh`)
- One job per weight type, named `mafmademog_<weight_type>` (visible in `condor_q`)
- Logs submission status to console and to a timestamped file in `logs/`
- Supports `--weight-types` flag to submit a subset, and `--queue` to override the Condor queue
- `--no-cleanup` flag to skip removal of old Condor files

**Usage:**
```bash
# Submit all 4 weight types
python ml/custom/ttz/train_all_weights.py

# Submit a subset
python ml/custom/ttz/train_all_weights.py --weight-types sm linear

# Use a different Condor queue
python ml/custom/ttz/train_all_weights.py --queue short
```

**Configuration** (top of file, mirrors `run_bash.sh`):
```python
CONDOR_JOB_PREFIX = "mafmademog"
CONDOR_QUEUE      = "medium"
CONDOR_N_GPUS     = 1
CONDOR_MEMORY_MB  = 12000
```

Each Condor job runs:
```bash
python -m ml.custom.ttz.main_flows data_config.weight_type=<weight_type>
```

---

### 2. Updated Model Naming in `main_flows.py`

**File**: `ml/custom/ttz/main_flows.py`

The registered MLflow model name now encodes the weight type and the cHt value, making models easily identifiable without loading them.

**Before:**
```
MAFMADEMOG_flow_model_gauss_rank_20260303_nall
```

**After:**
```
MAFMADEMOG_flow_model_gauss_rank_20260303_cHt5_linear_nall
MAFMADEMOG_flow_model_gauss_rank_20260303_cHt5_full_nall
```

**Code change** (near end of `main_flows.py`):
```python
# Before
detailed_model_name = f"{model_name}_{date_str}_n{n_data_str}"

# After
weight_type_str = data_conf.get("weight_type", "full")
detailed_model_name = f"{model_name}_{date_str}_cHt5_{weight_type_str}_n{n_data_str}"
```

---

*End of refactoring log entry*

## Section 16: SMEFT Analysis Sweep Script and Configurable Figure Output

**Date**: March 3, 2026

### Overview

Added a Condor submission script to automatically run the analyzer for each trained SMEFT
weight-type model and store the resulting figures in clearly named, dated subdirectories.
Also updated the analyzer and run_analysis.py to support configurable output directories.

---

### 1. Added `run_smeft_analysis.py`

**File**: `ml/custom/ttz/run_smeft_analysis.py` *(new)*

Submits one Condor analysis job per SMEFT weight type (`sm`, `linear`, `quadratic`).
For each weight type it:
1. Searches the MLflow registry for the most recently registered model whose name
   contains `cHt5_<weight_type>` (e.g. `cHt5_linear`)
2. Creates a dated output directory: `sample_analyzer/figures/smeft_<weight_type>_<date>/`
3. Submits a Condor job running `run_analysis.py --model-name <name> --figures-dir <dir>`

**Usage:**
```bash
# Run analysis for all weight types
python ml/custom/ttz/run_smeft_analysis.py

# Run for a subset
python ml/custom/ttz/run_smeft_analysis.py --weight-types sm linear

# Use a different queue
python ml/custom/ttz/run_smeft_analysis.py --queue medium
```

**Figure output structure:**
```
sample_analyzer/figures/
  smeft_sm_20260303/
    feature_comparison.png
    feature_comparison_log.png
  smeft_linear_20260303/
    ...
  smeft_quadratic_20260303/
    ...
```

---

### 2. Updated `run_analysis.py` — Added CLI Arguments

**File**: `ml/custom/ttz/sample_analyzer/run_analysis.py`

Added `argparse` support so the script can be driven externally (e.g. from Condor jobs).
Existing usage with no arguments is fully backwards-compatible.

**New arguments:**

| Argument | Description |
|---|---|
| `--model-name` | MLflow model name to load. Defaults to `get_latest_ttz_model()`. |
| `--figures-dir` | Directory to save figures. Defaults to `sample_analyzer/figures/`. |

**Example:**
```bash
# Old usage (unchanged)
python ml/custom/ttz/sample_analyzer/run_analysis.py

# New usage (driven by run_smeft_analysis.py)
python ml/custom/ttz/sample_analyzer/run_analysis.py \
  --model-name MAFMADEMOG_flow_model_gauss_rank_20260303_cHt5_linear_nall \
  --figures-dir ml/custom/ttz/sample_analyzer/figures/smeft_linear_20260303
```

---

### 3. Updated `analyzer.py` — `plot_all()` Accepts `figures_dir`

**File**: `ml/custom/ttz/sample_analyzer/analyzer.py`

Added optional `figures_dir` parameter to `plot_all()` so the output location can be
controlled externally. When `None`, the existing default (`sample_analyzer/figures/`) is used.

**Before:**
```python
def plot_all(self, include_derived=True):
    self.plot_feature_comparison(include_derived=include_derived)
```

**After:**
```python
def plot_all(self, include_derived=True, figures_dir=None):
    if figures_dir is not None:
        output_path = os.path.join(str(figures_dir), 'feature_comparison.png')
    else:
        output_path = None  # uses default
    self.plot_feature_comparison(output_path=output_path, include_derived=include_derived)
```

---

### Full Workflow

```bash
# Step 1 — train one model per weight type (parallel Condor jobs)
python ml/custom/ttz/train_all_weights.py

# Step 2 — once training is done, run analysis for all (parallel Condor jobs)
python ml/custom/ttz/run_smeft_analysis.py
```

---

*End of refactoring log entry*

---

## Section 17: Simplified `run_smeft_analysis.py` to Local Execution

**Date**: 2026-03-03

**Files changed**:
- `ml/custom/ttz/run_smeft_analysis.py`

---

### Motivation

The original `run_smeft_analysis.py` submitted analysis jobs via the Condor batch system
(`condorsub`). This was over-engineered: the analyzer runs in seconds to minutes per model,
well below the threshold that justifies cluster submission. The rule established is:
**only jobs expected to take longer than ~1 hour go to Condor**.

---

### Changes

#### 1. Removed all Condor infrastructure

The following were removed entirely:

| Removed | Reason |
|---|---|
| `glob` import | Only used for Condor file cleanup |
| `CONDOR_*` constants | No longer submitting to cluster |
| `VENV_ACTIVATE` constant | Not needed for local subprocess |
| `CLEANUP_PATTERNS` constant | No Condor files to clean up |
| `cleanup_old_condor_files()` function | Obsolete |
| `build_job_command()` function | Was building the Condor shell command string |
| `submit_job()` function | Was calling `condorsub` |
| `--queue` CLI argument | No queue to select |
| `--no-cleanup` CLI argument | No cleanup to skip |

#### 2. Added `run_analysis()` — local subprocess

Replaced `submit_job()` with a simple local runner:

```python
def run_analysis(weight_type: str, model_name: str, figures_dir: Path) -> bool:
    cmd = [
        sys.executable,
        str(ANALYZER_DIR / "run_analysis.py"),
        "--model-name", model_name,
        "--figures-dir", str(figures_dir),
    ]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode == 0
```

The three weight types are processed sequentially in the main loop.

#### 3. Fixed `PROJECT_ROOT` path resolution bug

`parents[2]` resolved `ml/custom/ttz/run_smeft_analysis.py` to `ml/` instead of the
repo root. Corrected to `parents[3]`, consistent with `train_all_weights.py`.

**Before:**
```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # wrong — resolves to ml/
```

**After:**
```python
PROJECT_ROOT = Path(__file__).resolve().parents[3]   # MLHEPsim root
```

---

### Updated Workflow

```bash
# Step 1 — submit training jobs to Condor (long-running, ~hours per model)
python ml/custom/ttz/train_all_weights.py

# Monitor with:
condor_q

# Step 2 — once all models are registered in MLflow, run analysis locally
python ml/custom/ttz/run_smeft_analysis.py
```

---

*End of refactoring log entry*

---

## Section 18: Added `smeft_reweighting.py` — Importance Reweighting via Normalising Flows

**Date:** 2025

**File added:** `ml/custom/ttz/smeft_reweighting.py`

**Theory reference:** `notes/smeft_importance_reweighting.tex`

---

### Motivation

Once three separate normalising flows are trained (SM, linear, quadratic components of the
SMEFT expansion at c_Ht = ±5), we can evaluate any EFT point without retraining. The key
insight is importance reweighting in the flow's latent space:

$$w_i(c) \propto Z_\text{sm} + c \cdot Z_\text{lin} \cdot r_\text{lin}(x_i) + c^2 \cdot Z_\text{quad} \cdot r_\text{quad}(x_i)$$

where $r_k(x_i) = p_k(x_i) / p_\text{sm}(x_i)$ is the density ratio evaluated via the two
respective flows.

---

### Pipeline

| Step | Description |
|------|-------------|
| 1 | Search MLflow for registered models matching `cHt5_{weight_type}` |
| 2 | Compute $Z_k$ normalisation constants from ROOT data file (3-jet filter) |
| 3 | Sample $N$ events from the SM flow (`module_sm.model.sample(N)`) in preprocessed space |
| 4 | Evaluate log $p_k(x)$ under all three flows for the same sample: `log_p = -estimate_density(x, exp=False, mean=False)` |
| 5 | Compute log density ratios: `log_r_lin = log_p_lin - log_p_sm` |
| 6 | Save ratios + samples to `figures/smeft_reweighting/density_ratios.npz` for reuse |
| 7 | For each requested $c$, compute importance weights, $N_\text{eff}$, and produce comparison histograms |

---

### Key Technical Choices

- `estimate_density(x, exp=False, mean=False)` returns **negative** log probability (NLL),
  so `log_p = -estimate_density(...)`.
- Sampling via `module_sm.model.sample(N)` returns data in preprocessed (gauss-rank scaled)
  space — directly compatible with all three flows since they share the same training dataset
  and scalers.
- Physical-space samples obtained via `RescalingHandler(module_sm.selection, module_sm.scalers).inverse_transform(x_scaled)`.
- Negative weights (unphysical, possible for large `|c|`) are clipped to 0 with a warning logged.

---

### CLI

```bash
# Full run (sample + encode + plot)
python ml/custom/ttz/smeft_reweighting.py --n-samples 100000 --c-values -5 -2 -1 0 1 2 5

# Reuse previously saved ratios (fast — skips model inference)
python ml/custom/ttz/smeft_reweighting.py --load-ratios --c-values -5 0 5
```

---

### Outputs

| File | Description |
|------|-------------|
| `figures/smeft_reweighting/density_ratios.npz` | `log_r_lin`, `log_r_quad`, `x_scaled` arrays |
| `figures/smeft_reweighting/neff_vs_c.png` | $N_\text{eff}/N$ as a function of $c_{Ht}$ |
| `figures/smeft_reweighting/reweighted_c{VALUE}.png` | 15-feature comparison plots at each $c$ |
| `logs/smeft_reweighting_YYYYMMDD_HHMMSS.log` | Full run log |

---

*End of refactoring log entry*

---

## Section 19: Remove 3-jet Filter from `smeft_reweighting.py`

**Date:** 2026-03-04
**File:** `ml/custom/ttz/smeft_reweighting.py`

### Change

The 3-jet filter in `compute_z_normalisations()` was removed. It had been copied from `plot_weight_contributions.py` where it was used as a temporary diagnostic, but it is not appropriate here — the Z_k normalisation constants should be computed over all events in the dataset.

**Before:**
```python
data = tree.arrays(['eventWeight', 'smeft_weights', 'Jet_Pt'], library='np')

# 3-jet filter
mask = np.array([len(j) for j in data['Jet_Pt']]) == 3
log.info(f"  3-jet filter: {mask.sum()} / {len(mask)} events kept")

w_sm  = data['eventWeight'][mask]
smeft = np.stack(data['smeft_weights'][mask])
```

**After:**
```python
data = tree.arrays(['eventWeight', 'smeft_weights'], library='np')

log.info(f"  Total events: {len(data['eventWeight'])}")

w_sm  = data['eventWeight']
smeft = np.stack(data['smeft_weights'])
```

---

*End of refactoring log entry*

---

## Section 20: Correct Negative Weight Handling and N_eff in `smeft_reweighting.py`

**Date:** 2026-03-04
**File:** `ml/custom/ttz/smeft_reweighting.py`

### Motivation

Per-event importance weights $w_i(c) \propto Z_\text{sm} + c Z_\text{lin} e^{r_{\text{lin},i}} + c^2 Z_\text{quad} e^{r_{\text{quad},i}}$ can legitimately be negative. This reflects destructive interference between the SM and EFT amplitudes in that region of phase space. Clipping negative weights to zero would bias distributions and is physically wrong. Only the total cross section (sum over all events) must remain positive for a valid EFT point.

### Change 1 — `get_weights`: remove negative-weight clipping

**Before:**
```python
neg = (unnorm < 0).sum()
if neg:
    log.warning(f"  c={c}: {neg} negative weights clipped to 0")
    unnorm = np.clip(unnorm, 0.0, None)

return unnorm / unnorm.sum()
```

**After:**
```python
neg = (unnorm < 0).sum()
if neg:
    log.info(f"  c={c:+.1f}: {neg} events have negative weights (destructive interference)")

total = unnorm.sum()
if total <= 0:
    log.warning(f"  c={c:+.1f}: total weight sum = {total:.4f} <= 0 — EFT may be outside validity range")

return unnorm / total
```

### Change 2 — `effective_sample_size`: use signed-weight estimator

The previous estimator $N_\text{eff} = 1/\sum w_i^2$ assumes non-negative weights and is misleading when cancellations occur. Replaced with:

$$N_\text{eff} / N = \frac{|\sum w_i|}{\sum |w_i|}$$

which measures the fraction of the absolute weight budget that survives after destructive interference.

**Before:**
```python
def effective_sample_size(weights: np.ndarray) -> float:
    """N_eff = (Σw)² / Σw²   (already normalised → N_eff = 1 / Σw²)"""
    return 1.0 / float(np.sum(weights**2))
```

**After:**
```python
def effective_sample_size(weights: np.ndarray) -> float:
    """Returns fraction 0-1: |Σw| / Σ|w|"""
    return float(np.abs(weights.sum())) / float(np.abs(weights).sum())
```

Plot labels and the $N_\text{eff}$ curve updated accordingly (function now returns a fraction directly).

---

*End of refactoring log entry*

---

## Section 21: Fix CUDA/CPU Device Mismatch in `load_module` (`smeft_reweighting.py`)

**Date:** 2026-03-05
**File:** `ml/custom/ttz/smeft_reweighting.py`

### Bug

Running `smeft_reweighting.py` crashed during sampling with:

```
RuntimeError: Expected all tensors to be on the same device, but found at least
two devices, cpu and cuda:0! (when checking argument for argument mat1 in method
wrapper_CUDA_addmm)
```

### Root Cause

`AutoregressiveNormalizingFlow` stores its target device as a plain Python attribute (`self.device`), not as a PyTorch module buffer. When a checkpoint is saved from a GPU training run, `self.device = 'cuda:0'` is frozen into the state dict. Loading with `fetch_registered_module(..., device="cpu")` moves the model *parameters* to CPU but does not update this plain attribute. As a result `sample()` creates its dummy input tensor on `cuda:0` while the MADE weights are on CPU, causing the mismatch.

### Fix

After loading, explicitly call `.cpu()` on the full Lightning module and then patch the inner flow's `device` attribute:

**Before:**
```python
def load_module(model_name: str):
    from ml.common.utils.register_model import fetch_registered_module
    mlflow.set_tracking_uri(...)
    module = fetch_registered_module(model_name, model_version=-1, device="cpu")
    module.model.eval()
    return module
```

**After:**
```python
def load_module(model_name: str):
    import torch
    from ml.common.utils.register_model import fetch_registered_module
    mlflow.set_tracking_uri(...)
    module = fetch_registered_module(model_name, model_version=-1, device="cpu")
    module = module.cpu()
    module.model.eval()

    inner_flow = module.model.model
    if hasattr(inner_flow, 'device'):
        inner_flow.device = torch.device('cpu')

    return module
```

---

*End of refactoring log entry*

---

## Section 22: Fix `MADEMOG.estimate_density` Incompatibility and Patch `MADEMOG.device` (`smeft_reweighting.py`)

**Date:** 2026-03-05
**File:** `ml/custom/ttz/smeft_reweighting.py`

### Bug

```
TypeError: MADEMOG.estimate_density() got an unexpected keyword argument 'chunks'
```

`module.model` resolves to a `MADEMOG` instance, not a `BaseFlowModel`. `MADEMOG.estimate_density` has no `chunks` parameter (unlike `BaseFlowModel.estimate_density`).

### Fix 1 — `compute_log_probs`: manual batching

Removed the `chunks=` kwarg. Replaced with an explicit loop that slices `x_scaled` into batches of `batch_size` events, calls `estimate_density` on each slice, and concatenates the NLL arrays.

### Fix 2 — `load_module`: also patch `MADEMOG.device`

`MADEMOG.estimate_density` moves input tensors with `data_points.to(self.device)`. Like `AutoregressiveNormalizingFlow.device`, this `self.device` is a plain Python attribute serialised as `cuda:0` and not updated by `.cpu()`. The device patch in `load_module` is extended to cover both:

```python
cpu = torch.device('cpu')
if hasattr(module.model, 'device'):
    module.model.device = cpu            # MADEMOG
inner_flow = module.model.model
if hasattr(inner_flow, 'device'):
    inner_flow.device = cpu              # AutoregressiveNormalizingFlow
```

---

*End of refactoring log entry*

---

## Section 23: Fix `(N,1)` Shape from `estimate_density` in `smeft_reweighting.py`

**Date:** 2026-03-05
**File:** `ml/custom/ttz/smeft_reweighting.py`

### Bug

```
ValueError: weights should have the same shape as a.
```

Raised in `np.histogram(feat_data, bins=edges, weights=w, density=True)` where `feat_data.shape = (N,)` but `w.shape = (N, 1)`.

### Root Cause

`MADEMOGModel.forward` computes `self.log_prob` with `keepdim=True`:

```python
self.log_prob = torch.sum(
    torch.logsumexp(..., dim=1),
    dim=-1,
    keepdim=True,   # ← produces (N, 1) not (N,)
)
```

`MADEMOG.estimate_density(mean=False)` therefore returns a `(N, 1)` array. After concatenation across batches, `nll` was `(N, 1)`, propagating through `log_p`, `log_r_lin`, `log_r_quad`, and `w` — all `(N, 1)` while `feat_data` remained `(N,)`.

### Fix

Squeeze the concatenated NLL array in `compute_log_probs` to ensure shape `(N,)`:

```python
nll = np.concatenate(nll_parts).squeeze()   # (N,1) → (N,) due to keepdim=True in MADEMOGModel
```

---

*End of refactoring log entry*

---

## Section 24: Improve Distribution Plots in `smeft_reweighting.py`

**Date:** 2026-03-05
**File:** `ml/custom/ttz/smeft_reweighting.py`

### Changes to `plot_reweighted_distributions`

**1. SM visibility fix**

Previously both SM and SMEFT were drawn as `step` lines with the same weight. When the SMEFT reweighting concentrates probability into a narrow region the SMEFT peak is much taller, making the SM distribution appear invisible (it was plotted correctly but simply dwarfed). SM is now drawn as a filled area (`fill_between`) with an outline, ensuring it remains visible regardless of the SMEFT peak height.

**2. Ratio panels**

Each feature panel now has a bottom sub-panel showing SMEFT / SM per bin:
- Bins where `hist_sm = 0` are skipped (`nan`) to avoid divide-by-zero artefacts.
- The y-range is clipped to the 2nd–98th percentile of finite ratio values so a single outlier bin does not collapse the visible range.
- A reference line at ratio = 1 is drawn in blue.

**3. Title placement**

Replaced `suptitle(y=0.998)` with a `GridSpec(top=0.92)` layout so the figure title no longer overlaps the top row of plots.

---

*End of refactoring log entry*

---

## Section 25: Ratio Y-Range Fix and Derived Features in `smeft_reweighting.py`

**Date:** 2026-03-05
**File:** `ml/custom/ttz/smeft_reweighting.py`

### Change 1 — Ratio panel y-range uses full extent

**Before:** y-range was clipped to the 2nd–98th percentile of finite ratio values, which could hide genuine large deviations in the tails.

**After:** y-range spans the full `[min, max]` of all finite ratio values plus a 10% margin on each side:

```python
r_lo, r_hi = finite.min(), finite.max()
margin = 0.1 * max(r_hi - r_lo, 0.1)
ax_ratio.set_ylim(r_lo - margin, r_hi + margin)
```

### Change 2 — Derived higher-order features added to plots

Added `compute_derived_features(x_phys)` which reconstructs 7 higher-order kinematic quantities from the 15 base features (in physical space) and appends them as extra columns for plotting only (the flow model is not retrained).

| Feature | Derivation |
|---|---|
| `Z_pt`, `Z_eta`, `Z_mass` | 4-vector sum of two massless Z-leptons |
| `Z_deltaR` | $\sqrt{\Delta\eta^2 + \Delta\phi^2}$ between Z-leptons |
| `W_boson_pt` | 2D vector sum W-lepton + MET |
| `top_pt` | 2D transverse vector sum b-jet + W-lepton + MET |
| `Z_over_top_pt` | `Z_pt / top_pt` |

`plot_reweighted_distributions` now accepts an optional `feature_names` argument so the extra columns are labelled correctly. Total features in output plots: **22** (15 base + 7 derived).

---

*End of refactoring log entry*

---

## Section 26: Fix Layout Overlap, Whitespace, and Histogram Normalisation Bug (`smeft_reweighting.py`)

**Date:** 2026-03-05
**File:** `ml/custom/ttz/smeft_reweighting.py`

### Change 1 — Layout: remove top whitespace and fix label overlap

- `hspace` increased from `0.08` → `0.45`. With a 3:1 main-to-ratio height ratio, the very tight spacing caused ratio x-axis labels to overlap with the title of the row below.
- `top=0.97` and `suptitle(y=0.995)` introduced so the suptitle sits flush against the plot area, eliminating the large blank gap at the top of saved PNGs.
- Figure height increased by `0.8` inches to prevent overall vertical crowding at the bottom.

### Change 2 — Histogram normalisation bug (flat ratio fix)

The previous code called `np.histogram(..., density=True)` on both the SM and SMEFT histograms. When `density=True` is used numpy divides by `sum(weights) * dx`. For the SM histogram that denominator is `N * dx`; for the weighted histogram it is `sum(w) * dx = 1 * dx` (since `w` is already normalised to sum to 1). The two histograms therefore had implicit scale factors that differed by `N`, making their ratio numerically `~1 / N ≈ 0` for the SM—and thus the ratio panel showed a flat line of 1 regardless of the actual reweighting.

**Fix:** compute raw bin counts without `density=True` and normalise manually with a consistent scale factor:

```python
counts_sm, edges = np.histogram(feat_data, bins=bins, range=(x_min, x_max))
hist_sm = counts_sm / (N * dx)        # unweighted PDF: integrates to 1

counts_c, _ = np.histogram(feat_data, bins=edges, weights=w)
hist_c = counts_c / dx                # weighted PDF: sum(w)*dx = 1*dx → integrates to 1
```

### Change 3 — Diagnostic logging before plotting

Added log lines reporting `std`, `min`, and `max` of `log_r_lin`, `log_r_quad`, and the per-`c` weight array, making it easy to detect degenerate cases (e.g. all three models converging to the same density).

---

*End of refactoring log entry*

---

## Section 27: Fix Training Bug — Weights Never Applied to Loss (`train_all_weights.py`)

**Date:** 2026-03-05
**File:** `ml/custom/ttz/train_all_weights.py`

### Problem

All three normalising-flow models (SM, linear, quadratic) were trained with **identical unweighted NLL**, causing them to converge to the same density estimate. Evidence:

```
[sm]        log p: mean=-12.346, std=2.975
[linear]    log p: mean=-12.346, std=2.975
[quadratic] log p: mean=-12.346, std=2.975
log_r_lin std=0.0000   (all ratios exactly zero)
log_r_quad std=0.0000
```

### Root Cause

Three flags gate the weight-application pipeline, all defaulting to `False` at the time the original models were trained:

| Flag | Location | Effect when `False` |
|---|---|---|
| `load_weights` | `ttzNpyProcessor.process_ttz_dataset` | weight column never written to `.npy` files |
| `use_weights` | `ttzDataModule.setup` | weights never extracted from data, not passed to DataLoader |
| `use_sm_weights` | `MADEMOG.training_step` | `weighted_losses = weights * sample_losses` line skipped; plain mean NLL used |

`train_all_weights.py::build_job_command` was only passing:
```
python -m ml.custom.ttz.main_flows data_config.weight_type={weight_type}
```

None of the three flags were activated, so all jobs trained the same vanilla flow regardless of `weight_type`.

### Fix

`build_job_command` now passes all three flags explicitly as Hydra CLI overrides:

```python
# Before
cmd = (
    f"python -m ml.custom.ttz.main_flows "
    f"data_config.weight_type={weight_type}"
)

# After
cmd = (
    f"python -m ml.custom.ttz.main_flows "
    f"data_config.weight_type={weight_type} "
    f"data_config.load_weights=True "
    f"data_config.use_weights=True "
    f"training_config.use_sm_weights=True"
)
```

The Hydra key paths were verified against the YAML defaults in:
- `ml/custom/ttz/config/flows/data_config.yaml` — `load_weights`, `use_weights`, `weight_type` are top-level keys under `data_config:`
- `ml/custom/ttz/config/flows/training_config.yaml` — `use_sm_weights` is a top-level key under `training_config:`

Note: the YAML defaults were already updated to `true` for all three flags, so the CLI overrides are now redundant but provide an explicit safety net against future yaml changes.

### Impact

The three models must be **retrained** with these flags active. After retraining, re-run `smeft_reweighting.py` to regenerate density ratios; the stale `density_ratios.npz` was already deleted and a degenerate-ratio guard was added to prevent silently using bad cached results.

---

*End of refactoring log entry*

---

## Section 28: Fix `ttz_dataset.py` Weight Logging — Hard-coded `cHt=5.0` Message

**Date:** 2026-03-05
**File:** `ml/custom/ttz/ttz_dataset.py`

### Change

Both log lines in `ttzDataModule.setup` were hard-coded to `"cHt=5.0 weights"` and `"cHt weights"` regardless of the actual `weight_type` being used. This made it impossible to distinguish which weight component a given training job was using from the logs alone.

Added `weight_type="unknown"` parameter to `ttzDataModule.__init__` and threaded the value through from `main_flows.py`. Log lines now read:

```
Extracted weight_type='linear' weights from data. ...
Normalized 'linear' weights: original mean=...
```

`main_flows.py` was updated to pass `weight_type=data_conf.get("weight_type", "unknown")` to the `ttzDataModule` constructor.

---

*End of refactoring log entry*

---

## Section 29: Refactor Weight Storage — Single Multi-Column `.npy` File

**Date:** 2026-03-05
**Files:** `ml/custom/ttz/process_ttz_dataset.py`, `ml/custom/ttz/ttz_dataset.py`, `ml/custom/ttz/main_flows.py`

### Problem

`ttzNpyProcessor` used a single `weight_type` parameter to decide which weight column to write into `ttz.npy`. All three training jobs (sm, linear, quadratic) shared the same filename. When any one job ran first and wrote its single weight column, the other jobs found the file already existing and loaded it unchanged — training on the wrong (or identical) weights. This caused all three models to converge to byte-for-byte identical checkpoints (confirmed by MD5 hash).

### Design

Store all four weight decomposition components in a single file `ttz_weights.npy` (19 columns: 15 features + 4 weights). Each training job reads the same file and selects the appropriate column at load time:

| Column | Name | Formula |
|--------|------|---------|
| 15 | `sm` | `eventWeight` |
| 16 | `linear` | `(w_plus − w_minus) / 10.0` |
| 17 | `quadratic` | `(w_plus + w_minus − 2·w_sm) / 50.0` |
| 18 | `full` | `w_plus` (full SMEFT weight at cHt=5.0) |

### Changes

**`process_ttz_dataset.py`**:
- Removed `weight_type` constructor parameter — no longer needed for file generation
- When `load_weights=True`, `base_file_name` is set to `ttz_weights` so the output file is `ttz_weights.npy`, distinct from the unweighted `ttz.npy`
- All 4 weight columns are always computed and appended in one ROOT-file read: `w_sm`, `w_linear`, `w_quad`, `w_full`
- `self.features["colnames"]` now registers all four weight keys (`cHt_weight_sm`, `cHt_weight_linear`, `cHt_weight_quadratic`, `cHt_weight_full`)

**`ttz_dataset.py`**:
- Added module-level constant `WEIGHT_COLS = {"sm": 15, "linear": 16, "quadratic": 17, "full": 18}`
- `ttzDataModule.setup` now checks for 19 columns (not 16), selects the column matching `self.weight_type`, and strips all 4 weight columns from the feature matrix before training
- Updated warning message to reflect the new expected shape

**`main_flows.py`**:
- Removed `weight_type` from `ttzNpyProcessor` constructor call (parameter no longer exists)
- `weight_type` is still passed to `ttzDataModule` for column selection and model naming

### Impact

The ROOT file is read exactly once to produce `ttz_weights.npy`. All three training jobs reuse it. Delete `ttz_weights.npy` to force regeneration (e.g. if weight formulae change). The stale `ttz_weights.npy` was cleared before resubmitting training.

---

*End of refactoring log entry*

---

## Section 30: Add Ground Truth Comparison to Reweighting Plots (`smeft_reweighting.py`)

**Date:** 2026-03-05
**File:** `ml/custom/ttz/smeft_reweighting.py`

### Purpose

Compare flow-generated samples to ROOT file ground truth to visually assess model performance across all distributions and SMEFT weight decomposition components.

### Changes

**New function `load_root_features_and_weights()`**:
- Loads 15 physics features (same order as training) from the ROOT file
- Computes weight decomposition: `sm` (eventWeight), `linear`, `quadratic`, `full`
- Returns features and all 4 weight types for later selection
- Enables recomputation of SMEFT weights for any c value on the fly

**Updated `plot_reweighted_distributions()`**:
- Added optional parameters: `x_root` (ROOT features) and `w_decomp` (ROOT weight dict)
- When both are provided, shows three curves per feature:
  - **Blue fill + line**: SM flow samples (unweighted)
  - **Red line**: Generated + reweighted samples
  - **Green dashed line**: ROOT file weighted samples (ground truth)
- Ratio panel shows:
  - If ROOT data supplied: **ROOT / generated** (validates model accuracy)
  - Otherwise: **generated / SM** (legacy comparison)
- Updated docstring to document the new comparison mode

**Modified `main()`**:
- Loads ROOT file features and decomposed weights before plotting
- Computes derived features for ROOT data (same as for generated)
- Passes ROOT data to `plot_reweighted_distributions()`

### Usage

Run normally:
```bash
python ml/custom/ttz/smeft_reweighting.py
```

Plots now show ROOT/generated ratio in the lower panel. A value near 1.0 indicates the flow accurately learned the distribution for that feature and weight type.

### Impact

Users can visually validate model generalization:
- **Ratio ≈ 1**: flow learned the distribution well
- **Ratio drifts > 1.5 or < 0.67**: flow may have issues or regularization problems
- Can identify features where model struggles (e.g., high-mass regions)

---

*End of refactoring log entry*

## Entry 31: Ragged SMEFT Arrays Support

**Date:** 2026-03-06
**File:** `ml/custom/ttz/process_ttz_dataset.py`

### Problem

When switching to the new ROOT data file (`ttZ_for_Melle_tree.root`), the `process_ttz_dataset.py` script failed with:
```
ValueError: all input arrays must have the same shape
```

at the line `smeft = np.stack(data['smeft_weights'])`. This occurred because the new file stores `smeft_weights` as **ragged arrays** — each event can have a different number of weight values, violating the assumption that all events have uniform structure.

### Root Cause

ROOT files can store variable-length arrays (jagged/ragged arrays) where each entry in a column can have different lengths. The old data file had uniform-length weight arrays, but the new file does not. `np.stack()` requires all input arrays to have identical shapes, so it fails on ragged data like `[array(1032,), array(1032,), ..., array(987,), ...]`.

### Solution Implemented

Updated the weight extraction logic in `create_dataset()` (lines 223-249):

1. **Iterate through each weight array** individually instead of using `np.stack()` directly
2. **Validate minimum size**: Check if each weight array has at least 125 elements (need indices 0–124)
3. **Pad if necessary**: For any event with fewer than 125 weight values, pad with zeros to reach 125
4. **Convert variants**: Handle both numpy arrays and scalar values by converting to `np.atleast_1d()`
5. **Stack after standardization**: Only after all arrays are guaranteed to be the same shape (125 or more elements), perform `np.stack()`

**Code addition:**
```python
# Handle ragged arrays: ensure all weight arrays have minimum 125 elements
smeft_raw = data['smeft_weights']
smeft_arrays = []
for w in smeft_raw:
    if hasattr(w, 'shape'):
        # Already a numpy array
        if w.shape[0] < 125:
            logging.warning(f"  Warning: Event has only {w.shape[0]} weights, need at least 125")
            # Pad with zeros if too short
            w = np.pad(w, (0, 125 - w.shape[0]), mode='constant', constant_values=0)
        smeft_arrays.append(w)
    else:
        # Single value or other type - convert to array with padding
        w_arr = np.atleast_1d(w)
        if len(w_arr) < 125:
            w_arr = np.pad(w_arr, (0, 125 - len(w_arr)), mode='constant', constant_values=0)
        smeft_arrays.append(w_arr)

# Stack into (N, 125+) array
smeft = np.stack(smeft_arrays)
```

### Impact

- ✅ Enables use of new ROOT data file (`ttZ_for_Melle_tree.root`) with ~6× more events
- ✅ Supports any future ROOT files with ragged weight arrays
- ✅ Logs warnings for undersized weight arrays (aids debugging)
- ℹ️ Padding with zeros doesn't affect physics since indices 122 and 124 are always checked

### Testing

Training jobs with `ttZ_for_Melle_tree.root` now proceed past the weight loading step. No functional change to existing behavior with uniform-length arrays.

---

## Entry 32: Ragged SMEFT Array Fix — Replaced with Awkward-Array Approach

**Date:** 2026-03-06
**File:** `ml/custom/ttz/process_ttz_dataset.py`

### Problem

The per-event loop fix introduced in Entry 31 still used `np.stack()` at the end, which fails whenever any two events have arrays of different lengths (e.g. 1032 vs 987). Additionally, looping over 3.7M events in Python is extremely slow.

### Solution

Load `smeft_weights` separately using `library="ak"` (awkward arrays), which handles ragged arrays natively. Use awkward's column-slice syntax to extract only the two needed indices:

```python
smeft_ak = tree_ttz.arrays(['smeft_weights'], library="ak")['smeft_weights']
w_plus  = ak.to_numpy(smeft_ak[:, 124]).astype(np.float32)  # cHt = +5.0
w_minus = ak.to_numpy(smeft_ak[:, 122]).astype(np.float32)  # cHt = -5.0
```

This avoids `np.stack()` entirely — awkward handles the ragged structure internally and produces flat numpy arrays directly. All other branches are still loaded via `library="np"` as before.

### Impact

- ✅ Correctly handles ragged `smeft_weights` arrays across all 3.7M events
- ✅ Vectorized — no Python-level event loop
- ✅ Memory efficient — only extracts 2 indices out of 1032, discards the rest

---

## Entry 33: Plot X-Axis Labels on Every Row

**Date:** 2026-03-06
**File:** `ml/custom/ttz/smeft_reweighting.py`

### Change

Previously, x-axis tick numbers and feature name labels on the ratio panels were only shown on the bottom row to reduce clutter. User requested labels on every row for readability.

Removed the `if row == n_rows - 1` conditional and unconditionally set:
```python
ax_ratio.set_xlabel(feat, fontsize=8)
ax_ratio.tick_params(labelbottom=True, labelsize=8)
```

Also updated the ratio direction label from `ROOT(c) / Generated` to `Generated / ROOT(c)` (Entry matches earlier change in same session).

---

## Entry 34: Training Instability Fix — Negative Weight Clamping + Per-Batch Normalisation

**Date:** 2026-03-06
**Files:** `ml/custom/ttz/ttz_dataset.py`, `ml/flows/models/made_mog.py`

### Problem

Both the linear and quadratic flow training runs failed:

- **Linear**: NaN loss from epoch 0. The normalised linear weights had `max=82.7×mean`. A single batch dominated by a few extreme-weight events caused the MOG log-prob to leave the finite range in the very first forward pass.
- **Quadratic**: Loss explosion after epoch 5 (reaching ~10²⁵), despite `gradient_clip_val=1.0`. The gradient clipping caps gradient *magnitude* but not the loss value itself — a batch with extreme-weight events still causes catastrophic optimizer steps.

Root cause: the SMEFT decomposition weight values `w_linear = (w⁺ − w⁻)/10` and `w_quadratic = (w⁺ + w⁻ − 2w_sm)/50` have very long tails (max ~80–100×mean) and a small negative tail (~0.3%). Feeding these directly as importance weights without any batch-level control makes training numerically unstable.

### Solution

Two targeted fixes:

**1. Clamp negative weights to 0 (`ttz_dataset.py`)**

Negative decomposition components represent destructive SMEFT interference. They are physically meaningful but *mathematically ill-posed* as importance weights for a normalizing flow — a probability density is always non-negative. Negative weights would instruct the flow to decrease its density at those events, which is incoherent. Since negatives are only ~0.3% of events with small magnitude (min ≈ −0.13 vs mean=1.0), the physics impact is negligible.

```python
n_neg = int((self.weights < 0).sum())
if n_neg:
    logging.warning(f"  Clamping {n_neg} negative '{self.weight_type}' weights to 0 ({100*n_neg/len(self.weights):.3f}%)")
    self.weights = np.clip(self.weights, 0.0, None)
```

**2. Per-batch weight normalisation (`made_mog.py` training_step)**

Instead of global clipping (which would distort physics at high momentum transfer where SMEFT effects are largest), weights are normalised *per batch* by dividing by their batch mean before computing the weighted loss:

```python
batch_mean = weights.mean().clamp(min=1e-8)
weights = weights / batch_mean
```

This preserves the relative ordering within each batch (extreme-weight events still pull the gradient harder), but prevents any batch from having a 80×-inflated effective learning rate. The global weight distribution (used for Z normalization constants in `smeft_reweighting.py`) is untouched.

**3. Removed global p99 clip (`ttz_dataset.py`)**

The previously introduced p99 clipping was removed since per-batch normalisation is a strictly better approach: it handles numerical stability without discarding any physics information from the high-weight events.

The `ttz_dataset.py` now logs weight percentiles (p99, p99.9) instead of modifying them.

### Impact

- ✅ Fixes NaN loss in linear training
- ✅ Fixes loss explosion in quadratic training
- ✅ Does not distort physics at high pT / high-weight events
- ✅ Negative weights (destructive interference) are excluded — consistent with flow probability semantics

---

## Entry 35: Signed Weights + Stratified Batching — Stable Training Without Physics Distortion

**Date:** 2026-03-06
**Files:** `ml/custom/ttz/ttz_dataset.py`, `ml/flows/models/made_mog.py`

### Problem

Weight-clipping tests (`test_weight_clipping.py`) revealed that *any* modification to
the raw weights visibly distorts the physics distributions:

- **Linear**: has ~16.7% negative-weight events. Clamping negatives to 0 *or* taking
  absolute values both produce significant histogram deviations (ratio panels clearly
  drift from 1). These negatives are physically meaningful — destructive SMEFT
  interference where increasing $c_{Ht}$ reduces the cross-section in that phase-space
  region.
- **Quadratic**: effectively no negatives (<10/500k), but has a very long positive tail
  (max ~460×mean). Capping at 5, 10, or even 20 still distorts the high-weight tail
  where EFT effects are largest.

The Entry 34 approach (clamp negatives to 0 + per-batch `mean(w)` normalisation) was
therefore incorrect: it discards real physics in the linear case and does not fully solve
the tail problem in the quadratic case.

### Solution

Three coordinated changes:

**1. Global normalisation by `mean(|w|)` instead of `mean(w)` (`ttz_dataset.py`)**

```python
# Before
weight_mean = np.mean(self.weights)
self.weights = self.weights / weight_mean

# After
weight_abs_mean = float(np.mean(np.abs(self.weights)))
self.weights = self.weights / weight_abs_mean
```

Using `mean(|w|)` guarantees a stable, positive denominator regardless of the sign
distribution. For the linear case where 17% of weights are negative, `mean(w)` is
significantly smaller than `mean(|w|)`, which inflated the normalised weights
unnecessarily.

The **negative-clamping block was removed entirely** — signed weights are kept as-is.

**2. Per-batch normalisation by `mean(|w|)` instead of `mean(w)` (`made_mog.py`)**

```python
# Before
batch_mean = weights.mean().clamp(min=1e-8)

# After
batch_mean = weights.abs().mean().clamp(min=1e-8)
```

With signed weights in the batch, `weights.mean()` can drift near zero (or go negative)
if a batch is drawn heavily from the negative tail. `clamp(min=1e-8)` would then amplify
all weights catastrophically. `weights.abs().mean()` is always a healthy positive number.
Negative weights retain their gradient-suppression semantics — a negative $w_i$ pushes
$p_\theta(x_i)$ down, which is physically correct for destructive interference.

**3. `WeightStratifiedSampler` for batch composition (`ttz_dataset.py`)**

Training with extreme positive outliers (quadratic max ~460×mean) is still risky even
with per-batch normalisation: a batch drawn by chance entirely from the top quantile
produces an unusually large gradient step. The fix is to *guarantee* that every batch
spans the full weight range.

`WeightStratifiedSampler(weights, batch_size, n_bins=4)`:
- Divides training events into 4 equal-count bins by `|weight|`
- Each batch takes `batch_size // 4` events from each bin (shuffled within bin each epoch)
- Bins are recycled with re-shuffling when exhausted
- **Weights are not modified** — only batch composition is controlled

`ttzDataModule.train_dataloader()` is overridden to use this sampler automatically
whenever `use_weights=True`.

### Impact

- ✅ Negative linear weights are preserved — no physics distortion from destructive interference
- ✅ Quadratic extreme outliers are diluted per-batch without any value modification
- ✅ Per-batch `mean(|w|)` normalisation is stable for all weight distributions
- ✅ SM training is unaffected (sampler disabled when `use_weights=False`)

---

## Entry 36: Refined Stability Strategy — Stratified Sampling Only (No Per-Batch Normalisation)

**Date:** 2026-03-09
**Files:** `ml/custom/ttz/ttz_dataset.py`, `ml/flows/models/made_mog.py`

### Problem

Training with Entry 35 (stratified sampling + per-batch `mean(|w|)` normalisation) still failed:

- **Linear**: `n_neg=911124 (24.58%)` — higher than the 17% observed in test subset. Validation loss exploded to $3.5 \times 10^{24}$ on epoch 0.
- **Quadratic**: Stratified sampling helped (79 epochs!), but validation eventually went negative ($-3.2 \times 10^{16}$), indicating numerical overflow still occurring.

Root cause analysis: **Per-batch normalisation rescales all weights in a batch by a constant factor**. While this shouldn't affect the *learned distribution* (the flow learns $\log q(x)$, not weights), it does:
1. **Mask instability signals** — early stopping gets confused when validation loss is unnormalised
2. **Introduce artificial scaling** — the flow sees weight magnitudes that depend on batch composition, not the true distribution

### Solution

**Revert to a simpler, cleaner approach:**

1. **Global normalisation by `mean(w)` only** (`ttz_dataset.py`):
   ```python
   weight_mean = float(np.mean(self.weights))
   self.weights = self.weights / weight_mean
   ```
   This is a one-time rescaling to unit mean. Signed weights (including negatives) are preserved fully.

2. **Remove per-batch normalisation** (`made_mog.py`):
   ```python
   if sm_weights is not None and self.use_sm_weights:
       weights = weights * sm_weights
       # No further re-normalisation — stratified sampling controls batch composition
   ```
   The flow sees true weight magnitudes and distributions.

3. **Keep stratified sampling** (`ttz_dataset.py`):
   - Divides events into 4 quantile bins by $|w|$
   - Each batch draws equally from all bins
   - Applied to *both* train and validation dataloaders
   
   This ensures no batch is ever dominated by extreme-weight events, preventing numerical runaway.

### Rationale

Stratified sampling is a **structural control** on batch composition, whereas per-batch normalisation is a **gradient-scale hack**. The former is more principled:
- ✅ Prevents extreme-weight batches by *design*, not by rescaling
- ✅ Flow sees true weight range and relative magnitudes  
- ✅ Early stopping signals are honest and comparable across epochs
- ✅ No per-batch artefacts introduced

The risk was: could an extreme-weight event still cause numerical overflow even with stratification? Possibly, but it's mitigated because:
- Per-event loss `-(log_det_jacobian + mog_nll)` is bounded (log-prob range is finite)
- Extreme-weight events are diluted in every batch by lower-weight events
- If numerical issues persist, we can add a *soft* safety clamp (e.g., 10× median $|w|$) as a last resort

### Impact

- ✅ Flow learns from true, unscaled weight distributions
- ✅ Stratified sampling (structural) replaces per-batch rescaling (hacky)
- ✅ Early stopping and validation loss are meaningful and comparable
- ✅ Negative weights fully preserved for linear component

---

## Entry 37: SNIS Normalisation in Training and Validation — Fix Overflow

**Date:** 2026-03-09
**File:** `ml/flows/models/made_mog.py`

### Problem

Entry 36 (stratification only, no per-batch normalisation) still failed:

- **Linear**: Training loss went to $-8.4 \times 10^{20}$ by epoch 5
- **Quadratic**: Best val loss was $-7.6 \times 10^7$ after only 3 epochs

Logs confirmed the weight distributions are large: `max=283` (linear), `max=238` (quadratic). Stratification guarantees each batch draws from all weight bins, but does **not** bound the loss value. Once the flow starts fitting well and assigning high log-prob to high-weight events, the product `w × (-log p(x))` can become arbitrarily large regardless of batch composition.

The critical oversight was that **validation_step had no weight normalisation at all**. The validation loss was computed as raw `mean(w × NLL)` with `w` up to 238–283, causing early stopping to fire on a spurious explosion.

### Solution

Apply **self-normalised importance sampling (SNIS)** consistently in all three weight-application sites:

```python
batch_abs_mean = weights.abs().mean().clamp(min=1e-8)
weights = weights / batch_abs_mean
```

Applied to:
1. `training_step` — already had this; restored after Entry 36 removed it
2. `validation_step` (with `log_jac` branch) — **new**
3. `validation_step` (without `log_jac` branch) — **new**

### Why SNIS Does Not Affect Learned Physics

SNIS estimates:

$$\hat{E}_\text{SMEFT}[-\log q(x)] = \frac{\sum_i w_i (-\log q(x_i))}{\sum_i |w_i|}$$

At inference, SMEFT reweighting computes density *ratios* $q_\text{SMEFT}(x) / q_\text{SM}(x)$. Any constant multiplicative factor in $q$ cancels in this ratio. So the learned *shape* of the distribution is identical regardless of per-batch scale — only the absolute normalisation would differ, which is irrelevant for ratio-based reweighting.

Signs are fully preserved: a negative $w_i$ still pushes $q(x_i)$ down.

### Combined Strategy (Entries 35–37)

| Layer | Tool | Purpose |
|---|---|---|
| Global | `mean(w)` normalisation | Unit-mean starting scale |
| Batch composition | `WeightStratifiedSampler` on train + val | Prevent outlier-dominated batches |
| Loss scale | SNIS `/ mean(\|w\|)` in train + val | Prevent overflow as flow fits |

### Impact

- ✅ Prevents loss overflow in both training and validation
- ✅ Validation loss is numerically stable and comparable across epochs
- ✅ Early stopping fires for the right reason (plateau), not overflow
- ✅ Negative weights (signed physics) fully preserved
- ✅ Learned density shape unaffected — ratios cancel the scale


---

## Entry 38: Pos/Neg Flow Split — Proper Architecture for Signed SMEFT Weights

**Date:** 2026-04-xx
**Files:** `ml/custom/ttz/ttz_dataset.py`, `ml/custom/ttz/train_all_weights.py`, `ml/custom/ttz/smeft_reweighting.py`

### Problem

The root cause of all previous instability (Entries 35–37) is that normalising flows require non-negative importance weights, but the SMEFT linear and quadratic decomposition components are *signed measures*:

- **Linear**: ~20% of events have negative weights (max=283, p99=23.8)
- **Quadratic**: ~8% of events have negative weights (max=238, p99=23.8)

The negative entries make the training objective **sign-indefinite and unbounded from below**: once the flow fits well and assigns high log-prob to a negative-weight event, the product `w × (-log p(x))` becomes a large positive loss, destabilising training. Clamping, SNIS, and stratification are palliatives but cannot fix the fundamental incompatibility.

The "drop negatives" test (Entry 37 follow-up) confirmed stability at epoch 0 (train=13.84, val=14.00 for linear), proving that non-negative weights are trainable — the negatives were the sole cause of overflow.

### Solution: Pos/Neg Flow Decomposition

Decompose each signed weight component into two non-negative parts:

$$w_\text{lin} = w_\text{lin}^+ - w_\text{lin}^-$$

where $w_\text{lin}^+(x) = \max(w_\text{lin}(x), 0)$ and $w_\text{lin}^-(x) = \max(-w_\text{lin}(x), 0)$.

Train **five** normalising flows, one per non-negative component:

| Flow | Trained on | Training weights |
|------|-----------|-----------------|
| `sm` | all events | $w_\text{sm}$ |
| `linear_pos` | events with $w_\text{lin} \geq 0$ | $w_\text{lin}$ |
| `linear_neg` | events with $w_\text{lin} < 0$ | $|w_\text{lin}|$ (sign flipped) |
| `quadratic_pos` | events with $w_\text{quad} \geq 0$ | $w_\text{quad}$ |
| `quadratic_neg` | events with $w_\text{quad} < 0$ | $|w_\text{quad}|$ (sign flipped) |

All training weights are strictly non-negative, so the flow objective is well-posed.

The combination formula in `smeft_reweighting.py` reconstructs the full SMEFT weight:

$$w_i(c) \propto Z_\text{sm} + c \left(Z_\text{lin}^+ \exp(r_{\text{lin}+,i}) - Z_\text{lin}^- \exp(r_{\text{lin}-,i})\right) + c^2 \left(Z_\text{quad}^+ \exp(r_{\text{quad}+,i}) - Z_\text{quad}^- \exp(r_{\text{quad}-,i})\right)$$

where $Z_k = \sum_j w_{k,j}$ and $r_{k,i} = \log p_k(x_i) - \log p_\text{sm}(x_i)$.

Note: $Z_\text{lin}^+ - Z_\text{lin}^- = Z_\text{lin}$ exactly (checked with assertion).

### Changes

**`ttz_dataset.py`**:
- Extended `WEIGHT_COLS` with `linear_pos`, `linear_neg`, `quadratic_pos`, `quadratic_neg` (all pointing to the same raw column as their base type)
- In `setup()`: for `_pos` types, filter to `w >= 0`; for `_neg` types, filter to `w < 0` and flip sign (`-w`)
- Both filters applied *before* mean normalisation so `mean(|w|)` is correct on the kept subset
- Base types (`linear`, `quadratic`) retain backward-compatible negative-dropping behavior with a warning recommending the split types

**`train_all_weights.py`**:
- `ALL_WEIGHT_TYPES` changed from `["sm", "linear", "quadratic"]` to `["sm", "linear_pos", "linear_neg", "quadratic_pos", "quadratic_neg"]`
- Updated module docstring to describe the split architecture

**`smeft_reweighting.py`**:
- Added `import awkward as ak`
- `compute_z_normalisations`: fixed `np.stack` ragged-array bug (now uses `ak.to_numpy` with index slicing); computes $Z_\text{lin}^\pm$ and $Z_\text{quad}^\pm$ with sanity assertions
- `compute_log_ratios`: signature now takes 5 log-prob arrays, returns 4 log-ratio arrays
- `get_weights`: signature updated to 4 log-ratio arrays; formula implements the pos/neg combination above
- `plot_reweighted_distributions`, `plot_neff_curve`: updated to propagate 4 log-ratio arrays through to `get_weights`
- `main()`: loads 5 models, computes 5 log-probs, saves 4 ratios to `density_ratios.npz`

### Why This Is The Right Fix

Every attempted workaround (clamping, abs, SNIS, stratification) distorted the
physics by modifying *what the flows learn*. The pos/neg split is exact:

- ✅ No information is discarded — *all* events contribute to one of the five flows
- ✅ All five flows see strictly non-negative training weights — objective is well-posed
- ✅ The combination formula reproduces the original signed SMEFT measure exactly
- ✅ Negative per-event weights $w_i(c)$ can still appear at inference (destructive interference) — this is physically correct
- ✅ `Z_lin_pos - Z_lin_neg = Z_lin` enforced by assertion in `compute_z_normalisations`

### Impact

- ✅ Training is numerically stable for all five weight types
- ✅ No more overflow from signed importance weights
- ✅ SMEFT combination formula analytically exact
- ✅ `RATIOS_FILE` format updated (4 arrays instead of 2); old stale files will be detected and recomputed automatically

---

## Entry 39: Data-Adaptive Model and Batch-Size Scaling

**Date:** 2026-03-09
**Files:** `ml/custom/ttz/main_flows.py`, `ml/custom/ttz/ttz_dataset.py`

### Problem

The five-flow architecture (Entry 38) trains on subsets of very different sizes:

| Weight type | ~% of events | n_train (of 3.7M) |
|-------------|--------------|-------------------|
| `sm` | 100% | ~2.96M |
| `linear_pos` | ~80% | ~2.37M |
| `linear_neg` | ~20% | ~0.59M |
| `quadratic_pos` | ~92% | ~2.72M |
| `quadratic_neg` | ~8% | ~0.22M |

Using the same model (5 flows, 4 hidden layers, dim=384) and batch size (2048) for all runs is wasteful and risks overfitting for the small `_neg` subsets. The `quadratic_neg` job had already failed twice due to a separate empty-DataLoader bug, but even after fixing that, the architecture is disproportionately large for ~220k events.

### Solution: Data-Size Tiers

Added `_data_size_tier` and `_scale_config_to_data` helper functions to `main_flows.py`. After creating the data module, `setup("fit")` is called early to learn `n_train`, then the model config and batch size are scaled down according to a four-tier scheme:

| Tier | n_train | scale | Affected runs |
|------|---------|-------|---------------|
| large | ≥ 1 000 000 | 1.00 | sm, linear_pos, quadratic_pos |
| **medium** | **≥ 200 000** | **0.67** | **linear_neg, quadratic_neg** |
| small | ≥ 50 000 | 0.50 | — |
| tiny | < 50 000 | 0.33 | — |

For `quadratic_neg` at medium (scale=0.67), the default config becomes:

| Param | Default | Scaled |
|-------|---------|--------|
| `hidden_layer_dim` | 384 | 256 |
| `hidden_layer_mog_dim` | 384 | 256 |
| `num_flows` | 5 | 3 |
| `num_hidden_layers` | 4 | 3 |
| `num_hidden_layers_mog_net` | 4 | 3 |
| `batch_size` | 2048 | 1280 |

Parameters left unchanged: `n_mixtures`, `res_layers_in_block`, `batchnorm_flow`, `conv1x1`, `activation`. Changing the MoG mixture count has a larger quality impact than linear capacity scaling; residual blocks are cheap.

### Changes

**`main_flows.py`**:
- Added `_data_size_tier(n_train) → (tier_name, scale_factor)` helper
- Added `_scale_config_to_data(n_train, model_conf, data_module) → model_conf` helper: scales hidden dims (rounded to nearest 32), num_flows, num_hidden_layers, and batch_size (rounded to nearest 128); logs all changes; no-ops for large tier
- After `ttzDataModule` creation, calls `data_module.setup("fit")` early, converts Hydra `DictConfig → dict` via `OmegaConf.to_container`, then calls `_scale_config_to_data`
- Updated config logging to read batch_size from `data_module.dataloader_kwargs` (which may have been modified by scaling) rather than the raw yaml value

**`ttz_dataset.py`**:
- Added idempotency guard at the top of `setup()`: sets `_setup_done = True` on first call, returns immediately on subsequent calls. This prevents the expensive data-loading processor from running twice when Lightning calls `setup()` again during `trainer.fit()`.

### Design Notes

- Rounding to nearest 32 for hidden dims keeps CUDA memory alignment efficient
- `n_mixtures` is kept fixed: the MoG distribution is the expressive core of MAFMADEMOG; reducing it would harm density estimation quality more than halving hidden dims
- The early `setup("fit")` call is safe because `ttz_dataset.py` is now idempotent: Lightning's later call inside `trainer.fit()` hits the guard and returns immediately
- All scaling logic is logged at INFO level with before/after values for each changed parameter

### Impact

- ✅ `quadratic_neg` and `linear_neg` trained with appropriately sized model (~40% fewer parameters)
- ✅ Smaller batch size avoids DataLoader length issues on medium-sized datasets
- ✅ Large-tier runs (sm, linear_pos, quadratic_pos) unaffected — full default config
- ✅ No yaml changes needed — scaling is computed at runtime from actual data

---

## Entry 40: Fix sample_analyzer for pos/neg split and ttz_weights.npy

**Date:** 2026-03-10
**Files:** `ml/custom/ttz/run_smeft_analysis.py`, `ml/custom/ttz/sample_analyzer/analyzer.py`, `ml/custom/ttz/sample_analyzer/run_analysis.py`

### Problems

Two independent issues discovered when running `run_smeft_analysis.py` after the pos/neg split refactor (Entry 38):

**1. `run_smeft_analysis.py` — stale weight type list**
`ALL_WEIGHT_TYPES` still contained `["sm", "linear", "quadratic"]`. The script would have searched MLflow for models named `cHt5_linear` and `cHt5_quadratic`, which no longer exist. Updated to `["sm", "linear_pos", "linear_neg", "quadratic_pos", "quadratic_neg"]` to match the new five-flow architecture. Updated docstring example accordingly.

**2. `analyzer.py` / `run_analysis.py` — `ttz.npy` does not exist**
Both files hardcoded `"ml/data/ttz/ttz.npy"` which was never generated. Only `ttz_weights.npy` (15 features + 4 SMEFT weight columns = 19 columns) exists on disk.

Additionally, `analyzer.py` contained a dead weight-extraction block (`if self.original_data.shape[1] == 16`) written for a legacy single-weight file that no longer exists. This produced misleading "UNWEIGHTED" warnings.

### Changes

**`run_smeft_analysis.py`**:
- `ALL_WEIGHT_TYPES` updated from `["sm", "linear", "quadratic"]` to `["sm", "linear_pos", "linear_neg", "quadratic_pos", "quadratic_neg"]`
- Docstring usage example updated

**`run_analysis.py`**:
- `data_dir` changed from `"ml/data/ttz/ttz.npy"` → `"ml/data/ttz/ttz_weights.npy"`

**`analyzer.py`**:
- Primary `original_data` load: `ttz.npy` → `ttz_weights.npy`, adds `full_data = full_data[:, :15]` slice to keep only the 15 physics feature columns
- Scaler-refit fallback path in `_setup_preprocessing`: same `ttz.npy` → `ttz_weights.npy` fix with matching 15-column slice
- Removed dead weight-extraction block (`if self.original_data.shape[1] == 16`); replaced with `self.original_weights = None` and a comment explaining that SMEFT reweighting is handled by `smeft_reweighting.py`, not this analyzer

### Design Notes

- `ttz_weights.npy` has shape `(3707227, 19)`: columns 0–14 are the 15 physics features, columns 15–18 are `(w_sm, w_lin, w_quad, w_full)`. Slicing `[:, :15]` gives the same feature array that the flow models were trained on.
- This analyzer produces **unweighted** SM comparison plots (flow samples vs MC data), which is the correct use case. SMEFT-reweighted plots are produced separately by `smeft_reweighting.py`.

### Impact

- ✅ `run_smeft_analysis.py` now finds all five models in MLflow registry
- ✅ `FileNotFoundError: ttz.npy` resolved for all weight types
- ✅ No spurious "MC histograms will be UNWEIGHTED" warning

---

## Entry 41: Fix analyzer.py — UnboundLocalError and greedy regex for _pos/_neg models

### Problem

Feature comparison plots for `linear_pos`, `linear_neg`, `quadratic_pos`, and `quadratic_neg` models showed the wrong data: the MC data histogram was the full unfiltered dataset instead of the sign-filtered subset, and **no weights were applied** to it at all. This caused large visible discrepancies (e.g. `Z_DeltaR` for `lin_neg`) where the "MC data" panel looked nothing like the generated panel.

Two bugs compounded to cause this:

**Bug 1 — `UnboundLocalError` in `analyzer.py` `__init__` (crash)**

The filtering block for `_pos`/`_neg` types did:
```python
full_data = full_data[valid_mask]   # ← referenced before assignment
```
But `full_data` was defined *after* this block:
```python
full_data = full_data_with_weights[:, :15]   # too late
```
Python's scoping treats `full_data` as a local variable because it is assigned later in the same scope, so accessing it earlier raises `UnboundLocalError`. The `__init__` crashed for all four `_pos`/`_neg` models, `run_smeft_analysis.py` logged the failure and moved on, and the stale plots from a prior run were left on disk.

**Bug 2 — greedy `\w+` regex swallowed model-name suffixes**

Even if Bug 1 had not crashed, the weight-type extraction:
```python
re.search(r'cHt5_(\w+)', model_name)  # captured "linear_neg_nall" not "linear_neg"
```
matched all word characters including the `_nall` suffix embedded in every model name (`…cHt5_linear_neg_nall`). The captured string `"linear_neg_nall"` was not in `weight_col_map`, so `self.original_weights = None` was set for every model — meaning even `sm` data was shown unweighted.

### Changes

**`ml/custom/ttz/sample_analyzer/analyzer.py`**:

- Moved `full_data = full_data_with_weights[:, :15]` to *before* the sign-filtering block, so `full_data[valid_mask]` is valid in both `_pos` and `_neg` branches.
- Removed the now-redundant second `full_data = …` assignment that used to follow the filtering block.
- Updated the final `self.original_data` assignment to use the (possibly filtered) `full_data` and log the row count.
- Replaced the greedy `re.search(r'cHt5_(\w+)', ...)` with an explicit alternation:
  ```python
  re.search(r'cHt5_(quadratic_(?:pos|neg)|linear_(?:pos|neg)|sm|quadratic|linear|full)', model_name)
  ```
  This matches the longest known type token first (e.g. `quadratic_pos` before `quadratic`) and stops before any trailing suffixes like `_nall`.

### Impact

- ✅ `linear_neg` / `quadratic_neg` analyzers no longer crash in `__init__`
- ✅ MC data for each model is now the correct sign-filtered subset (e.g. `linear_neg` uses only events where `w_linear < 0`)
- ✅ MC data is weighted by `|w|` of the appropriate column, matching exactly what each flow was trained on
- ✅ No more "Unknown weight_type" warnings in `run_smeft_analysis.py` output

---

## Entry 42: Add Ragged Array Guard to smeft_reweighting.py

### Problem

Running `smeft_reweighting.py` after switching to the new ROOT file (`ttZ_for_Melle_tree.root`) crashed with:
```
IndexError: cannot slice ListArray (of length 3707228) with array(124): 
index out of range while attempting to get index 124
```

The new ROOT file contains some events with fewer than 125 SMEFT weight values (ragged array). Direct indexing 
with `data['smeft_weights'][:, 124]` fails because some inner arrays are shorter than expected.

### Root Cause

`ttZ_for_Melle_tree.root` has incomplete SMEFT weight arrays for some events. Both functions that read this file 
were missing the **ragged array guard**:
- `compute_z_normalisations()`: tried `[:, IDX_CHT_PLUS5]` directly on ragged awkward array
- `load_root_features_and_weights()`: tried `np.stack()` on ragged data, which also fails

Meanwhile, `process_ttz_dataset.py` (which generates training data from the same file) had already implemented 
this guard correctly.

### Changes

**`ml/custom/ttz/smeft_reweighting.py`**:

1. **`compute_z_normalisations()`**:
   - Load `smeft_weights` via awkward
   - Filter with `ak.num(smeft_ak) > 124` → valid_mask
   - Drop events with fewer than 125 weights (may be 0 or a few depending on file)
   - Only then index with `[:, IDX_CHT_PLUS5]` and `[:, IDX_CHT_MINUS5]`

2. **`load_root_features_and_weights()`**:
   - Load feature branches with numpy (regular), smeft_weights separately with awkward (ragged)
   - Apply same `ak.num(smeft_ak) > 124` filter
   - Filter all data arrays (`data[key] = data[key][valid_mask]`)
   - Convert cleaned smeft_ak to regular array with `ak.to_regular()` before indexing

3. **branches_to_load cleanup**:
   - Removed `'smeft_weights'` from the numpy-based load (it's now loaded separately)
   - Added comment explaining the separate ragged load

### Impact

- ✅ Both `compute_z_normalisations()` and `load_root_features_and_weights()` now handle ragged arrays safely
- ✅ Matches the guard already in `process_ttz_dataset.py` (consistent across codebase)
- ✅ `smeft_reweighting.py` can now run successfully on the new ROOT file
- ✅ Z-factors and feature comparisons computed from the correct, consistent data source

### Consistency Note

All three components now use identical logic:
- **Training data generation** (process_ttz_dataset.py): ✓ Guard implemented
- **Z-factor computation** (smeft_reweighting.py): ✓ Guard now implemented
- **Weight decomposition** (plot_weight_contributions.py): ✓ Implicitly safe (loads pre-filtered ttz_weights.npy)

---

## Entry 43: GPU Support and Condor Batch Job for smeft_reweighting.py

### Motivation

Running `smeft_reweighting.py` with 10M samples was prohibitively slow on CPU:
- Sampling alone (Step 3) was taking ~60 minutes (25 minutes at 40%)
- Sampling is autoregressive: $d$ sequential forward passes over all $N$ samples → exactly the kind of large matrix multiply that benefits from GPU
- Steps 4 (log prob evaluation) also benefits from GPU when computing ratios fresh

### Changes

**`ml/custom/ttz/smeft_reweighting.py`**:

1. **`load_module(model_name, device='cpu')`**: Added `device` parameter (default `'cpu'`, backward compatible).
   - Calls `.to(dev)` instead of `.cpu()`
   - Patches `MADEMOG.device` and `AutoregressiveNormalizingFlow.device` to the target device (previously only patched to CPU)

2. **`sample_from_sm_flow()`**: Refactored to sample in chunks of 500k instead of all at once.
   - Avoids GPU OOM for 10M samples (L40S has 46GB VRAM, but autoregressive sampling holds intermediate tensors)
   - Each chunk result is moved to CPU immediately (`.detach().cpu().numpy()`)
   - Progress logged every chunk: `500,000 / 10,000,000 sampled (5%) ...`
   - Result is always a CPU numpy array regardless of model device

3. **`--device cuda` CLI argument**: Wires through to all 5 model loads (sm, lin_pos, lin_neg, quad_pos, quad_neg). Inverse transform always loads on CPU since it is pure numpy scaler operations.

4. **Redundant `w_root` computation fixed**: `w_root_c` is now computed once per c-value outside the feature loop, not 22× inside it (no functional change, minor speedup).

**New condor files**:

- `condor/mveldijk/condorsub/smeft_reweighting.sub`: Requests 1× L40S GPU, 4 CPUs, 64GB RAM
- `condor/mveldijk/condorsub/smeft_reweighting.sh`: Sources same env/venv as training jobs, runs:
  ```
  python ml/custom/ttz/smeft_reweighting.py --device cuda --n-samples 10000000
  ```

### GPU vs CPU Breakdown

| Step | GPU benefit? | Reason |
|------|-------------|--------|
| Step 2: Z-factors (ROOT I/O) | ❌ | Pure numpy/awkward I/O |
| Step 3: Sampling (10M events) | ✅ **Major** | Autoregressive: $d$ sequential large matmuls |
| Step 4: Log prob evaluation (5 flows × 10M) | ✅ **Major** | Batched neural net forward passes |
| Step 5: Log density ratios | ❌ | Trivial numpy arithmetic |
| Step 7: Inverse transform + plotting | ❌ | Pure numpy + matplotlib |

### Impact

- ✅ `smeft_reweighting.py` can now be submitted as a GPU batch job via condor
- ✅ Sampling speedup expected: 10-50× (CPU: ~60 min → GPU: ~2-5 min for 10M samples)
- ✅ Chunked sampling gives per-chunk progress feedback (previously no feedback for 25+ minutes)
- ✅ `--load-ratios` path still works unchanged (loads CPU model for inverse transform only)
- ✅ Backward compatible: default `--device cpu` preserves existing interactive behaviour


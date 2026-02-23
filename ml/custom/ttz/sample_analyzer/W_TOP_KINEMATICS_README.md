# W Boson and Top Quark Reconstruction

## Overview

Extended the physics utilities to include reconstruction of W boson and top quark candidates from ttZ events. These higher-order variables provide additional validation of the model's ability to learn complex multi-particle correlations.

## Physics Context

In ttZ production with 3 leptons:

```
ttZ → (W b) (W̄ b̄) (ll)
      ↓      ↓     ↓
    lν bjet  qq̄ bbar  ll (from Z)
```

Our analysis focuses on the semi-leptonic channel where:
- **Z boson**: Decays to 2 leptons (already implemented)
- **W boson**: Decays to 1 lepton + neutrino (neutrino = MET)
- **Top quark**: Decays to W + b-jet

## Implementation

### W Boson Reconstruction

File: [`physics_utils.py`](physics_utils.py)

**Function**: `calculate_w_kinematics(data, feature_names)`

**Method**:
1. Identify the non-Z lepton (the one not forming the Z candidate)
2. Combine this lepton's transverse momentum with MET
3. Calculate transverse mass (since we don't have neutrino pz)

**Returns**:
- `W_Pt`: Transverse momentum of W (GeV)
- `W_Phi`: Azimuthal angle of W (radians)
- `W_MT`: **Transverse mass** of W (GeV)

**Key Formula** - Transverse Mass:
```python
MT_W = sqrt(2 * pt_lep * MET * (1 - cos(Δφ)))
```

This is the standard W reconstruction method used in collider physics when the neutrino's longitudinal momentum is unknown.

**Expected Physics**:
- Peak at W mass: ~80.4 GeV
- Width: ~2-3 GeV (natural W width + detector resolution)
- Shape: Jacobian peak (characteristic of transverse mass)

### Top Quark Reconstruction

File: [`physics_utils.py`](physics_utils.py)

**Function**: `calculate_top_kinematics(data, feature_names)`

**Method**:
1. Get W transverse momentum (from lepton + MET)
2. Add b-jet (Jet1) transverse momentum
3. Calculate transverse mass for the 3-body system

**Returns**:
- `Top_Pt`: Transverse momentum of top (GeV)
- `Top_Phi`: Azimuthal angle of top (radians)
- `Top_MT`: **Transverse mass** of top (GeV)

**Key Formula** - Three-body Transverse Mass:
```python
MT_top = sqrt((MT_W + pT_b)^2 + 2 * pT_W * pT_b * (1 - cos(Δφ)))
```

**Expected Physics**:
- Peak at top mass: ~172.8 GeV
- Width: ~20-30 GeV (broader due to 3-body system and detector effects)
- Distribution extends to higher values (combinatorial background)

### Why Transverse Mass?

**Problem**: Neutrino from W decay escapes detection → only transverse components known (MET)

**Solution**: Use transverse mass instead of invariant mass:
- Transverse mass uses only pT and φ (not pz, E)
- Still shows characteristic resonance structure
- Standard technique in W/top reconstruction at colliders

**Limitation**: Transverse mass is always ≤ invariant mass, so distributions will be shifted and broadened compared to true mass peaks.

## Usage

### In Analyzer

The analyzer automatically calculates these variables when `include_derived=True`:

```python
analyzer = ttzSampleAnalyzer(model_name="ttz_medium_10x768_phi_scaled")
analyzer.generate_samples(n_samples=100000)
analyzer.plot_all(include_derived=True)  # Includes W and top variables
```

### Standalone Usage

```python
from physics_utils import calculate_w_kinematics, calculate_top_kinematics

# data: shape (n_events, n_features)
# feature_names: list of feature names

# Calculate W kinematics
w_kin = calculate_w_kinematics(data, feature_names)
w_pt = w_kin['W_Pt']
w_mt = w_kin['W_MT']

# Calculate top kinematics
top_kin = calculate_top_kinematics(data, feature_names)
top_pt = top_kin['Top_Pt']
top_mt = top_kin['Top_MT']
```

## Validation Plots

When running the analyzer, look for these comparison plots:

### W_MT (W Transverse Mass)
- **Peak location**: Should be at ~80 GeV
- **Peak width**: ~2-3 GeV
- **Shape**: Jacobian edge (characteristic of MT)
- **Tail**: Falls off sharply above W mass

**What to check**:
- ✓ Model matches real data peak position
- ✓ Model matches real data width
- ✗ If peak is shifted: Model hasn't learned W mass constraint
- ✗ If too narrow: Model is overconfident
- ✗ If too wide: Model hasn't learned lepton-neutrino correlations

### W_Pt (W Transverse Momentum)
- **Range**: Typically 50-150 GeV for ttZ
- **Shape**: Exponentially falling spectrum
- **Physics**: Reflects W boost from top decay

**What to check**:
- ✓ Model matches real data spectrum shape
- ✗ If too soft: Model generates low-pT leptons/MET
- ✗ If too hard: Model generates high-pT leptons/MET

### Top_MT (Top Transverse Mass)
- **Peak location**: Should be around 150-180 GeV
- **Peak width**: ~20-30 GeV (broader than W)
- **Shape**: Broad peak (3-body system)
- **Tail**: Extends to higher values

**What to check**:
- ✓ Model matches real data peak position
- ✓ Model matches real data width and shape
- ✗ If peak is wrong: Model hasn't learned top→Wb decay
- ✗ If too narrow: Model isn't capturing b-jet pT distribution
- ✗ If completely off: Check jet ordering (is Jet1 really the b-jet?)

### Top_Pt (Top Transverse Momentum)  
- **Range**: Typically 100-300 GeV for ttZ
- **Shape**: Broad distribution
- **Physics**: Reflects top production kinematics

**What to check**:
- ✓ Model matches real data spectrum
- ✗ If wrong: Model hasn't learned event-level kinematics

## Physics Assumptions

1. **Massless leptons**: We assume electrons and muons are massless (good approximation at high pT)

2. **Jet1 is b-jet**: We assume the leading jet (Jet1) is the b-jet from top decay. This is usually true but not always.

3. **Correct lepton-jet pairing**: We assume the non-Z lepton and Jet1 come from the same top quark. In real events, there's combinatorial ambiguity.

4. **Single W, single top**: We only reconstruct one W and one top (the leptonic branch). The other top decays hadronically and is not reconstructed here.

## Expected Results

For a **well-trained model**:

```
W_MT distribution:
  - Peak at 78-82 GeV ✓
  - σ ≈ 2-3 GeV ✓
  - Matches real data shape ✓

Top_MT distribution:
  - Peak at 150-180 GeV ✓
  - σ ≈ 20-30 GeV ✓
  - Matches real data shape ✓
  
W/Top pT distributions:
  - Match real data spectra ✓
  - No unphysical spikes ✓
```

For a **poorly-trained model**:

```
W_MT distribution:
  - Peak shifted from 80 GeV ✗
  - Much broader or narrower ✗
  - Different shape from real data ✗

Top_MT distribution:
  - Peak far from expected range ✗
  - Completely wrong shape ✗
  
W/Top pT distributions:
  - Don't match real data ✗
  - Unphysical features ✗
```

## Testing

Test script: [`test_w_top_kinematics.py`](test_w_top_kinematics.py)

```bash
cd /project/atlas/users/mveldijk/MLHEPsimtest/MLHEPsim
source venv311/bin/activate
python ml/custom/ttz/sample_analyzer/test_w_top_kinematics.py
```

This creates a test event and verifies the calculations produce reasonable values.

## References

1. **W transverse mass**: Standard method for W mass measurement at hadron colliders
   - Used by ATLAS and CMS for W mass measurements
   - Jacobian edge provides clean signature

2. **Top reconstruction**: Multiple methods exist (kinematic fitting, χ² minimization)
   - Our method: Simple transverse mass (no full kinematic fit)
   - Trade-off: Less precise but more robust

3. **ttZ physics**: 
   - Cross section: ~1 pb at √s = 13 TeV
   - Important background for Higgs and BSM searches
   - Tests higher-order EW corrections

## Next Steps

Possible extensions:

1. **Full kinematic reconstruction**: Solve for neutrino pz using W mass constraint
2. **Both tops**: Reconstruct both top quarks (leptonic + hadronic)
3. **Chi-squared fitting**: Use likelihood fit to choose best lepton-jet pairing
4. **B-tagging**: Use Jet1_BTag score to verify b-jet assignment
5. **HT and missing HT**: Scalar sum of all jet pT, MET significance

These would provide even more stringent tests of model quality!

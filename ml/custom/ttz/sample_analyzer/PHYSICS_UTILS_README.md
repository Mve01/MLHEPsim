# Physics Utilities for ttZ Analysis

This module provides functions to calculate derived physical quantities from the generated particle kinematics.

## Higher-Order Variables

### Z Boson Kinematics

For ttZ events (3 leptons + jets), we reconstruct the Z boson candidate from the two leptons whose invariant mass is closest to the Z mass (91.2 GeV).

**Calculated quantities:**
- **Z_Pt**: Transverse momentum of the Z candidate (GeV)
- **Z_Eta**: Pseudorapidity of the Z candidate
- **Z_Phi**: Azimuthal angle of the Z candidate (radians)
- **Z_Mass**: Invariant mass of the two leptons forming the Z candidate (GeV)
- **Z_DeltaR**: Angular distance (ΔR) between the two leptons forming the Z

## Why This Matters

These derived quantities test whether the generative model learns **physical correlations** between particles, not just individual feature distributions.

### What Good Generation Looks Like:
- **Z_Mass peak at ~91 GeV**: The model correctly learns that two leptons should come from Z decay
- **Z_DeltaR distribution**: The model learns the typical angular separation of Z decay products
- **Z_Pt, Z_Eta, Z_Phi**: The model learns the kinematic properties of boosted Z bosons in ttZ events

### What Bad Generation Looks Like:
- Z_Mass scattered broadly (no peak at 91 GeV) → model doesn't understand Z physics
- Unrealistic Z_DeltaR (too small or too large) → model doesn't learn decay kinematics  
- Wrong Z_Pt spectrum → model doesn't learn event topology

## Physics Background

### Z Boson Decay

In ttZ events:
```
pp → ttZ → (b + leptons/jets)(b + leptons/jets)(Z → ℓ⁺ℓ⁻)
```

We have 3 leptons: 2 from Z decay + 1 from top decay. The Z decay leptons should:
1. Have opposite charges (SFOS: same-flavor opposite-sign)
2. Have invariant mass ~91.2 GeV
3. Have angular separation depending on Z boost

### Invariant Mass

For two particles with 4-momenta (E₁, p⃗₁) and (E₂, p⃗₂):

$$m_{inv} = \\sqrt{(E_1 + E_2)^2 - (\\vec{p}_1 + \\vec{p}_2)^2}$$

Assuming massless leptons:
- $E = p_T \\cosh(\\eta)$
- $p_x = p_T \\cos(\\phi)$
- $p_y = p_T \\sin(\\phi)$
- $p_z = p_T \\sinh(\\eta)$

### DeltaR Distance

Angular distance in (η, φ) space:

$$\\Delta R = \\sqrt{(\\Delta \\eta)^2 + (\\Delta \\phi)^2}$$

where Δφ is wrapped to [-π, π].

## Usage

### In Analyzer

The analyzer automatically calculates Z kinematics when plotting:

```python
from ml.custom.ttz.sample_analyzer.analyzer import ttzSampleAnalyzer

analyzer = ttzSampleAnalyzer(
    data_dir="ml/data/ttz/ttz.npy",
    variables_json_path="ml/data/ttz/variables.json",
    model_name="ttz_production"
)

# Generate samples
analyzer.generate_samples(n_samples=100000)

# Plot with derived variables (default)
analyzer.plot_all(include_derived=True)

# Or plot only basic features
analyzer.plot_all(include_derived=False)
```

### Standalone Usage

You can also use the physics utilities directly:

```python
from ml.custom.ttz.sample_analyzer.physics_utils import calculate_z_kinematics

# data: np.ndarray of shape (n_events, n_features)
# feature_names: list of feature names
z_kinematics = calculate_z_kinematics(data, feature_names)

# Access results
z_pt = z_kinematics['Z_Pt']         # array of shape (n_events,)
z_mass = z_kinematics['Z_Mass']     # array of shape (n_events,)
z_delta_r = z_kinematics['Z_DeltaR']  # array of shape (n_events,)
```

## Implementation Details

### Z Candidate Selection

The algorithm selects the Z candidate as follows:

1. Calculate invariant mass for all 3 possible lepton pairs: (1,2), (1,3), (2,3)
2. For each event, select the pair with mass closest to 91.2 GeV
3. Calculate Z kinematics from that pair's 4-momentum sum

This mimics the physics selection used in real ttZ analyses.

### Assumptions

- **Leptons are massless**: Good approximation (electron: 0.5 MeV, muon: 105 MeV << typical pT of 20-100 GeV)
- **No charge requirements**: We don't check opposite charges (assumes preprocessing already selected SFOS pairs)
- **All events have 3 leptons**: Assumes data comes from ttZ selection with exactly 3 leptons

## Validation

To test the physics calculations:

```bash
python ml/custom/ttz/sample_analyzer/test_z_kinematics.py
```

This runs unit tests on dummy data to verify:
- Invariant mass calculation is correct
- DeltaR calculation handles φ wrapping properly
- Z candidate selection picks the right lepton pair
- All derived quantities are physically reasonable

## Expected Results

For well-trained models on ttZ data, you should see:

**Z_Mass distribution:**
- Sharp peak at 91.2 GeV
- Width ~2-3 GeV (from Z natural width and detector resolution)
- Very few events outside 70-110 GeV range

**Z_Pt distribution:**
- Peaks around 50-100 GeV (typical for ttZ)  
- Extends to high pT (Z recoiling against heavy system)
- Should match real data distribution

**Z_DeltaR distribution:**
- Typical range: 0.5 - 4.0
- Peak depends on average Z pT
- Higher pT → smaller ΔR (more collimated)

**Z_Eta, Z_Phi:**
- Z_Eta should be roughly uniform around 0 (central production)
- Z_Phi should be uniform in [-π, π]

## Troubleshooting

**Q: Z_Mass peak is too broad or shifted**
- Model isn't learning Z mass constraint well
- May need more training or more capacity
- Check if individual lepton pT/eta/phi distributions are correct first

**Q: Z_DeltaR looks wrong**  
- Model may not be learning correlations between leptons
- Check if this is an autoregressive flow - feature ordering matters
- Verify leptons are close to each other in feature order

**Q: All derived quantities look random**
- Model completely failed to learn physics
- Check basic feature distributions first
- Verify training converged (low validation loss)
- Check for preprocessing bugs (wrong scalers, feature order mismatch)

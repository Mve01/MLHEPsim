# ttZ Sample Analysis Tools

This directory contains tools for analyzing ttZ samples from generative models.

## Overview

This analyzer validates ttZ data (3 leptons + 1 b-jet + MET = 15 features). It focuses on:
- Loading trained models and generating samples
- Comparing feature distributions between real and generated data
- Computing physics-based derived quantities (Z mass, W transverse mass, event kinematics)
- Correlation analysis

## File Structure

```
sample_analyzer/
├── analyzer.py                    # Main ttzSampleAnalyzer class
├── physics_utils.py              # Physics calculations (Z, W, top kinematics)
├── plotting.py                   # ttZ plotting functions
├── run_analysis.py               # Main execution script
├── run_smeft_vs_sm_comparison.py # SMEFT vs SM reweighting comparison
├── smeft_vs_sm_plotting.py       # SMEFT comparison visualization
├── figures/                      # Output directory for plots
└── README.md                     # This file
```

## Quick Start

### Run Analysis

```bash
python ml/custom/ttz/sample_analyzer/run_analysis.py
```

This will:
1. Load the trained model and preprocessed ttZ data
2. Generate samples from the model
3. Create comparison plots for all 15 features
4. Compute and display physics-based derived quantities
5. Save plots to `figures/`
analyzer = ttzSampleAnalyzer(
    data_dir="ml/data/ttz/ttz.npy",
    variables_json_path="ml/data/ttz/variables.json",
    model_name="MAFMADEMOG_flow_model_gauss_rank"
)

# Generate samples
analyzer.generate_samples(n_samples=100000)

# Plot comparisons
analyzer.plot_all()
analyzer.compute_masses()

# For reduced model: Compute PT of negative muon from mass
analyzer.compute_pt_negative()

# Create plots (all save to figures/ directory)
analyzer.plot_feature_comparison()
analyzer.plot_invariant_mass()  # Or plot_pt_negative_comparison() for reduced model
analyzer.plot_correlation_plots(gridsize=200)
```

## Module Documentation

## Physics-Based Validation

The analyzer computes derived physics quantities to validate model quality beyond individual feature distributions:

### Z Boson Kinematics
From Z_Lepton1 and Z_Lepton2:
- **Z_Pt**: Transverse momentum
- **Z_Eta**: Pseudorapidity
- **Z_Phi**: Azimuthal angle
- **Z_Mass**: Invariant mass (should peak at ~91.2 GeV)
- **Z_DeltaR**: Angular separation between leptons

### W Boson Kinematics
From W_Lepton and MET:
- **W_Pt**: Transverse momentum
- **W_Phi**: Azimuthal angle
- **W_MT**: Transverse mass (should peak at ~80 GeV)

### Top Quark Kinematics
From BJet, W_Lepton, and MET:
- **Top_Pt**: Transverse momentum
- **Top_Phi**: Azimuthal angle
- **Top_MT**: Transverse mass (should peak at ~160 GeV)

These derived quantities provide stringent tests of whether the model has learned proper correlations between particles and physics constraints.

## Module Documentation

### `analyzer.py`
Main analyzer class that orchestrates the analysis pipeline.

**Key Methods:**
- `generate_samples()` - Generate samples from trained model
- `plot_feature_comparison()` - Plot individual feature distributions
- Additional methods for physics validation

### `physics_utils.py`
Physics calculations for particle kinematics and invariant masses.

**Functions:**
- `calculate_invariant_mass()` - Compute invariant mass from 4-momenta
- `calculate_delta_r()` - Angular separation between particles
- `calculate_z_kinematics()` - Z boson 4-momentum and kinematics
- `calculate_w_kinematics()` - W boson transverse mass and kinematics
- `calculate_top_kinematics()` - Top quark transverse mass and kinematics

### `plotting.py`
All visualization functions.

**Functions:**
- `plot_feature_comparison()` -Compare real vs generated features
- Physics quantity comparison plots
- Correlation analysis

## Configuration

Edit `run_analysis.py` to modify:
- Data paths
- Model configuration:
  - `variables_json`: Choose `'ml/data/drellyan/variables.json'` (full) or `'ml/data/drellyan/variables_reduced.json'` (reduced)
- Model name
- Number of samples to generate
- Plot parameters:
  - `gridsize`: Number of bins for 2D histograms in correlation plots (default: 200)
  - Bins for 1D histograms (default: 100)

## Output

All plots are saved to the `sample_analyzer/figures/` directory (auto-created):

### Full Model (`variables.json`)
- `feature_comparison.png` - Individual feature distributions with ratio plots
- `invariant_mass_comparison.png` - Dimuon mass distribution
- `correlation_comparison.png` - 2D correlation plots (PT vs PT, Eta vs Eta, Phi vs Phi)

### Reduced Model (`variables_reduced.json`)
- `feature_comparison.png` - Individual feature distributions with ratio plots
- `pt_negative_comparison.png` - PT negative muon distribution (derived from mass)
- `correlation_comparison.png` - 2D correlation plots (Eta vs Eta, Phi vs Phi only)

## Key Features

### Adaptive Correlation Plots
The correlation plotting automatically detects which feature set is being used:
- **Full model**: Plots PT₁ vs PT₂, η₁ vs η₂, φ₁ vs φ₂ using Lead/Sub naming
- **Reduced model**: Plots η₁ vs η₂, φ₁ vs φ₂ only (no PT correlation) using Pos/Neg naming

### Outlier Handling
- **Feature plots**: Use percentile-based range limits (1th to 99th percentile) to exclude extreme outliers
- **PT negative plot**: Automatically excludes outliers to focus on the bulk distribution
- **Correlation plots**: Full range shown, 2D histograms handle density naturally

### 2D Histogram Visualization
All correlation plots use 2D histograms exclusively:
- **Real data**: Blue colormap
- **Generated data**: Red colormap
- **Difference map**: Diverging red-blue colormap showing (Real - Generated) density
- Controllable resolution via `gridsize` parameter

## Notes

- **Cache disabled**: Samples are always freshly generated (important after retraining)
- **Preprocessing**: Uses Gaussian rank transformation for continuous features
- **Mass cuts**: Training data has 110-160 GeV mass window (configured in `process_drellyan_dataset.py`)
- **Feature naming**: 
  - Full model uses `Lead/Sub` convention (leading/subleading muons by PT)
  - Reduced model uses `Pos/Neg` convention (positive/negative charge muons)
- **PT calculation**: For reduced model, PT₂ is derived using: PT₂ = M²/(2·PT₁·(cosh(Δη) - cos(Δφ)))

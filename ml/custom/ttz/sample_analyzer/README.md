# ttZ Sample Analysis Tools

This directory contains minimal tools for analyzing ttZ samples from generative models.

## Overview

This is a simplified analyzer for ttZ data (3 leptons + 3 jets = 24 features). It focuses on:
- Loading trained models and generating samples
- Comparing feature distributions between real and generated data
- Basic correlation plots

**Note**: This analyzer does NOT include complex physics calculations (mass reconstruction, top quark kinematics, etc.). It's designed for basic distribution validation.

## File Structure

```
sample_analyzer/
├── analyzer_simple.py       # Main ttzSampleAnalyzer class
├── plotting.py              # Clean ttZ plotting functions
├── run_analysis.py          # Main execution script
├── figures/                 # Output directory for plots
├── analyzer_drellyan_old.py # (Legacy - old Drell-Yan analyzer)
├── plotting_drellyan_old.py # (Legacy - old Drell-Yan plots)
├── mass_calculator.py       # (Legacy from Drell-Yan - not used)
├── pt_calculator.py         # (Legacy from Drell-Yan - not used)
├── system_calculator.py     # (Legacy from Drell-Yan - not used)
└── README.md                # This file
```

## Quick Start

### Run Analysis

```bash
python ml/custom/ttz/sample_analyzer/run_analysis.py
```

This will:
1. Load the trained model and preprocessed ttZ data
2. Generate samples from the model
3. Create comparison plots:
   - Feature distributions (24 features)
   - Correlation plots

### Use as a Library

```python
from ml.custom.ttz.sample_analyzer.analyzer import ttzSampleAnalyzer

# Initialize
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

### `analyzer.py`
Main analyzer class that orchestrates the analysis pipeline.

**Key Methods:**
- `generate_samples()` - Generate samples from trained model
- `compute_masses()` - Calculate invariant masses (for full model)
- `compute_pt_negative()` - Calculate PT of negative muon from mass (for reduced model)
- `plot_feature_comparison()` - Plot 1D feature distributions with ratio plots
- `plot_invariant_mass()` - Plot invariant mass distribution
- `plot_pt_negative_comparison()` - Plot PT negative distribution
- `plot_correlation_plots()` - Plot 2D correlations with difference maps

### `mass_calculator.py`
Physics calculations for dimuon invariant mass.

**Functions:**
- `calculate_dimuon_invariant_mass()` - Compute M from kinematics using M² = 2·PT₁·PT₂·(cosh(Δη) - cos(Δφ))
- `compute_masses_for_dataset()` - Helper to compute for full dataset

### `pt_calculator.py`
Physics calculations for negative muon transverse momentum.

**Functions:**
- `calculate_pt_negative_muon()` - Compute PT₂ from mass and kinematics using PT₂ = M²/(2·PT₁·(cosh(Δη) - cos(Δφ)))
- `compute_pt_negative_for_dataset()` - Helper to compute for full dataset

### `plotting.py`
All visualization functions using 2D histograms exclusively.

**Functions:**
- `plot_feature_comparison()` - Compare real vs generated features with Generated/Real ratio plots
- `plot_invariant_mass()` - Compare invariant mass distributions
- `plot_pt_negative_comparison()` - Compare PT negative distributions with outlier exclusion
- `plot_correlation_comparison()` - 2D correlation plots with difference maps (auto-detects Lead/Sub vs Pos/Neg naming)
- `get_range_limits()` - Helper for plot ranges using percentile-based outlier exclusion

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

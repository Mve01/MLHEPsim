#!/usr/bin/env python
"""Script to compare SM and SMEFT models and data."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

import matplotlib
matplotlib.use('Agg')

import numpy as np
import json
import pandas as pd
import uproot
from analyzer import ttzSampleAnalyzer
from smeft_vs_sm_plotting import create_both_scale_plots


def main():
    """Run SM vs SMEFT comparison."""
    
    print("="*60)
    print("SM vs SMEFT Comparison Analysis")
    print("="*60)
    
    # Paths
    data_dir = "ml/data/ttz/ttz.npy"
    variables_json = "ml/data/ttz/variables.json"
    output_path = "ml/custom/ttz/sample_analyzer/figures/sm_vs_smeft_comparison.png"
    
    # Model names - use the registered MLflow model names
    # Note: These are the auto-generated names from training runs
    sm_model_name = "MAFMADEMOG_flow_model_gauss_rank_20260122_nall"  # SM weights
    smeft_model_name = "MAFMADEMOG_flow_model_gauss_rank_20260123_nall"  # cHt=5.0 weights
    
    print(f"\nUsing models:")
    print(f"  SM model: {sm_model_name}")
    print(f"  SMEFT model: {smeft_model_name}")
    
    # Initialize analyzers for both models
    print("\nInitializing SM analyzer...")
    sm_analyzer = ttzSampleAnalyzer(
        data_dir=data_dir,
        variables_json_path=variables_json,
        model_name=sm_model_name
    )
    
    print("Initializing SMEFT analyzer...")
    smeft_analyzer = ttzSampleAnalyzer(
        data_dir=data_dir,
        variables_json_path=variables_json,
        model_name=smeft_model_name
    )
    
    # Generate samples from both models
    n_samples = 1000000
    print(f"\nGenerating {n_samples} samples from SM model...")
    sm_analyzer.generate_samples(n_samples=n_samples, chunks=10, debug=True)
    
    print(f"\nGenerating {n_samples} samples from SMEFT model...")
    smeft_analyzer.generate_samples(n_samples=n_samples, chunks=10, debug=True)
    
    # Get the real data (same physics data, but we'll load it separately for each analyzer)
    # Note: Both analyzers load the same ttz.npy file
    sm_real_data = sm_analyzer.original_data[:, :21]  # Remove weight column, keep only features
    smeft_real_data = smeft_analyzer.original_data[:, :21]  # Same features
    
    # Load BOTH weight sets from the ROOT file for proper weighted histograms
    print("\nLoading SM and SMEFT weights from ROOT file...")
    root_file_path = "/project/atlas/users/kdevries/FinalStateTransformer/datasets/Tom_Original/full_ttZ.root"
    with uproot.open(root_file_path) as f:
        tree = f["Events"]
        sm_weights = tree["eventWeight"].array(library="np")
        smeft_weights_array = tree["smeft_weights"].array(library="np")
        smeft_weights = np.array([w[124] for w in smeft_weights_array])  # Extract cHt=5.0 weights
    
    print(f"  SM weights: min={sm_weights.min():.6e}, max={sm_weights.max():.6e}, mean={sm_weights.mean():.6e}")
    print(f"  SMEFT weights: min={smeft_weights.min():.6e}, max={smeft_weights.max():.6e}, mean={smeft_weights.mean():.6e}")
    
    # Get generated data
    sm_gen_data = sm_analyzer.generated_data
    smeft_gen_data = smeft_analyzer.generated_data
    
    # Get feature names and other metadata
    selected_features = sm_analyzer.selected_features
    selection = sm_analyzer.selection
    
    # Load variables
    with open(variables_json, 'r') as f:
        variables = json.load(f)
    
    print(f"\nCreating comparison plots...")
    print(f"  Features: {len(selected_features)}")
    print(f"  SM real data shape: {sm_real_data.shape}")
    print(f"  SM generated data shape: {sm_gen_data.shape}")
    print(f"  SMEFT real data shape: {smeft_real_data.shape}")
    print(f"  SMEFT generated data shape: {smeft_gen_data.shape}")
    
    # Create the comparison plots (both normal and log scale)
    create_both_scale_plots(
        sm_real_data=sm_real_data,
        sm_gen_data=sm_gen_data,
        smeft_real_data=smeft_real_data,
        smeft_gen_data=smeft_gen_data,
        sm_real_weights=sm_weights,
        smeft_real_weights=smeft_weights,
        selected_features=selected_features,
        selection=selection,
        variables=variables,
        output_path=output_path,
        bins=50,
        n_cols=3
    )
    
    print("\n" + "="*60)
    print("Comparison complete!")
    print(f"Plots saved to:")
    print(f"  {output_path}")
    print(f"  {output_path.replace('.png', '_log.png')}")
    print("="*60)


if __name__ == "__main__":
    main()

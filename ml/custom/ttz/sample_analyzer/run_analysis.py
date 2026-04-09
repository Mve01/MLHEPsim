#!/usr/bin/env python
"""Main script to run ttZ sample analysis."""

import sys
import os
import argparse
from pathlib import Path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive

from analyzer import ttzSampleAnalyzer


def get_latest_ttz_model():
    """Find the most recent ttZ model based on date and n_data in model name."""
    import re
    import mlflow
    
    models_dir = "mlruns/models"
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    # List all models
    all_models = os.listdir(models_dir)
    
    # Filter for ttZ models (MAFMADEMOG with date but without 'sidebands' or 'full_data')
    ttz_models = [m for m in all_models 
                  if m.startswith("MAFMADEMOG_flow_model_gauss_rank_") 
                  and "sidebands" not in m 
                  and "full_data" not in m
                  and "_202" in m]  # Has a date
    
    if not ttz_models:
        raise ValueError("No ttZ models found")
    
    # Sort by date and n_data: extract date (YYYYMMDD) and n_data number
    def model_sort_key(model_name):
        # Extract date: YYYYMMDD format
        date_match = re.search(r'_(\d{8})_', model_name)
        date = int(date_match.group(1)) if date_match else 0
        
        # Extract n_data: either 'nXXXXX' or 'nall'
        ndata_match = re.search(r'_n(\d+)$', model_name)
        ndata = int(ndata_match.group(1)) if ndata_match else 999999999  # 'nall' gets highest priority
        
        return (date, ndata)
    
    ttz_models.sort(key=model_sort_key)
    latest_model = ttz_models[-1]
    
    # Get version information from MLflow
    try:
        mlflow_models = {}
        for r in mlflow.MlflowClient().search_model_versions():
            if r.name not in mlflow_models:
                mlflow_models[r.name] = []
            mlflow_models[r.name].append(int(r.version))
        
        if latest_model in mlflow_models:
            versions = sorted(mlflow_models[latest_model])
            latest_version = versions[-1]
            print(f"Found {len(ttz_models)} ttZ model(s)")
            print(f"Using latest model: {latest_model} (version {latest_version} of {len(versions)})")
        else:
            print(f"Found {len(ttz_models)} ttZ model(s)")
            print(f"Using latest model: {latest_model}")
    except Exception as e:
        print(f"Found {len(ttz_models)} ttZ model(s)")
        print(f"Using latest model: {latest_model} (could not fetch version info: {e})")
    
    return latest_model


def main():
    """Run the complete ttZ analysis pipeline."""

    parser = argparse.ArgumentParser(description="Run ttZ sample analysis.")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="MLflow model name to load. Defaults to the latest model found in mlruns/models.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Directory to save figures. Defaults to sample_analyzer/figures/.",
    )
    args = parser.parse_args()

    print("="*60)
    print("ttZ Sample Analysis")
    print("="*60)

    # Configuration
    # Use the direct-phi TTZ cache by default.
    # If weight columns are present, the analyzer handles them dynamically.
    data_dir = "ml/data/ttz/ttz_ptetaphi_weights.npy"
    variables_json = "ml/data/ttz/variables.json"
    model_name = args.model_name or get_latest_ttz_model()
    figures_dir = args.figures_dir  # None → analyzer uses its own default

    # Initialize analyzer
    print("\nInitializing ttZ analyzer...")
    print("\nUsing model:", model_name)
    analyzer = ttzSampleAnalyzer(
        data_dir=data_dir,
        variables_json_path=variables_json,
        model_name=model_name
    )

    # Generate samples
    print("\nGenerating samples from model...")
    n_samples = 1000000
    analyzer.generate_samples(n_samples=n_samples, chunks=10, debug=True)

    # Create all plots
    print("\nCreating comparison plots...")
    analyzer.plot_all(figures_dir=figures_dir)

    out_dir = figures_dir or "ml/custom/ttz/sample_analyzer/figures/"
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Plots saved to: {out_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

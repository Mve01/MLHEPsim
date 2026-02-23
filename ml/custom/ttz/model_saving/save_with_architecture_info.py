#!/usr/bin/env python
"""Helper script to save models with architecture-based names."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

import mlflow
import argparse
from datetime import datetime


def get_model_info(model_name, version=None):
    """Get information about a registered model."""
    client = mlflow.MlflowClient()
    
    try:
        if version is None:
            # Get latest version
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                return None
            version = max([int(v.version) for v in versions])
        
        model_version = client.get_model_version(model_name, version)
        
        # Load the model to get architecture info
        model_uri = f"models:/{model_name}/{version}"
        model = mlflow.pytorch.load_model(model_uri)
        
        # Extract architecture info from the module
        if hasattr(model, 'config'):
            config = model.config
            n_flows = config.get('n_flows', 'unknown')
            hidden_dims = config.get('hidden_dims', 'unknown')
        else:
            n_flows = 'unknown'
            hidden_dims = 'unknown'
        
        return {
            'name': model_name,
            'version': version,
            'n_flows': n_flows,
            'hidden_dims': hidden_dims,
            'run_id': model_version.run_id,
            'creation_time': datetime.fromtimestamp(model_version.creation_timestamp/1000)
        }
    except Exception as e:
        print(f"Error getting model info: {e}")
        return None


def list_recent_models():
    """List recent ttZ models with their architecture info."""
    models_dir = "mlruns/models"
    if not os.path.exists(models_dir):
        print(f"Models directory not found: {models_dir}")
        return []
    
    # Get all registered models
    all_models = sorted(
        [m for m in os.listdir(models_dir) 
         if m.startswith("MAFMADEMOG_flow_model_gauss_rank_") 
         and "sidebands" not in m],
        key=lambda x: x,
        reverse=True
    )[:10]  # Get 10 most recent
    
    client = mlflow.MlflowClient()
    
    results = []
    for model_name in all_models:
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
            if versions:
                version = max([int(v.version) for v in versions])
                
                # Try to load config from the checkpoint
                model_version = client.get_model_version(model_name, version)
                run_id = model_version.run_id
                
                # Try to get architecture from artifacts
                artifacts_path = f"mlruns/0/{run_id}/artifacts"
                checkpoint_path = f"{artifacts_path}/model/checkpoint.ckpt"
                
                if os.path.exists(checkpoint_path):
                    import torch
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    
                    # Try to extract config
                    n_flows = checkpoint.get('hyper_parameters', {}).get('n_flows', 'N/A')
                    hidden_dims = checkpoint.get('hyper_parameters', {}).get('hidden_dims', 'N/A')
                else:
                    n_flows = 'N/A'
                    hidden_dims = 'N/A'
                
                results.append({
                    'name': model_name,
                    'version': version,
                    'n_flows': n_flows,
                    'hidden_dims': hidden_dims,
                    'run_id': run_id,
                    'date': model_name.split('_')[-2]
                })
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Save models with architecture info')
    parser.add_argument('--source', type=str, help='Source model name')
    parser.add_argument('--version', type=int, default=None, help='Model version (default: latest)')
    parser.add_argument('--name', type=str, help='New model name')
    parser.add_argument('--list', action='store_true', help='List recent models')
    
    args = parser.parse_args()
    
    if args.list:
        print("\n" + "="*80)
        print("Recent ttZ Models")
        print("="*80)
        models = list_recent_models()
        for i, model in enumerate(models, 1):
            print(f"\n{i}. {model['name']}")
            print(f"   Version: {model['version']}")
            print(f"   Architecture: {model['n_flows']} flows × {model['hidden_dims']} hidden dims")
            print(f"   Date: {model['date']}")
            print(f"   Run ID: {model['run_id'][:8]}...")
        print("\n" + "="*80)
        return
    
    if not args.source or not args.name:
        print("Error: --source and --name are required")
        print("Use --list to see available models")
        return
    
    # Use the existing save script with proper arguments
    import subprocess
    cmd = [
        'python', 
        'ml/custom/ttz/model_saving/save_model_with_name.py',
        '--source', args.source,
        '--name', args.name
    ]
    if args.version:
        cmd.extend(['--version', str(args.version)])
    
    subprocess.run(cmd)


if __name__ == "__main__":
    main()

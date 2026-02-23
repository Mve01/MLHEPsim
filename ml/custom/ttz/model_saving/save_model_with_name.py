#!/usr/bin/env python
"""Script to register a model with a custom user-friendly name."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

import mlflow
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def save_model_with_custom_name(source_model_name, new_model_name, version=None, description=None):
    """
    Register an existing MLflow model under a new user-friendly name using MLflow's proper API.
    
    This properly registers the model in MLflow's tracking database, preserving all metadata
    including scalers, selection, and model architecture.
    
    Parameters
    ----------
    source_model_name : str
        The current registered model name (e.g., MAFMADEMOG_flow_model_gauss_rank_20260218_nall)
    new_model_name : str
        The new user-friendly name (e.g., ttz_cHt5_phi_scaled)
    version : int, optional
        The version to copy. If None, uses the latest version.
    description : str, optional
        Description for the new model version
    """
    mlflow.set_tracking_uri("file:./mlruns")
    client = mlflow.MlflowClient()
    
    try:
        # Search for source model versions
        source_versions = client.search_model_versions(f"name='{source_model_name}'")
        if not source_versions:
            raise ValueError(f"No versions found for model: {source_model_name}")
        
        # Determine which version to use
        if version is None:
            version = max([int(v.version) for v in source_versions])
        
        # Get the source model version object
        source_version_obj = [v for v in source_versions if int(v.version) == version][0]
        source_uri = f"models:/{source_model_name}/{version}"
        
        print(f"Loading model: {source_model_name} (version {version})")
        print(f"  Source URI: {source_uri}")
        
        # Load the complete model (includes model, scalers, selection, etc.)
        loaded_module = mlflow.pytorch.load_model(source_uri)
        
        print(f"  Model loaded successfully")
        print(f"  - Has scalers: {hasattr(loaded_module, 'scalers')}")
        print(f"  - Has selection: {hasattr(loaded_module, 'selection')}")
        
        # Create description
        if description is None:
            desc_parts = [f"Copy of {source_model_name} v{version}"]
            if source_version_obj.description:
                desc_parts.append(source_version_obj.description)
            description = " | ".join(desc_parts)
        
        print(f"\nRegistering as new model: {new_model_name}")
        
        # Log and register under new name
        # We need to start a new run to log the model
        with mlflow.start_run(run_name=f"register_{new_model_name}"):
            # Log the model with the new name
            mlflow.pytorch.log_model(
                loaded_module,
                artifact_path="model",
                registered_model_name=new_model_name
            )
            
            # Log some metadata
            mlflow.set_tag("source_model", source_model_name)
            mlflow.set_tag("source_version", version)
            mlflow.set_tag("registration_date", "2026-02-18")
        
        # Update the description of the newly registered model
        new_versions = client.search_model_versions(f"name='{new_model_name}'")
        latest_new_version = max([int(v.version) for v in new_versions])
        
        client.update_model_version(
            name=new_model_name,
            version=latest_new_version,
            description=description
        )
        
        print(f"\n{'='*60}")
        print(f"✓ Successfully registered model as: {new_model_name}")
        print(f"  Version: {latest_new_version}")
        print(f"  Description: {description}")
        print(f"\nYou can now use this model with:")
        print(f'  analyzer = ttzSampleAnalyzer(..., model_name="{new_model_name}")')
        print(f"{'='*60}\n")
        
        return new_model_name, latest_new_version
        
    except Exception as e:
        logging.error(f"Failed to register model: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("Register Models with Custom Names")
    print("="*60)
    print()
    
    # Get user input or use defaults
    import argparse
    parser = argparse.ArgumentParser(description="Register MLflow model with user-friendly name")
    parser.add_argument("--source", type=str, 
                       default="MAFMADEMOG_flow_model_gauss_rank_20260218_nall",
                       help="Source model name")
    parser.add_argument("--name", type=str, default=None,
                       help="New user-friendly name (default: keeps original name)")
    parser.add_argument("--version", type=int, default=None,
                       help="Source model version (default: latest)")
    parser.add_argument("--description", type=str, default=None,
                       help="Description for the new model")
    
    args = parser.parse_args()
    
    # If no custom name provided, use the source name
    if args.name is None:
        args.name = args.source
        print("⚠️  WARNING: No custom name provided via --name argument.")
        print(f"   Registering with the same name as source: '{args.name}'")
        print(f"   This will create a NEW VERSION of the existing model.")
        print(f"   To use a different name, run with: --name <new_model_name>\n")
        
        # Ask for confirmation
        try:
            response = input("Do you want to continue? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("\nCancelled by user.")
                sys.exit(0)
        except (KeyboardInterrupt, EOFError):
            print("\n\nCancelled by user.")
            sys.exit(0)
    
    # Check if source and target names are the same (even if explicitly set)
    if args.name == args.source:
        logging.warning(f"Source and target model names are identical: '{args.name}'")
        logging.warning("This will create a NEW VERSION of the existing model, not a copy.")
    
    # Register the model
    save_model_with_custom_name(
        source_model_name=args.source,
        new_model_name=args.name,
        version=args.version,
        description=args.description
    )
    
    print("\nDone! Model is now registered.")
    print("To register with a different name, run:")
    print("  python save_model_with_name.py --source <source_name> --name <new_name>")

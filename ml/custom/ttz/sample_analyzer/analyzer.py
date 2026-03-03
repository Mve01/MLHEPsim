"""Minimal ttZ sample analyzer for basic feature comparisons."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

import numpy as np
import json
import logging
import yaml
from ml.common.nn.gen_model_sampler import GenModelSampler
from ml.common.data_utils.feature_scaling import RescalingHandler
from ml.common.data_utils.processors import Preprocessor
from ml.custom.ttz.process_ttz_dataset import ttzFeatureSelector, ttzNpyProcessor

# Import plotting and physics utilities
from plotting import plot_feature_comparison
from physics_utils import (
    calculate_z_kinematics,
    calculate_w_kinematics,
    calculate_top_kinematics
)


class ttzSampleAnalyzer:
    """Minimal analyzer for ttZ samples - focused on basic feature comparisons."""
    
    def __init__(self, data_dir, variables_json_path, model_name="MAFMADEMOG_flow_model_gauss_rank"):
        """
        Initialize the ttZ analyzer.
        
        Parameters
        ----------
        data_dir : str
            Path to the preprocessed ttZ data .npy file
        variables_json_path : str
            Path to the variables.json file
        model_name : str
            Name of the model to load for generation
        """
        self.data_dir = data_dir
        self.variables_json_path = variables_json_path
        self.model_name = model_name
        
        # Load variables configuration
        with open(variables_json_path, 'r') as f:
            self.variables = json.load(f)
        
        # Load ORIGINAL unprocessed data directly for comparison
        original_data_path = "ml/data/ttz/ttz.npy"
        if not os.path.exists(original_data_path):
            raise FileNotFoundError(f"Original data file not found: {original_data_path}")
        
        full_data = np.load(original_data_path)
        logging.info(f"Loaded full original ttZ data shape: {full_data.shape}")
        
        # Use FULL dataset for comparison (training + validation + test)
        # Note: This provides more statistics for comparison plots than test set alone
        self.original_data = full_data
        logging.info(f"Using FULL DATASET for comparison: {self.original_data.shape}")
        
        # Also load preprocessed data (for reference)
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Preprocessed data file not found: {data_dir}")
        
        self.preprocessed_data = np.load(data_dir)
        logging.info(f"Loaded preprocessed ttZ data shape: {self.preprocessed_data.shape}")
        
        # Initialize preprocessing
        self._setup_preprocessing()
        
        # Storage for generated data
        self.generated_data = None
    
    def _setup_preprocessing(self):
        """Setup preprocessing pipeline and scalers."""
        # Load preprocessing config to match training setup
        config_path = os.path.join(project_root, "ml/custom/ttz/config/flows/data_config.yaml")
        with open(config_path, 'r') as f:
            data_config = yaml.safe_load(f)
        preprocessing_config = data_config['data_config']['preprocessing']
        
        # Get feature names (now only 'cont' and 'disc' types with Cartesian coordinates)
        features = [name for name, type_ in self.variables['colnames'].items() 
                   if type_ in ['cont', 'uni', 'disc']]  # Include all feature types
        
        # Initialize processor and selector
        npy_proc = ttzNpyProcessor(
            data_dir="ml/data/ttz/", 
            base_file_name="ttz", 
            list_data_features=features
        )
        
        f_sel = ttzFeatureSelector(
            file_path=npy_proc.npy_file, 
            features=self.variables,
            drop_types=['label'],  # Keep disc features (charges) - model was trained with them
        )
        
        self.selection = f_sel._select_colnames()
        
        # Load scalers from trained model checkpoint instead of refitting
        # This ensures we use the exact same CDFs (fitted on training data) for inverse transform
        try:
            from ml.common.utils.register_model import fetch_registered_module
            logging.info(f"Loading scalers from model checkpoint: {self.model_name}")
            module = fetch_registered_module(self.model_name, model_version=-1, device="cpu")
            
            if hasattr(module, 'scalers') and module.scalers is not None:
                self.scalers = module.scalers
                logging.info("✓ Successfully loaded scalers from model checkpoint (fitted on training set)")
                # Debug: Check scaler dimensions
                if 'cont' in self.scalers and self.scalers['cont']:
                    for scaler_name, scaler in self.scalers['cont']:
                        if scaler_name == 'gauss rank transform':
                            n_interp = len(scaler.interp_funcs_lst)
                            logging.info(f"  Gauss rank scaler has {n_interp} interpolation functions")
            else:
                raise AttributeError("Model checkpoint does not contain saved scalers")
                
            if hasattr(module, 'selection') and module.selection is not None:
                self.selection = module.selection
                logging.info("✓ Successfully loaded feature selection from model checkpoint")
                logging.info(f"  Selection shape: {self.selection.shape}")
                n_cont = len(self.selection[self.selection['type'].isin(['cont', 'uni'])])
                n_disc = len(self.selection[self.selection['type'] == 'disc'])
                logging.info(f"  Features: {n_cont} continuous/uni, {n_disc} discrete")
            else:
                logging.warning("Model checkpoint does not contain selection - using local selection")
                
        except Exception as e:
            logging.error(f"Failed to load scalers from checkpoint: {e}")
            logging.warning("FALLBACK: Refitting scalers on full dataset (may cause distribution mismatch!)")
            
            # Fallback: refit scalers (not recommended - will cause CDF mismatch)
            original_data_path = "ml/data/ttz/ttz.npy"
            if not os.path.exists(original_data_path):
                raise FileNotFoundError(f"Original data file not found: {original_data_path}")
            
            original_data = np.load(original_data_path)
            logging.info(f"Loaded original ttZ data shape: {original_data.shape}")
            
            pre = Preprocessor(**preprocessing_config)
            _, self.selection, self.scalers = pre(original_data, self.selection)
        
        # Remove weight column from selection to match model output dimensions
        self.selection = self.selection[self.selection['feature'] != 'cHt_weight'].reset_index(drop=True)
        logging.info(f"After removing weight: selection shape = {self.selection.shape}")
        n_cont = len(self.selection[self.selection['type'].isin(['cont', 'uni'])])
        n_disc = len(self.selection[self.selection['type'] == 'disc'])
        logging.info(f"  Features after weight removal: {n_cont} continuous/uni, {n_disc} discrete")
        
        # Setup rescaling handler
        self.rescale_handler = RescalingHandler(self.selection, self.scalers)
        
        # Use the feature order from selection dataframe
        self.selected_features = self.selection['feature'].tolist()
        
        # Extract weight column if present
        logging.info(f"Checking for weights in data: shape = {self.original_data.shape}")
        print(f"\n=== Weight Extraction Debug ===")
        print(f"Original data shape: {self.original_data.shape}")
        if self.original_data.shape[1] == 16:  # 15 features (cylindrical coords, no charges) + 1 weight
            self.original_weights = self.original_data[:, -1]  # Last column is weight
            self.original_data = self.original_data[:, :-1]  # Remove weight from features
            print(f"✓ Extracted cHt=5.0 weights from data")
            print(f"  Weights: min={np.min(self.original_weights):.6e}, max={np.max(self.original_weights):.6e}, mean={np.mean(self.original_weights):.6e}")
            print(f"  Data after extraction: {self.original_data.shape}, Weights: {self.original_weights.shape}")
            logging.info(f"✓ Extracted cHt=5.0 weights from data: min={np.min(self.original_weights):.6e}, "
                        f"max={np.max(self.original_weights):.6e}, mean={np.mean(self.original_weights):.6e}")
            logging.info(f"  After weight extraction: data shape = {self.original_data.shape}, weights shape = {self.original_weights.shape}")
        else:
            self.original_weights = None
            print(f"✗ WARNING: No weights found!")
            print(f"  Expected 16 columns (15 cylindrical features + 1 weight), got {self.original_data.shape[1]}")
            print(f"  MC histograms will be UNWEIGHTED!")
            logging.warning(f"✗ No weights found in data! Expected 16 columns (15 cylindrical features + 1 weight), got {self.original_data.shape[1]}")
            logging.warning("  MC histograms will be UNWEIGHTED - this is incorrect for cHt=5.0 comparison!")
        print("="*40 + "\n")
        
        # Note: Data is now created in physics-motivated order (1 Jet → 3 Leptons → MET) directly
        # from ROOT files, so no reordering is needed. Old .npy files may need to be regenerated.
        
    def generate_samples(self, n_samples=100000, chunks=10, debug=False):
        """
        Generate samples from the trained model.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        chunks : int
            Number of chunks to split generation into
        debug : bool
            Whether to print debug information
            
        Returns
        -------
        np.ndarray
            Generated samples in original (physical) units
        """
        # Initialize sampler
        sampler = GenModelSampler(
            model_names=self.model_name, 
            save_dir="ml/data/ttz",
            file_name="ttz_generated",
            disable_cache=True  # Always generate fresh samples
        )
        
        # Generate samples
        print(f"Generating {n_samples} samples...")
        generated_samples = sampler.sample(n_samples, chunks=chunks)
        generated_data = generated_samples[self.model_name][0]
        
        # Check for and remove NaN values if present
        if np.isnan(generated_data).any():
            nan_mask = np.isnan(generated_data)
            nan_rows = np.where(nan_mask.any(axis=1))[0]
            if len(nan_rows) > 0 and len(nan_rows) < 100:
                logging.warning(f"Removing {len(nan_rows)} rows with NaN values")
                generated_data = generated_data[~nan_mask.any(axis=1)]
        
        if debug:
            print("\n=== Before inverse_transform ===")
            print(f"Generated data shape: {generated_data.shape}")
            print(f"Selection shape: {self.selection.shape}")
            for i, feature in enumerate(self.selected_features):
                col_data = generated_data[:, i]
                print(f"{feature}: min={col_data.min():.3f}, "
                      f"max={col_data.max():.3f}, mean={col_data.mean():.3f}, "
                      f"nan_count={np.isnan(col_data).sum()}")
        
        # Apply inverse transform to convert from scaled space to physical units
        logging.info(f"Applying inverse transform: data shape = {generated_data.shape}")
        self.generated_data = self.rescale_handler.inverse_transform(generated_data)
        
        if debug:
            for i, feature in enumerate(self.selected_features):
                col_data = self.generated_data[:, i]
                print(f"{feature}: min={col_data.min():.3f}, "
                      f"max={col_data.max():.3f}, mean={col_data.mean():.3f}, "
                      f"nan_count={np.isnan(col_data).sum()}")
        
        return self.generated_data
    
    def plot_feature_comparison(self, output_path=None, bins=50, n_cols=3, include_derived=True):
        """
        Plot comparison of all features between real and generated data.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the plot
        bins : int
            Number of histogram bins
        n_cols : int
            Number of columns in the plot grid
        include_derived : bool
            Whether to include derived Z kinematics variables
        """
        if self.generated_data is None:
            raise ValueError("No generated data available. Call generate_samples() first.")
        
        if output_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            figures_dir = os.path.join(script_dir, 'figures')
            os.makedirs(figures_dir, exist_ok=True)
            output_path = os.path.join(figures_dir, 'feature_comparison.png')
        
        real_data = self.original_data
        gen_data = self.generated_data
        features = self.selected_features.copy()
        
        # Calculate and add derived Z kinematics if requested
        if include_derived:
            print("Calculating derived physics variables...")
            
            # Calculate Z kinematics for real data
            real_z_kin = calculate_z_kinematics(real_data, features)
            
            # Calculate Z kinematics for generated data
            gen_z_kin = calculate_z_kinematics(gen_data, features)
            
            # Calculate W kinematics for real data
            real_w_kin = calculate_w_kinematics(real_data, features)
            
            # Calculate W kinematics for generated data
            gen_w_kin = calculate_w_kinematics(gen_data, features)
            
            # Calculate top kinematics for real data
            real_top_kin = calculate_top_kinematics(real_data, features)
            
            # Calculate top kinematics for generated data
            gen_top_kin = calculate_top_kinematics(gen_data, features)
            
            # Add all derived variables to data arrays
            derived_features = [
                'Z_Pt', 'Z_Eta', 'Z_Phi', 'Z_Mass', 'Z_DeltaR',
                'W_Pt', 'W_Phi', 'W_MT',
                'Top_Pt', 'Top_Phi', 'Top_MT'
            ]
            
            for feat in ['Z_Pt', 'Z_Eta', 'Z_Phi', 'Z_Mass', 'Z_DeltaR']:
                real_data = np.column_stack([real_data, real_z_kin[feat]])
                gen_data = np.column_stack([gen_data, gen_z_kin[feat]])
                features.append(feat)
            
            for feat in ['W_Pt', 'W_Phi', 'W_MT']:
                real_data = np.column_stack([real_data, real_w_kin[feat]])
                gen_data = np.column_stack([gen_data, gen_w_kin[feat]])
                features.append(feat)
            
            for feat in ['Top_Pt', 'Top_Phi', 'Top_MT']:
                real_data = np.column_stack([real_data, real_top_kin[feat]])
                gen_data = np.column_stack([gen_data, gen_top_kin[feat]])
                features.append(feat)
            
            print(f"Added {len(derived_features)} derived variables: {', '.join(derived_features)}")
        
        # Use original data order (no reordering)
        plot_feature_comparison(
            real_data, gen_data, features,
            self.selection, self.variables, output_path, bins, n_cols,
            real_weights=self.original_weights  # Pass weights for proper comparison
        )
        
    def plot_all(self, include_derived=True):
        """
        Generate all comparison plots.
        
        Parameters
        ----------
        include_derived : bool
            Whether to include derived Z kinematics variables
        """
        self.plot_feature_comparison(include_derived=include_derived)
        print("Plots saved successfully!")

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
from plotting import plot_correlation_diagnostics
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
        
        # Extract weight type from model name (e.g., "cHt5_linear_pos_nall_..." → "linear_pos")
        # Use explicit alternation to avoid capturing trailing suffixes like "_nall"
        import re
        weight_match = re.search(
            r'cHt5_(quadratic_(?:pos|neg)|linear_(?:pos|neg)|sm|quadratic|linear|full)',
            model_name
        )
        self.weight_type = weight_match.group(1) if weight_match else "unknown"
        logging.info(f"Detected weight type from model name: '{self.weight_type}'")
        
        # Load variables configuration
        with open(variables_json_path, 'r') as f:
            self.variables = json.load(f)
        self.original_feature_order = list(self.variables['colnames'].keys())
        
        # Load comparison data directly from the configured cache.
        # The current schema uses 15 physics features; some caches append 4 weight columns.
        original_data_path = data_dir
        if not os.path.exists(original_data_path):
            raise FileNotFoundError(f"Original data file not found: {original_data_path}")

        full_data_with_weights = np.load(original_data_path)
        logging.info(f"Loaded original ttZ data with weights shape: {full_data_with_weights.shape}")

        n_physics_features = len(self.variables['colnames'])
        full_data = full_data_with_weights[:, :n_physics_features]
        logging.info(f"Extracted physics features shape: {full_data.shape}")

        # Extract weights for this model's weight type if appended columns are present.
        weight_col_map = {
            "sm": n_physics_features,
            "linear": n_physics_features + 1,
            "linear_pos": n_physics_features + 1,
            "linear_neg": n_physics_features + 1,
            "quadratic": n_physics_features + 2,
            "quadratic_pos": n_physics_features + 2,
            "quadratic_neg": n_physics_features + 2,
            "full": n_physics_features + 3,
        }
        
        if self.weight_type in weight_col_map and full_data_with_weights.shape[1] > weight_col_map[self.weight_type]:
            col_idx = weight_col_map[self.weight_type]
            raw_weights = full_data_with_weights[:, col_idx]
            
            # For _pos/_neg types, filter to the appropriate subset and apply sign-flipping
            if self.weight_type.endswith('_pos'):
                valid_mask = raw_weights >= 0
                self.original_weights = raw_weights[valid_mask]
                full_data = full_data[valid_mask]
                component = self.weight_type.split('_')[0]  # "linear" or "quadratic"
                logging.info(f"Filtered to {valid_mask.sum()} events with {component} >= 0")
            elif self.weight_type.endswith('_neg'):
                valid_mask = raw_weights < 0
                self.original_weights = -raw_weights[valid_mask]  # flip sign to positive
                full_data = full_data[valid_mask]
                component = self.weight_type.split('_')[0]  # "linear" or "quadratic"
                logging.info(f"Filtered to {valid_mask.sum()} events with {component} < 0 (sign flipped to positive)")
            else:
                # SM or base types (linear, quadratic, full) — no filtering
                self.original_weights = raw_weights
            
            logging.info(f"Loaded weights from column {col_idx} for weight_type='{self.weight_type}'")
            logging.info(f"  Weight stats: min={self.original_weights.min():.6e}, max={self.original_weights.max():.6e}, mean={self.original_weights.mean():.6e}")
        else:
            self.original_weights = None
            logging.warning(
                f"Unknown or unavailable weight_type '{self.weight_type}' in data shape {full_data_with_weights.shape} "
                "— using unweighted comparison"
            )
        
        self.original_data = full_data
        logging.info(f"Using {full_data.shape[0]} events for comparison: shape {self.original_data.shape}")
        
        # Also load preprocessed data (for reference)
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Preprocessed data file not found: {data_dir}")
        
        self.preprocessed_data = np.load(data_dir)
        logging.info(f"Loaded preprocessed ttZ data shape: {self.preprocessed_data.shape}")
        
        # Initialize preprocessing
        self._setup_preprocessing()
        
        # Storage for generated data
        self.generated_data = None

    @staticmethod
    def _align_feature_columns(data, source_features, target_features, label="data"):
        """Return a view of data with columns reordered from source_features to target_features."""
        source_set = set(source_features)
        missing = [f for f in target_features if f not in source_set]
        if missing:
            raise KeyError(
                f"Cannot align {label}: missing features in source ordering: {missing}"
            )

        idx_map = [source_features.index(f) for f in target_features]
        return data[:, idx_map]
    
    def _setup_preprocessing(self):
        """Setup preprocessing pipeline and scalers."""
        # Load preprocessing config to match training setup
        config_path = os.path.join(project_root, "ml/custom/ttz/config/data_config.yaml")
        with open(config_path, 'r') as f:
            data_config = yaml.safe_load(f)
        preprocessing_config = data_config['data_config']['preprocessing']
        
        # Get feature names from current variables schema.
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
            original_data_path = self.data_dir
            if not os.path.exists(original_data_path):
                raise FileNotFoundError(f"Original data file not found: {original_data_path}")

            original_data = np.load(original_data_path)
            n_features = len(self.variables['colnames'])
            if original_data.shape[1] >= n_features:
                original_data = original_data[:, :n_features]
            logging.info(f"Loaded original ttZ data shape: {original_data.shape}")
            
            pre = Preprocessor(**preprocessing_config)
            _, self.selection, self.scalers = pre(original_data, self.selection)
        
        # Remove any appended weight columns from selection to match generated model outputs.
        self.selection = self.selection[~self.selection['feature'].str.startswith('cHt_weight')].reset_index(drop=True)
        logging.info(f"After removing weight: selection shape = {self.selection.shape}")
        n_cont = len(self.selection[self.selection['type'].isin(['cont', 'uni'])])
        n_disc = len(self.selection[self.selection['type'] == 'disc'])
        logging.info(f"  Features after weight removal: {n_cont} continuous/uni, {n_disc} discrete")
        
        # Setup rescaling handler
        self.rescale_handler = RescalingHandler(self.selection, self.scalers)
        
        # Use the feature order from selection dataframe
        self.selected_features = self.selection['feature'].tolist()
        
        # Keep weight handling determined in __init__ (do not override here).
        logging.info(f"Checking for weights in data: shape = {self.original_data.shape}")
        print(f"\n=== Weight Extraction Debug ===")
        print(f"Original data shape: {self.original_data.shape}")
        if self.original_weights is None:
            logging.info("Analyzer comparison mode: unweighted MC reference")
            print("No MC weights attached to comparison data")
        else:
            logging.info(
                "Analyzer comparison mode: weighted MC reference "
                f"(n={len(self.original_weights)}, mean={np.mean(self.original_weights):.6e})"
            )
            print(
                f"Using MC weights: n={len(self.original_weights)}, "
                f"min={np.min(self.original_weights):.6e}, "
                f"max={np.max(self.original_weights):.6e}"
            )
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
        
        real_data = self._align_feature_columns(
            self.original_data,
            self.original_feature_order,
            self.selected_features,
            label="real_data",
        )
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
            real_weights=self.original_weights,
            weight_type=self.weight_type
        )

    def plot_correlation_diagnostics(self, output_path=None, bins_1d=60, bins_2d=60):
        """Plot focused correlation diagnostics used to debug higher-order observables."""
        if self.generated_data is None:
            raise ValueError("No generated data available. Call generate_samples() first.")

        if output_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            figures_dir = os.path.join(script_dir, 'figures')
            os.makedirs(figures_dir, exist_ok=True)
            output_path = os.path.join(figures_dir, 'correlation_diagnostics.png')

        plot_correlation_diagnostics(
            self.original_data,
            self.generated_data,
            self.selected_features,
            output_path,
            real_weights=self.original_weights,
            bins_1d=bins_1d,
            bins_2d=bins_2d,
        )
        
    def plot_all(self, include_derived=True, include_correlation_diagnostics=True, figures_dir=None):
        """
        Generate all comparison plots.
        
        Parameters
        ----------
        include_derived : bool
            Whether to include derived Z kinematics variables
        include_correlation_diagnostics : bool
            Whether to generate DeltaEta/DeltaPhi correlation diagnostics
        figures_dir : str or Path, optional
            Directory to save figures. Defaults to sample_analyzer/figures/.
        """
        if figures_dir is not None:
            import pathlib
            figures_dir = str(figures_dir)
            os.makedirs(figures_dir, exist_ok=True)
            output_path = os.path.join(figures_dir, 'feature_comparison.png')
            corr_output_path = os.path.join(figures_dir, 'correlation_diagnostics.png')
        else:
            output_path = None  # plot_feature_comparison will use its own default
            corr_output_path = None
        self.plot_feature_comparison(output_path=output_path, include_derived=include_derived)
        if include_correlation_diagnostics:
            self.plot_correlation_diagnostics(output_path=corr_output_path)
        print("Plots saved successfully!")

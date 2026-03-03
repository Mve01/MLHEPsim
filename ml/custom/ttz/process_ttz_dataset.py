import uproot
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from ml.common.data_utils.processors import FeatureSelector, NpyProcessor
from ml.common.data_utils.utils import url_download
from ml.common.nn.gen_model_sampler import GenModelSampler

class ttzNpyProcessor(NpyProcessor):
    def __init__(
        self,
        data_dir,
        list_data_features,
        base_file_name="ttz",
        keep_ratio=1.0,
        shuffle=True,
        hold_mode=False,
        use_hold=False,
        hold_ratio=0.2,
        cut=None,
        load_weights=False,
        weight_type="full"
    ):
        """ttz dataset to .npy starting processor.

        Note
        ----
        Supports holdout mode, where data is split into two partitions. Partition 1 is used for training and partition 2
        is used for independent holdout evaluation.

        Parameters
        ----------
        keep_ratio : float, optional
            Keep only a fraction of the data, by default 1.0.
        shuffle : bool, optional
            Shuffle data loaded from starting file, by default True.
        hold_mode : bool, optional
            Holdout mode to use partition_1 of partition_2 of holdout data, by default False.
        use_hold : bool, optional
            If True use partition_1 else use partition_2, by default False.
        hold_ratio : float, optional
            Ratio of holdout data in partition_2, by default 0.2.
        """
        super().__init__(data_dir, base_file_name)
        self.file_name = None
        self.keep_ratio = keep_ratio
        self.shuffle = shuffle
        self.hold_mode, self.use_hold, self.hold_ratio = hold_mode, use_hold, 1 - hold_ratio
        self.load_weights = load_weights
        self.weight_type = weight_type
        self.list_data_features = list_data_features
        
        # Define features for processed data - must match the order in create_dataset()
        # Using cylindrical coordinates (Pt, Eta, Phi) - physics-motivated representation
        # Physics-motivated order: Z leptons (2×3=6) → W lepton (1×3=3) → BJet (1×4=4) → MET (2) = 15 features total
        # Charge features removed: not used in any physics calculation and trivially ±1
        processed_features = []
        # Z leptons first - from Z boson decay
        for i in [1, 2]:
            for var in ['Pt', 'Eta', 'Phi']:
                processed_features.append(f'Z_Lepton{i}_{var}')
        # W lepton second - from W boson decay
        for var in ['Pt', 'Eta', 'Phi']:
            processed_features.append(f'W_Lepton_{var}')
        # BJet third - from top quark decay
        for var in ['Pt', 'Eta', 'Phi', 'Mass']:
            processed_features.append(f'BJet_{var}')
        # MET last - global event properties
        processed_features.extend(['MET', 'MET_Phi'])
        
        # Create features dict with 15 physics features
        # Phi angles are type 'uni' (uniform in [-pi, pi]) - Gaussian rank scaled for training
        # Pt/Eta/Mass/MET are continuous
        self.features = {"colnames": {}}
        for feature in processed_features:
            if 'Phi' in feature:
                self.features["colnames"][feature] = "uni"
            else:
                self.features["colnames"][feature] = "cont"
        
        # Add cHt weight if loading weights
        if self.load_weights:
            self.features["colnames"]["cHt_weight"] = "weight"
        
        # Save updated features to variables.json (overwrite parent class load)
        import json
        import os
        variables_path = os.path.join(self.data_dir, 'variables.json')
        with open(variables_path, 'w') as f:
            json.dump(self.features, f, indent=4)
        logging.info(f"Updated {variables_path} with new feature order")

        if self.hold_mode:
            self.hold_npy_partition_1 = self.npy_file.replace(".npy", "_hold_partition_1.npy")
            self.hold_npy_partition_2 = self.npy_file.replace(".npy", "_hold_partition_2.npy")
    
    def _select_npy_file(self):
        if self.hold_mode and not self.use_hold:
            logging.info(f"Using holdout partition 1 from {self.hold_npy_partition_1}!")
            return self.hold_npy_partition_1, self.features
        elif self.hold_mode and self.use_hold:
            logging.info(f"Using holdout partition 2 from {self.hold_npy_partition_2}!")
            return self.hold_npy_partition_2, self.features
        else:
            logging.info(f"Using {self.npy_file}!")
            return self.npy_file, self.features

    def __call__(self, *args, **kwargs):
        dataset = self.get_dataset()
        
        if dataset is None:
            logging.info(f"{self.npy_file} already exists!")
            return self._select_npy_file()

        dataset = self.process_dataset(dataset)

        if self.hold_mode:
            hold_idx = int(len(dataset) * self.hold_ratio)

            hold_dataset = dataset[:hold_idx]
            dataset = dataset[hold_idx:]

            self.make_npy_file(dataset, self.hold_npy_partition_1)
            self.make_npy_file(hold_dataset, self.hold_npy_partition_2)
        else:
            dataset = self.process_dataset(dataset)
            self.make_npy_file(dataset, self.npy_file)

        return self._select_npy_file()

    def get_dataset(self):
        """Creates ttz dataset dataframe if not existing yet, otherwise loads existing .npy file.
        Parameters
        ----------
        data_dir : str, optional
            Path to ttZ data, by default "data/".

        Returns
        -------
        pd.DataFrame
            12 dim dataframe of all downloaded data.

        """
    
        # Check if .npy file already exists
        if Path(self.npy_file).is_file():
            logging.info(f"{self.npy_file} already exists, loading...")
            return None
            
        logging.info(f"Creating new dataset...")
        
        # Create .npy file from ROOT files and return dataframe
        self.create_dataset(list_data_features=self.list_data_features)
        
        try:
            data = np.load(self.npy_file, allow_pickle=True)
            return pd.DataFrame(data)
        except Exception as e:
            logging.warning(f"⚠️ Failed to load {self.npy_file}")
            return None


    def create_dataset(self, list_data_features):
        """Creates ttz dataset from ROOT files with pre-identified physics objects.
        
        The input file already has physics objects identified:
        - 2 leptons from Z decay (Z_Lepton1, Z_Lepton2)
        - 1 lepton from W decay (W_Lepton)
        - 1 b-jet (BJet)
        - MET (missing transverse energy)
        
        No event selection is performed - all events in the file are used.
        """

        logging.info("Loading ttz dataset from ROOT files with pre-identified physics objects!")

        # File path - new ATLAS processed file with physics objects already identified
        file_path_1 = "/project/atlas/users/kdevries/EventLoop/full_SR_ttZ_tree.root"

        # Required branches - load cylindrical coordinates directly from ROOT file
        branches_to_load = [
            'Z_Lepton1_Pt', 'Z_Lepton1_Eta', 'Z_Lepton1_Phi',
            'Z_Lepton2_Pt', 'Z_Lepton2_Eta', 'Z_Lepton2_Phi',
            'W_Lepton_Pt', 'W_Lepton_Eta', 'W_Lepton_Phi',
            'BJet_Pt', 'BJet_Eta', 'BJet_Phi', 'BJet_Mass',
            'MET', 'MET_phi'
        ]
        if self.load_weights:
            # For SMEFT weight decomposition, we need eventWeight and smeft_weights
            branches_to_load.extend(['eventWeight', 'smeft_weights'])

        # Load and concatenate data from all files
        all_filtered_data = []
        
        for file_path in [file_path_1]:
            with uproot.open(file_path) as file_ttz:
                tree_ttz = file_ttz["Events"]
                data = tree_ttz.arrays(branches_to_load, library="np")

            n_events = len(data['MET'])
            logging.info(f"Processing {n_events} events from {file_path}")

            # Build output array with vectorized numpy stacking (no Python for loop).
            # This avoids the enormous per-element Python object overhead of a list-of-lists
            # approach (~24 bytes/float vs 4 bytes for float32), saving several GB of RAM.
            # Order: Z_Lepton1 (3) → Z_Lepton2 (3) → W_Lepton (3) → BJet (4) → MET (2) = 15 features
            columns = [
                data['Z_Lepton1_Pt'],  data['Z_Lepton1_Eta'], data['Z_Lepton1_Phi'],
                data['Z_Lepton2_Pt'],  data['Z_Lepton2_Eta'], data['Z_Lepton2_Phi'],
                data['W_Lepton_Pt'],   data['W_Lepton_Eta'],  data['W_Lepton_Phi'],
                data['BJet_Pt'],       data['BJet_Eta'],       data['BJet_Phi'],      data['BJet_Mass'],
                data['MET'],           data['MET_phi'],
            ]
            if self.load_weights:
                # Compute appropriate SMEFT weight based on weight_type
                # Weight indices for ttz (from ml/data/ttz/cHt_weight_indices.txt):
                #   122: cHt_m5p0 (cHt = -5.0)
                #   124: cHt_p5p0 (cHt = +5.0)
                # Note: Decomposition is hardcoded for cHt = ±5.0
                
                if self.weight_type == "full":
                    # Use full SMEFT weight at cHt = +5.0
                    weight = np.stack(data['smeft_weights'])[:, 124]
                    logging.info(f"  Using full SMEFT weight at cHt=5.0")
                        
                elif self.weight_type == "sm":
                    # Use SM weight only (from eventWeight branch)
                    weight = data['eventWeight']
                    logging.info(f"  Using SM weight only (eventWeight)")
                    
                elif self.weight_type == "linear":
                    # Extract linear term: w_lin = (w_plus - w_minus) / (2*c)
                    # With c=5.0: w_lin = (w_plus - w_minus) / 10.0
                    w_plus = np.stack(data['smeft_weights'])[:, 124]  # cHt = +5.0
                    w_minus = np.stack(data['smeft_weights'])[:, 122]  # cHt = -5.0
                    weight = (w_plus - w_minus) / 10.0
                    logging.info(f"  Using linear SMEFT term (cHt=5.0 decomposition)")
                    
                elif self.weight_type == "quadratic":
                    # Extract quadratic term: w_quad = (w_plus + w_minus - 2*w_sm) / (2*c^2)
                    # With c=5.0: w_quad = (w_plus + w_minus - 2*w_sm) / 50.0
                    w_plus = np.stack(data['smeft_weights'])[:, 124]  # cHt = +5.0
                    w_minus = np.stack(data['smeft_weights'])[:, 122]  # cHt = -5.0
                    w_sm = data['eventWeight']
                    weight = (w_plus + w_minus - 2.0 * w_sm) / 50.0
                    logging.info(f"  Using quadratic SMEFT term (cHt=5.0 decomposition)")
                    
                else:
                    raise ValueError(f"Invalid weight_type: {self.weight_type}. "
                                   f"Must be 'full', 'sm', 'linear', or 'quadratic'")
                
                columns.append(weight)

            self.x = np.column_stack(columns).astype(np.float32)
            logging.info(f"  Loaded {len(self.x)} events (all events in file)")

            if self.load_weights:
                logging.info(f"Kept {len(self.x)} events with cylindrical coordinates and {self.weight_type} weights")
            else:
                logging.info(f"Kept {len(self.x)} events with cylindrical coordinates")
            all_filtered_data.append(self.x) 

        dataset = np.concatenate(all_filtered_data, axis=0)
        logging.info(f"Final ttz dataset shape with cylindrical coordinates: {dataset.shape}")
        if self.load_weights:
            weight_desc = {
                "full": "full SMEFT weight at cHt=5.0",
                "sm": "SM weight (eventWeight)",
                "linear": "linear SMEFT term (cHt=5.0)",
                "quadratic": "quadratic SMEFT term (cHt=5.0)"
            }[self.weight_type]
            logging.info(f"Shape: {dataset.shape[0]} events, {dataset.shape[1]-1} features (15 cylindrical) + 1 weight column ({weight_desc})")
        
        # Shuffle the dataset before saving to ensure random sampling
        # Use fixed seed for reproducibility across data regenerations
        rng = np.random.RandomState(42)
        rng.shuffle(dataset)
        logging.info("Shuffled dataset before saving (seed=42 for reproducibility)")

        np.save(self.npy_file, dataset)
        logging.info(f"saved {self.npy_file} of shape {dataset.shape}!")


    def process_dataset(self, dataset):
        logging.info("Processing dataset!")

        if self.shuffle:
            logging.info("Shuffling data!")
            dataset = dataset.sample(frac=1).reset_index(drop=True)

        if self.keep_ratio < 1.0:
            logging.info(f"Keeping only {self.keep_ratio:.2f} of data!")
            dataset = dataset[: int(len(dataset) * self.keep_ratio)]

        return dataset

    def make_npy_file(self, dataset, npy_file=None):
        if npy_file is None:
            npy_file = self.npy_file
        
        # Convert DataFrame to numpy array if needed
        if hasattr(dataset, 'values'):
            dataset = dataset.values
        
        np.save(npy_file, dataset)
        logging.info(f"Saved {npy_file} with shape {dataset.shape}")

    def download(self):
        pass


class ttzFeatureSelector(FeatureSelector):
    def __init__(self, file_path, n_data=None, **kwargs):
        super().__init__(file_path, **kwargs)
        self.n_data = n_data

    def load_data(self):
        logging.info(f"Loading data from {self.file_path}!")
        data = np.load(self.file_path)

        if self.n_data is not None:
            data = data[: self.n_data]
            logging.info(f"Using {self.n_data} data points!")

        return data

    def select_features(self, data):
        return super().select_features(data)
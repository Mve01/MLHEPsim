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
        load_weights=False
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
        self.list_data_features = list_data_features
        
        # Define features for processed data - must match the order in create_dataset() line 340
        # Physics-motivated order: Jets (3×5=15) → Leptons (3×4=12) → MET (2) = 29 features total
        # This groups correlated variables together for better autoregressive flow learning
        processed_features = []
        # Jets first - group all properties per jet (strongly correlated)
        for i in range(1, 4):
            for var in ['Pt', 'Eta', 'Phi', 'Mass', 'BTag']:
                processed_features.append(f'Jet{i}_{var}')
        # Leptons second - group all properties per lepton (strongly correlated)
        for i in range(1, 4):
            for var in ['Pt', 'Eta', 'Phi', 'Charge']:
                processed_features.append(f'Lepton{i}_{var}')
        # MET last - global event properties
        processed_features.extend(['MET', 'MET_Phi'])
        
        # Create features dict with 29 physics features
        # Mark Phi angles as "uni" (uniform/periodic) - they should NOT be Gaussian rank scaled
        self.features = {"colnames": {}}
        for feature in processed_features:
            if 'Phi' in feature:
                self.features["colnames"][feature] = "uni"
            elif 'Charge' in feature:
                self.features["colnames"][feature] = "disc"
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
        """Creates ttz dataset from ROOT files with physics-based particle selection.
        
        Selection follows ttZ event topology:
        - 2 leptons from Z decay (OSSF pair closest to mZ)
        - 1 lepton from W decay (remaining lepton)
        - 1 b-jet (highest BTag score)
        - 2 light jets (forward jet + radiative jet)
        - MET (from neutrino)
        """

        logging.info("Loading ttz dataset from ROOT files with physics selection!")

        # File path - new ATLAS processed file
        file_path_1 = "/project/atlas/users/kdevries/EventLoop/ATLASSMEFT_ttZ_tree.root"

        # Required branches for physics selection
        branches_to_load = [
            'Electron_Pt', 'Electron_Eta', 'Electron_Phi', 'Electron_E', 'Electron_Charge',
            'Muon_Pt', 'Muon_Eta', 'Muon_Phi', 'Muon_E', 'Muon_Charge',
            'Jet_Pt', 'Jet_Eta', 'Jet_Phi', 'Jet_Mass', 'Jet_E', 'Jet_BTag',
            'MET', 'MET_phi'
        ]
        if self.load_weights:
            branches_to_load.append('smeft_weights')

        # Load and concatenate data from all files
        all_filtered_data = []
        
        for file_path in [file_path_1]:
            with uproot.open(file_path) as file_ttz:
                tree_ttz = file_ttz["Events"]
                data = tree_ttz.arrays(branches_to_load, library="np")
    
            n_events = len(data['MET'])
            logging.info(f"Processing {n_events} events from {file_path}")
            
            # Prepare output array: 23 features + optional weight column
            event_data = []
            n_valid = 0
            n_wrong_lepton_count = 0
            n_no_ossf = 0
            n_no_jets = 0
            
            for i in range(n_events):
                # === STEP 1: Build lepton list with charge and flavor ===
                leptons = []
                
                # Electrons (flavor = 11)
                n_el = len(data['Electron_Pt'][i])
                for j in range(n_el):
                    leptons.append({
                        'pt': data['Electron_Pt'][i][j],
                        'eta': data['Electron_Eta'][i][j],
                        'phi': data['Electron_Phi'][i][j],
                        'E': data['Electron_E'][i][j],
                        'charge': data['Electron_Charge'][i][j],
                        'flavor': 11
                    })
                
                # Muons (flavor = 13)
                n_mu = len(data['Muon_Pt'][i])
                for j in range(n_mu):
                    leptons.append({
                        'pt': data['Muon_Pt'][i][j],
                        'eta': data['Muon_Eta'][i][j],
                        'phi': data['Muon_Phi'][i][j],
                        'E': data['Muon_E'][i][j],
                        'charge': data['Muon_Charge'][i][j],
                        'flavor': 13
                    })
                
                # Require exactly 3 leptons
                if len(leptons) != 3:
                    n_wrong_lepton_count += 1
                    continue
                
                # === STEP 2: Find Z boson (OSSF pair closest to mZ = 91.1876 GeV) ===
                mZ = 91.1876
                best_z_pair = None
                best_mass_diff = float('inf')
                
                for idx1 in range(len(leptons)):
                    for idx2 in range(idx1 + 1, len(leptons)):
                        lep1 = leptons[idx1]
                        lep2 = leptons[idx2]
                        
                        # Check OSSF: same flavor, opposite sign
                        if lep1['flavor'] != lep2['flavor']:
                            continue
                        if lep1['charge'] * lep2['charge'] >= 0:
                            continue
                        
                        # Calculate invariant mass
                        px1 = lep1['pt'] * np.cos(lep1['phi'])
                        py1 = lep1['pt'] * np.sin(lep1['phi'])
                        pz1 = lep1['pt'] * np.sinh(lep1['eta'])
                        px2 = lep2['pt'] * np.cos(lep2['phi'])
                        py2 = lep2['pt'] * np.sin(lep2['phi'])
                        pz2 = lep2['pt'] * np.sinh(lep2['eta'])
                        
                        E_tot = lep1['E'] + lep2['E']
                        px_tot = px1 + px2
                        py_tot = py1 + py2
                        pz_tot = pz1 + pz2
                        mass = np.sqrt(E_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2)
                        
                        mass_diff = abs(mass - mZ)
                        if mass_diff < best_mass_diff:
                            best_mass_diff = mass_diff
                            best_z_pair = (idx1, idx2)
                
                # Require valid Z candidate within 10 GeV window
                if best_z_pair is None or best_mass_diff > 10.0:
                    n_no_ossf += 1
                    continue
                
                # === STEP 3: Identify leptons ===
                z_idx1, z_idx2 = best_z_pair
                # Order Z leptons by pT
                if leptons[z_idx1]['pt'] > leptons[z_idx2]['pt']:
                    z_lep1, z_lep2 = leptons[z_idx1], leptons[z_idx2]
                else:
                    z_lep1, z_lep2 = leptons[z_idx2], leptons[z_idx1]
                
                # Third lepton (from W)
                w_lep_idx = [idx for idx in range(3) if idx not in best_z_pair][0]
                w_lep = leptons[w_lep_idx]
                
                # === STEP 4: Select jets ===
                n_jets = len(data['Jet_Pt'][i])
                if n_jets < 3:
                    n_no_jets += 1
                    continue
                
                jets = []
                for j in range(n_jets):
                    jets.append({
                        'pt': data['Jet_Pt'][i][j],
                        'eta': data['Jet_Eta'][i][j],
                        'phi': data['Jet_Phi'][i][j],
                        'mass': data['Jet_Mass'][i][j],
                        'E': data['Jet_E'][i][j],
                        'btag': data['Jet_BTag'][i][j],
                        'idx': j
                    })
                
                # B-jet: highest BTag score
                b_jet = max(jets, key=lambda j: j['btag'])
                non_b_jets = [j for j in jets if j['idx'] != b_jet['idx']]
                
                if len(non_b_jets) < 2:
                    n_no_jets += 1
                    continue
                
                # Forward jet: maximizes invariant mass with b-jet
                best_forward_jet = None
                max_mass = -1
                for jet in non_b_jets:
                    # Calculate M(b-jet, jet)
                    px_b = b_jet['pt'] * np.cos(b_jet['phi'])
                    py_b = b_jet['pt'] * np.sin(b_jet['phi'])
                    pz_b = b_jet['pt'] * np.sinh(b_jet['eta'])
                    px_j = jet['pt'] * np.cos(jet['phi'])
                    py_j = jet['pt'] * np.sin(jet['phi'])
                    pz_j = jet['pt'] * np.sinh(jet['eta'])
                    
                    E_tot = b_jet['E'] + jet['E']
                    px_tot = px_b + px_j
                    py_tot = py_b + py_j
                    pz_tot = pz_b + pz_j
                    mass = np.sqrt(max(0, E_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2))
                    
                    if mass > max_mass:
                        max_mass = mass
                        best_forward_jet = jet
                
                forward_jet = best_forward_jet
                
                # Radiative jet: highest pT among remaining jets
                remaining_jets = [j for j in non_b_jets if j['idx'] != forward_jet['idx']]
                if len(remaining_jets) == 0:
                    n_no_jets += 1
                    continue
                radiative_jet = max(remaining_jets, key=lambda j: j['pt'])
                
                # === STEP 5: Assemble features in physics-motivated order ===
                # Order: Jets → Leptons → MET (groups correlated variables)
                event_row = [
                    # B-jet (jet1) - highest BTag score
                    b_jet['pt'], b_jet['eta'], b_jet['phi'], b_jet['mass'], b_jet['btag'],
                    # Forward jet (jet2) - most forward eta
                    forward_jet['pt'], forward_jet['eta'], forward_jet['phi'], forward_jet['mass'], forward_jet['btag'],
                    # Radiative jet (jet3) - highest pT of remaining
                    radiative_jet['pt'], radiative_jet['eta'], radiative_jet['phi'], radiative_jet['mass'], radiative_jet['btag'],
                    # Z leptons (ordered by pT)
                    z_lep1['pt'], z_lep1['eta'], z_lep1['phi'], z_lep1['charge'],
                    z_lep2['pt'], z_lep2['eta'], z_lep2['phi'], z_lep2['charge'],
                    # W lepton
                    w_lep['pt'], w_lep['eta'], w_lep['phi'], w_lep['charge'],
                    # MET - global event property
                    data['MET'][i], data['MET_phi'][i]
                ]
                
                # Add weight if requested
                if self.load_weights:
                    event_row.append(data['smeft_weights'][i][124])  # cHt=5.0 weight
                
                event_data.append(event_row)
                n_valid += 1
            
            logging.info(f"  Valid events: {n_valid}/{n_events}")
            logging.info(f"  Rejected - wrong lepton count (!= 3): {n_wrong_lepton_count}")
            logging.info(f"  Rejected - no OSSF pair: {n_no_ossf}")
            logging.info(f"  Rejected - insufficient jets: {n_no_jets}")
            
            if len(event_data) > 0:
                self.x = np.array(event_data, dtype=np.float32)
                if self.load_weights:
                    logging.info(f"Kept {len(self.x)} events with physics selection and cHt=5.0 weights")
                else:
                    logging.info(f"Kept {len(self.x)} events with physics selection")
                all_filtered_data.append(self.x)
            else:
                logging.warning(f"No valid events found in {file_path}") 

        dataset = np.concatenate(all_filtered_data, axis=0)
        logging.info(f"Final ttz dataset shape after physics selection: {dataset.shape}")
        if self.load_weights:
            logging.info(f"Shape: {dataset.shape[0]} events, {dataset.shape[1]-1} features + 1 weight column")
        
        # Shuffle the dataset before saving to ensure random sampling
        np.random.shuffle(dataset)
        logging.info("Shuffled dataset before saving")

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
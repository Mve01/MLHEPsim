import numpy as np
import logging
import torch

from ml.common.data_utils.data_modules import DataModule, SupervisedDataset


class ttzDataset(SupervisedDataset):
    def __init__(self, data, selection, weights=None):
        super().__init__(data, selection)
        self.weights = weights
        if weights is not None:
            self.weights = torch.from_numpy(weights) if isinstance(weights, np.ndarray) else weights
    
    def __getitem__(self, idx):
        if self.weights is not None:
            return self.X[idx], self.y[idx], self.weights[idx]
        return self.X[idx], self.y[idx]


class ttzDataModule(DataModule):
    def __init__(self, processor, dataset=ttzDataset, use_weights=False, **kwargs):
        super().__init__(processor, dataset, **kwargs)
        self.use_weights = use_weights
        self.weights = None

    def setup(self, stage=None):
        data, self.selection, self.scalers = self.processor()
        data = np.float32(data)
        
        # Extract weights from last column if requested
        if self.use_weights:
            if data.shape[1] == 30:  # 29 features + 1 weight column
                self.weights = data[:, -1]  # Last column is the weight
                data = data[:, :-1]  # Remove weight column from features
                # Remove cHt_weight from selection since it's no longer in data
                self.selection = self.selection[self.selection["feature"] != "cHt_weight"].reset_index(drop=True)
                
                # Normalize weights to have mean=1.0 to preserve loss scale
                weight_mean = np.mean(self.weights)
                self.weights = self.weights / weight_mean
                logging.info(f"Extracted cHt=5.0 weights from data. Features shape: {data.shape}, Weights shape: {self.weights.shape}")
                logging.info(f"Normalized cHt weights: original mean={weight_mean:.6e}, new mean={np.mean(self.weights):.6f}, min={np.min(self.weights):.6f}, max={np.max(self.weights):.6f}")
            else:
                logging.warning(f"use_weights=True but data shape is {data.shape}, expected 30 columns (29 features + 1 weight). Training without weights.")
                self.use_weights = False

        self._get_splits(len(data))

        if stage == "fit" or stage is None:
            train_weights = self.weights[self.train_idx] if self.use_weights else None
            val_weights = self.weights[self.val_idx] if self.use_weights else None
            self.train = self.dataset(data[self.train_idx], self.selection, train_weights)
            self.val = self.dataset(data[self.val_idx], self.selection, val_weights)

        if stage == "test":
            test_weights = self.weights[self.test_idx] if self.use_weights else None
            self.test = self.dataset(data[self.test_idx], self.selection, test_weights)

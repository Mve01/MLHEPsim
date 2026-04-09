import numpy as np
import logging
import torch
from torch.utils.data import DataLoader, Sampler

from ml.common.data_utils.data_modules import DataModule, SupervisedDataset

# Weight feature names in the processed selection table.
# The _pos and _neg variants share the same base feature and apply sign-based
# event filtering in setup(): _pos keeps w>=0, _neg keeps w<0 and flips sign so
# the flow always sees positive training weights.
WEIGHT_FEATURES = {
    "sm": "cHt_weight_sm",
    "linear": "cHt_weight_linear",
    "quadratic": "cHt_weight_quadratic",
    "full": "cHt_weight_full",
    "linear_pos": "cHt_weight_linear",
    "linear_neg": "cHt_weight_linear",
    "quadratic_pos": "cHt_weight_quadratic",
    "quadratic_neg": "cHt_weight_quadratic",
}


class WeightStratifiedSampler(Sampler):
    """Sample batch indices so each batch draws roughly equally from weight-magnitude
    quantile bins.

    Motivation: SMEFT quadratic weights have a very long tail (max ~460x mean). A
    standard random shuffle risks forming batches dominated by a handful of extreme-weight
    events, causing sudden large gradient steps that destabilise training.

    This sampler divides events into ``n_bins`` equal-count bins by |weight|, then
    constructs each batch by taking ``batch_size // n_bins`` events from each bin (with
    within-bin shuffling each epoch). The weight values themselves are NOT modified —
    relative differences are fully preserved, only the batch composition is controlled.
    """

    def __init__(self, weights: np.ndarray, batch_size: int, n_bins: int = 4):
        abs_w = np.abs(np.asarray(weights, dtype=np.float64))
        edges = np.percentile(abs_w, np.linspace(0, 100, n_bins + 1))
        self.bins = []
        for i in range(n_bins):
            if i == n_bins - 1:
                mask = abs_w >= edges[i]
            else:
                mask = (abs_w >= edges[i]) & (abs_w < edges[i + 1])
            idx = np.where(mask)[0]
            if len(idx) > 0:
                self.bins.append(idx)
        self.n_bins = len(self.bins)
        self.batch_size = batch_size
        self.n_per_bin = max(1, batch_size // self.n_bins)
        self.n_samples = len(abs_w)

    def __iter__(self):
        shuffled = [np.random.permutation(b) for b in self.bins]
        pos = [0] * self.n_bins
        indices = []
        while len(indices) + self.batch_size <= self.n_samples:
            batch = []
            for k in range(self.n_bins):
                p = pos[k]
                if p + self.n_per_bin > len(shuffled[k]):
                    shuffled[k] = np.random.permutation(self.bins[k])
                    pos[k] = 0
                    p = 0
                batch.extend(shuffled[k][p:p + self.n_per_bin].tolist())
                pos[k] += self.n_per_bin
            np.random.shuffle(batch)
            indices.extend(batch)
        return iter(indices)

    def __len__(self):
        return (self.n_samples // self.batch_size) * self.batch_size


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
    def __init__(self, processor, dataset=ttzDataset, use_weights=False, weight_type="unknown", **kwargs):
        super().__init__(processor, dataset, **kwargs)
        self.use_weights = use_weights
        self.weight_type = weight_type
        self.weights = None

    def setup(self, stage=None):
        # Idempotency guard: Lightning (and our early setup call in main_flows.py)
        # may invoke setup() more than once.  The processor is expensive to re-run,
        # and re-creating datasets with different random splits is incorrect.
        if getattr(self, '_setup_done', False):
            return
        self._setup_done = True

        data, self.selection, self.scalers = self.processor()
        data = np.float32(data)
        
        # Extract weights from the multi-weight file if requested.
        if self.use_weights:
            if self.weight_type not in WEIGHT_FEATURES:
                raise ValueError(f"weight_type='{self.weight_type}' not in {list(WEIGHT_FEATURES)}")

            feature_names = self.selection["feature"].astype(str).tolist()
            requested_weight_feature = WEIGHT_FEATURES[self.weight_type]
            if requested_weight_feature not in feature_names:
                raise ValueError(
                    f"Requested weight feature '{requested_weight_feature}' for weight_type '{self.weight_type}' "
                    "not found in processor selection."
                )

            weight_feature_set = {
                "cHt_weight_sm",
                "cHt_weight_linear",
                "cHt_weight_quadratic",
                "cHt_weight_full",
            }
            col = feature_names.index(requested_weight_feature)
            self.weights = data[:, col]

            # Keep only physics features (drop all appended weight columns).
            feature_cols = [i for i, name in enumerate(feature_names) if name not in weight_feature_set]
            data = data[:, feature_cols]
            self.selection = self.selection.iloc[feature_cols].reset_index(drop=True)

            # For _pos/_neg split types, filter by sign BEFORE normalisation so that
            # mean(|w|) is computed only over the kept events.
            n_total_before = len(data)
            if self.weight_type.endswith('_pos'):
                valid_mask = self.weights >= 0
                n_kept = int(valid_mask.sum())
                n_dropped = n_total_before - n_kept
                data = data[valid_mask]
                self.weights = self.weights[valid_mask]
                logging.info(
                    f"  '{self.weight_type}': kept {n_kept} positive-weight events, "
                    f"dropped {n_dropped} ({100*n_dropped/n_total_before:.2f}%) negative-weight events."
                )
            elif self.weight_type.endswith('_neg'):
                valid_mask = self.weights < 0
                n_kept = int(valid_mask.sum())
                n_dropped = n_total_before - n_kept
                data = data[valid_mask]
                self.weights = -self.weights[valid_mask]  # flip sign -> always positive for training
                logging.info(
                    f"  '{self.weight_type}': kept {n_kept} negative-weight events (sign flipped to positive), "
                    f"dropped {n_dropped} ({100*n_dropped/n_total_before:.2f}%) non-negative events."
                )

            # Global normalisation to unit mean — no per-batch scaling.
            # With stratified sampling, batch composition is controlled and
            # the flow sees true weight magnitudes.
            weight_mean = float(np.mean(self.weights))
            self.weights = self.weights / weight_mean
            logging.info(
                f"Extracted weight_type='{self.weight_type}' weights (col {col}) "
                f"from data. Features shape: {data.shape}, Weights shape: {self.weights.shape}"
            )
            n_neg = int((self.weights < 0).sum())
            p99 = float(np.percentile(self.weights, 99))
            p999 = float(np.percentile(self.weights, 99.9))
            logging.info(
                f"  '{self.weight_type}' weights normalised by mean(w)={weight_mean:.6e}: "
                f"mean={np.mean(self.weights):.6f}, min={np.min(self.weights):.6f}, "
                f"p99={p99:.4f}, p99.9={p999:.4f}, max={np.max(self.weights):.6f}, "
                f"n_neg={n_neg} ({100*n_neg/len(self.weights):.2f}%)"
            )

            # For base signed types (linear, quadratic), drop remaining negatives for
            # backward compatibility.  The proper solution for signed weights is to use
            # the _pos / _neg split types above.
            if n_neg > 0 and not self.weight_type.endswith(('_pos', '_neg')):
                valid_mask = self.weights >= 0
                n_dropped = int((~valid_mask).sum())
                n_before_drop = len(self.weights)
                data = data[valid_mask]
                self.weights = self.weights[valid_mask]
                logging.warning(
                    f"  Dropped {n_dropped} ({100*n_dropped/n_before_drop:.2f}%) events with "
                    f"negative '{self.weight_type}' weights. Remaining: {len(data)} events. "
                    f"Consider using '{self.weight_type}_pos' / '{self.weight_type}_neg' instead."
                )
                if len(self.weights) > 0:
                    self.weights = self.weights / np.mean(self.weights)

        self._get_splits(len(data))

        if stage == "fit" or stage is None:
            train_weights = self.weights[self.train_idx] if self.use_weights else None
            val_weights = self.weights[self.val_idx] if self.use_weights else None
            self.train = self.dataset(data[self.train_idx], self.selection, train_weights)
            self.val = self.dataset(data[self.val_idx], self.selection, val_weights)

        if stage == "test":
            test_weights = self.weights[self.test_idx] if self.use_weights else None
            self.test = self.dataset(data[self.test_idx], self.selection, test_weights)

    def train_dataloader(self):
        if self.use_weights and self.train.weights is not None:
            batch_size = self.dataloader_kwargs.get('batch_size', 1024)
            n_train = len(self.train.weights)
            
            # If training set is too small for stratified sampling, use simple random sampler
            if n_train < batch_size * 2:  # Need at least 2x batch size for meaningful stratification
                batch_size = max(128, n_train // 4)  # reduce batch size aggressively
                if not getattr(self, '_train_small_logged', False):
                    logging.info(f"Training set too small ({n_train} events) for stratified sampling; using batch_size={batch_size}")
                    self._train_small_logged = True
                kw = {k: v for k, v in self.dataloader_kwargs.items() if k != 'batch_size'}
                return DataLoader(self.train, batch_size=batch_size, shuffle=True, **kw)
            
            sampler = WeightStratifiedSampler(
                self.train.weights.numpy(), batch_size=batch_size
            )
            if not getattr(self, '_train_sampler_logged', False):
                logging.info(
                    f"WeightStratifiedSampler (train): {sampler.n_bins} bins, "
                    f"{sampler.n_per_bin} events/bin/batch, batch_size={batch_size}"
                )
                self._train_sampler_logged = True
            # Remove batch_size from kwargs so it isn't passed twice; shuffle is
            # handled by the sampler itself so we must not pass shuffle=True.
            kw = {k: v for k, v in self.dataloader_kwargs.items() if k != 'batch_size'}
            return DataLoader(self.train, batch_size=batch_size, sampler=sampler, **kw)
        return DataLoader(self.train, shuffle=True, **self.dataloader_kwargs)

    def val_dataloader(self):
        """Validation uses stratified sampler too to prevent batch composition explosions."""
        if self.use_weights and self.val.weights is not None:
            batch_size = self.dataloader_kwargs.get('batch_size', 1024)
            n_val = len(self.val.weights)
            
            # If validation set is too small for stratified sampling, use simple random sampler
            # Stratified sampler requires n_samples >= batch_size to form complete batches
            if n_val < batch_size * 2:  # Need at least 2x batch size for meaningful stratification
                batch_size = max(128, n_val // 4)  # reduce batch size aggressively
                if not getattr(self, '_val_small_logged', False):
                    logging.info(f"Validation set too small ({n_val} events) for stratified sampling; using batch_size={batch_size}")
                    self._val_small_logged = True
                kw = {k: v for k, v in self.dataloader_kwargs.items() if k != 'batch_size'}
                return DataLoader(self.val, batch_size=batch_size, shuffle=False, **kw)
            
            sampler = WeightStratifiedSampler(
                self.val.weights.numpy(), batch_size=batch_size
            )
            if not getattr(self, '_val_sampler_logged', False):
                logging.info(
                    f"WeightStratifiedSampler (val): {sampler.n_bins} bins, "
                    f"{sampler.n_per_bin} events/bin/batch, batch_size={batch_size}"
                )
                self._val_sampler_logged = True
            kw = {k: v for k, v in self.dataloader_kwargs.items() if k != 'batch_size'}
            return DataLoader(self.val, batch_size=batch_size, sampler=sampler, **kw)
        return DataLoader(self.val, shuffle=False, **self.dataloader_kwargs)

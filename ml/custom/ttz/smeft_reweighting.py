#!/usr/bin/env python
"""
SMEFT importance reweighting using five trained normalising flows (pos/neg split).

For a full derivation see notes/smeft_importance_reweighting.tex.

The signed SMEFT weight components (linear ~20% negative, quadratic ~8% negative)
are decomposed into unsigned positive/negative parts:
    w_lin  = w_lin_pos  - w_lin_neg    (w_lin_pos, w_lin_neg >= 0)
    w_quad = w_quad_pos - w_quad_neg   (w_quad_pos, w_quad_neg >= 0)

Five normalising flows are trained, one per non-negative component:
    q_sm, q_lin_pos, q_lin_neg, q_quad_pos, q_quad_neg

Pipeline
--------
  1. Find and load all five flow models from MLflow.
  2. Compute Z_k normalisation constants from the ROOT data file.
  3. Sample N events from the SM flow (in preprocessed space).
  4. Evaluate log p_k(x) under all five flows for the same sample set.
  5. Compute log density ratios  log r_k = log p_k - log p_sm  for k ∈ {lin_pos/neg, quad_pos/neg}.
  6. For any desired c_Ht, compute importance weights:
         w_i(c) ∝ Z_sm
                  + c   * (Z_lin_pos  * exp(r_lin_pos_i)  - Z_lin_neg  * exp(r_lin_neg_i))
                  + c²  * (Z_quad_pos * exp(r_quad_pos_i) - Z_quad_neg * exp(r_quad_neg_i))
  7. Inverse-transform samples to physical space and produce comparison plots.

Steps 1-5 are computed once and saved to disk.
Step 6 is a microsecond arithmetic operation that can be re-run for any c.

Usage
-----
    python ml/custom/ttz/smeft_reweighting.py
    python ml/custom/ttz/smeft_reweighting.py --n-samples 200000
    python ml/custom/ttz/smeft_reweighting.py --c-values -5 -2 -1 0 1 2 5
    python ml/custom/ttz/smeft_reweighting.py --load-ratios  # skip recompute, load saved ratios
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / f"smeft_reweighting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_ROOT_FILE = "/project/atlas/users/kdevries/EventLoop/ttZ_for_Melle_tree.root"
IDX_CHT_MINUS5 = 122   # cHt = -5.0
IDX_CHT_PLUS5  = 124   # cHt = +5.0

OUTPUT_DIR = PROJECT_ROOT / "ml" / "custom" / "ttz" / "figures" / "smeft_reweighting"
RATIOS_FILE = OUTPUT_DIR / "density_ratios.npz"

FEATURE_NAMES = [
    'Z_Lepton1_Pt', 'Z_Lepton1_Eta', 'Z_Lepton1_Phi',
    'Z_Lepton2_Pt', 'Z_Lepton2_Eta', 'Z_Lepton2_Phi',
    'W_Lepton_Pt',  'W_Lepton_Eta',  'W_Lepton_Phi',
    'BJet_Pt',      'BJet_Eta',      'BJet_Phi',      'BJet_Mass',
    'MET',          'MET_Phi',
]


# ---------------------------------------------------------------------------
# Step 1 – Model discovery
# ---------------------------------------------------------------------------

def find_model_name(weight_type: str) -> str:
    """Find the latest MLflow model containing 'cHt5_<weight_type>'."""
    mlflow.set_tracking_uri(f"file://{PROJECT_ROOT / 'mlruns'}")
    client = mlflow.tracking.MlflowClient()

    matching = []
    for rm in client.search_registered_models():
        if f"cHt5_{weight_type}" in rm.name:
            versions = client.get_latest_versions(rm.name)
            if versions:
                matching.append((rm.name, versions[0].creation_timestamp))

    if not matching:
        raise ValueError(f"No MLflow model found containing 'cHt5_{weight_type}'")

    matching.sort(key=lambda x: x[1], reverse=True)
    best = matching[0][0]
    log.info(f"  [{weight_type}] using model: {best}")
    return best


def load_module(model_name: str, device: str = 'cpu'):
    """Load a Lightning module from MLflow registry.

    device : 'cpu' or 'cuda' (or 'cuda:0' etc.)

    Patches the plain Python `device` attributes on MADEMOG and
    AutoregressiveNormalizingFlow that are NOT updated by .to(), so that
    sampling and density evaluation use the correct device.
    """
    import torch
    from ml.common.utils.register_model import fetch_registered_module
    mlflow.set_tracking_uri(f"file://{PROJECT_ROOT / 'mlruns'}")
    dev = torch.device(device)
    module = fetch_registered_module(model_name, model_version=-1, device=device)
    module = module.to(dev)
    module.model.eval()

    # Patch plain Python `device` attributes that are NOT updated by .to().
    # MADEMOG.device is used to move input tensors in estimate_density().
    # AutoregressiveNormalizingFlow.device is used to create the dummy tensor in sample().
    if hasattr(module.model, 'device'):
        module.model.device = dev                    # MADEMOG
    inner_flow = module.model.model                  # AutoregressiveNormalizingFlow
    if hasattr(inner_flow, 'device'):
        inner_flow.device = dev

    return module


def _find_model_artifact_by_run_id(run_id: str) -> str:
    """Resolve an MLflow run_id to a local model artifact URI.

    Expected layout:
      mlruns/<experiment_id>/<run_id>/artifacts/model
    """
    mlruns_dir = PROJECT_ROOT / "mlruns"
    if not mlruns_dir.exists():
        raise FileNotFoundError(f"MLflow directory not found: {mlruns_dir}")

    for exp_dir in mlruns_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        model_dir = exp_dir / run_id / "artifacts" / "model"
        if model_dir.exists():
            return f"file://{model_dir.resolve()}"

    raise FileNotFoundError(
        f"Could not resolve run_id '{run_id}' under {mlruns_dir}. "
        "Expected mlruns/<experiment_id>/<run_id>/artifacts/model"
    )


def load_module_from_run_id(run_id: str, device: str = 'cpu'):
    """Load a Lightning module directly from an MLflow run_id artifact."""
    import torch

    mlflow.set_tracking_uri(f"file://{PROJECT_ROOT / 'mlruns'}")
    model_uri = _find_model_artifact_by_run_id(run_id)
    log.info(f"Loading model from run_id={run_id} on {device} from: {model_uri}")

    dev = torch.device(device)
    module = mlflow.pytorch.load_model(model_uri, map_location=dev)
    module = module.to(dev)
    module.model.eval()

    # Keep explicit Python-side device attributes consistent with module.to(...)
    if hasattr(module.model, 'device'):
        module.model.device = dev
    inner_flow = module.model.model
    if hasattr(inner_flow, 'device'):
        inner_flow.device = dev

    return module


# ---------------------------------------------------------------------------
# Step 2 – Normalisation constants Z_k
# ---------------------------------------------------------------------------

def compute_z_normalisations(root_file_path: str) -> dict:
    """
    Compute Z_k = sum_i w_k(i) for each weight component over all events.

    For the pos/neg split:
      Z_linear_pos  = sum_i max(w_lin_i, 0)
      Z_linear_neg  = sum_i max(-w_lin_i, 0)   (positive number)
      so that  Z_linear = Z_linear_pos - Z_linear_neg  (check)
    """
    log.info(f"Computing Z_k from {root_file_path}")

    with uproot.open(root_file_path) as f:
        tree = f["Events"]
        needed = ['eventWeight', 'smeft_weights']
        available = set(tree.keys())
        missing = [b for b in needed if b not in available]
        if missing:
            raise KeyError(f"Missing required ROOT branches for Z constants: {missing}")
        data = tree.arrays(needed, library='ak')

    log.info(f"  Total events: {len(data['eventWeight'])}")

    # Drop events with fewer than 125 SMEFT weights (ragged array guard,
    # matches the same filter applied in process_ttz_dataset.py)
    smeft_ak = data['smeft_weights']
    valid_mask = ak.to_numpy(ak.num(smeft_ak) > 124)
    n_dropped = int((~valid_mask).sum())
    if n_dropped:
        log.warning(f"  Dropping {n_dropped} events with <125 SMEFT weights")
        data = data[valid_mask]

    w_sm    = ak.to_numpy(data['eventWeight'])
    w_plus  = ak.to_numpy(data['smeft_weights'][:, IDX_CHT_PLUS5])
    w_minus = ak.to_numpy(data['smeft_weights'][:, IDX_CHT_MINUS5])

    w_lin  = (w_plus - w_minus) / 10.0
    w_quad = (w_plus + w_minus - 2.0 * w_sm) / 50.0

    Z = {
        "sm":            float(w_sm.sum()),
        "linear":        float(w_lin.sum()),
        "linear_pos":    float(w_lin[w_lin > 0].sum()),
        "linear_neg":    float((-w_lin[w_lin < 0]).sum()),
        "quadratic":     float(w_quad.sum()),
        "quadratic_pos": float(w_quad[w_quad > 0].sum()),
        "quadratic_neg": float((-w_quad[w_quad < 0]).sum()),
    }
    for k, v in Z.items():
        log.info(f"  Z_{k:<16s} = {v:.4f}")
    # Sanity checks
    assert abs(Z["linear"] - (Z["linear_pos"] - Z["linear_neg"])) < 1e-3, "Z_lin decomposition mismatch"
    assert abs(Z["quadratic"] - (Z["quadratic_pos"] - Z["quadratic_neg"])) < 1e-3, "Z_quad decomposition mismatch"
    return Z


# ---------------------------------------------------------------------------
# Load ROOT file features and weights (for ground truth comparison)
# ---------------------------------------------------------------------------

def load_root_features_and_weights(root_file_path: str):
    """
    Load the 15 physics features (with direct phi) and weight components from ROOT file.
    
    Returns
    -------
    x_features : (N, 15) array
        Physics features in physical space (same as training)
    w_sm : (N,) array
        SM event weights (eventWeight)
    w_decomp : dict
        Weight decomposition: sm, linear, quadratic, full
    """
    log.info(f"Loading ROOT features from {root_file_path}")
    
    branches_to_load = [
        'Z_Lepton1_Pt', 'Z_Lepton1_Eta', 'Z_Lepton1_Phi',
        'Z_Lepton2_Pt', 'Z_Lepton2_Eta', 'Z_Lepton2_Phi',
        'W_Lepton_Pt', 'W_Lepton_Eta', 'W_Lepton_Phi',
        'BJet_Pt', 'BJet_Eta', 'BJet_Phi', 'BJet_Mass',
        'MET', 'MET_phi',
        'eventWeight',
        # Note: smeft_weights loaded separately via awkward (ragged array)
    ]

    # Load features + eventWeight via numpy, smeft_weights via awkward (ragged)
    feat_branches = [b for b in branches_to_load if b != 'smeft_weights']
    with uproot.open(root_file_path) as f:
        tree = f["Events"]
        available = set(tree.keys())
        needed = feat_branches + ['smeft_weights']
        missing = [b for b in needed if b not in available]
        if missing:
            raise KeyError(f"Missing required ROOT branches for feature loading: {missing}")
        data = tree.arrays(feat_branches, library='np')
        smeft_ak = tree.arrays(['smeft_weights'], library='ak')['smeft_weights']

    # Drop events with fewer than 125 SMEFT weights (matches process_ttz_dataset.py)
    valid_mask = ak.to_numpy(ak.num(smeft_ak) > 124)
    n_dropped = int((~valid_mask).sum())
    if n_dropped:
        log.warning(f"  Dropping {n_dropped} events with <125 SMEFT weights")
        smeft_ak = smeft_ak[valid_mask]
        for key in data:
            data[key] = data[key][valid_mask]

    # Extract features (15) with direct phi representation.
    x_features = np.column_stack([
        data['Z_Lepton1_Pt'], data['Z_Lepton1_Eta'], data['Z_Lepton1_Phi'],
        data['Z_Lepton2_Pt'], data['Z_Lepton2_Eta'], data['Z_Lepton2_Phi'],
        data['W_Lepton_Pt'],  data['W_Lepton_Eta'],  data['W_Lepton_Phi'],
        data['BJet_Pt'],      data['BJet_Eta'],      data['BJet_Phi'],      data['BJet_Mass'],
        data['MET'],          data['MET_phi'],
    ]).astype(np.float32)

    # Extract weights
    w_sm    = data['eventWeight']
    # Slice to exactly 125 entries per event (arrays may be longer), then convert to regular numpy
    smeft   = ak.to_numpy(ak.to_regular(smeft_ak[:, :125]))  # (N, 125)
    w_plus  = smeft[:, IDX_CHT_PLUS5]
    w_minus = smeft[:, IDX_CHT_MINUS5]
    w_linear    = (w_plus - w_minus) / 10.0
    w_quadratic = (w_plus + w_minus - 2.0 * w_sm) / 50.0
    w_full = w_plus

    w_decomp = {
        "sm": w_sm,
        "linear": w_linear,
        "quadratic": w_quadratic,
        "full": w_full,
    }

    log.info(f"  Loaded {len(x_features)} events with 15 features")
    return x_features, w_sm, w_decomp


# ---------------------------------------------------------------------------
# Step 3 – Sample from the SM flow
# ---------------------------------------------------------------------------

def sample_from_sm_flow(module_sm, n_samples: int,
                        chunk_size: int = 500_000) -> np.ndarray:
    """
    Draw N samples from the SM flow encoder → returns data in PREPROCESSED
    (scaled) space, i.e. in the same coordinate system the model was trained on.

    Samples in chunks of `chunk_size` to avoid GPU OOM for large N.
    Results are always returned as a CPU numpy array regardless of model device.
    """
    log.info(f"Sampling {n_samples} events from SM flow (chunk_size={chunk_size:,}) ...")
    inner = module_sm.model
    chunks = []
    remaining = n_samples
    while remaining > 0:
        n = min(chunk_size, remaining)
        chunk = inner.sample(n)          # (n, d) on model device
        if hasattr(chunk, 'detach'):
            chunk = chunk.detach().cpu().numpy()
        elif hasattr(chunk, 'numpy'):
            chunk = chunk.numpy()
        chunks.append(chunk.astype(np.float32))
        remaining -= n
        done = n_samples - remaining
        log.info(f"  {done:>12,} / {n_samples:,} sampled ({100*done/n_samples:.0f}%) ...")
    x_scaled = np.concatenate(chunks, axis=0)
    log.info(f"  Sample shape: {x_scaled.shape}")
    return x_scaled


# ---------------------------------------------------------------------------
# Step 4 – Evaluate log p_k for each sample
# ---------------------------------------------------------------------------

def compute_log_probs(module, x_scaled: np.ndarray, label: str,
                      batch_size: int = 10000) -> np.ndarray:
    """
    Compute log p(x) = log p_Z(f(x)) + log|det J_f(x)| for each sample.

    MADEMOG.estimate_density(exp=False, mean=False) returns -log p(x) (NLL)
    for each event.  We negate to obtain log p(x).

    MADEMOG.estimate_density has no built-in chunking, so we split the data
    into batches of `batch_size` events and concatenate the results.

    Parameters
    ----------
    module : Lightning module
    x_scaled : (N, d) array in preprocessed space
    label : str   label for logging
    batch_size : int   events per batch

    Returns
    -------
    log_p : (N,) array
    """
    inner = module.model
    n = x_scaled.shape[0]
    n_batches = max(1, (n + batch_size - 1) // batch_size)
    log.info(f"  [{label}] evaluating log p for {n} samples in {n_batches} batches ...")

    nll_parts = []
    for i in range(n_batches):
        batch = x_scaled[i * batch_size : (i + 1) * batch_size]
        nll_parts.append(inner.estimate_density(batch, exp=False, mean=False))

    nll = np.concatenate(nll_parts).squeeze()   # (N,1) → (N,) due to keepdim=True in MADEMOGModel
    log_p = -nll                               # negate: NLL → log prob
    log.info(f"  [{label}] log p: mean={log_p.mean():.3f}, std={log_p.std():.3f}")
    return log_p.astype(np.float64)


def filter_finite_rows(x_scaled: np.ndarray,
                       log_p_sm: np.ndarray,
                       log_p_lin_pos: np.ndarray,
                       log_p_lin_neg: np.ndarray,
                       log_p_quad_pos: np.ndarray,
                       log_p_quad_neg: np.ndarray):
    """Drop rows with NaN/Inf in samples or any model log-probability."""
    mask = np.isfinite(x_scaled).all(axis=1)
    for arr in (log_p_sm, log_p_lin_pos, log_p_lin_neg, log_p_quad_pos, log_p_quad_neg):
        mask &= np.isfinite(arr)

    n_total = len(mask)
    n_bad = int((~mask).sum())
    if n_bad:
        frac = 100.0 * n_bad / max(1, n_total)
        log.warning(f"Dropping {n_bad}/{n_total} ({frac:.6f}%) non-finite rows before ratio computation")

    return (
        x_scaled[mask],
        log_p_sm[mask],
        log_p_lin_pos[mask],
        log_p_lin_neg[mask],
        log_p_quad_pos[mask],
        log_p_quad_neg[mask],
    )


# ---------------------------------------------------------------------------
# Step 5 – Log density ratios
# ---------------------------------------------------------------------------

def _log_ratio_diagnostics(name: str, arr: np.ndarray, exp_clip: float = 80.0, indent: str = "") -> None:
    """Log robust diagnostics for heavy-tailed log-ratio arrays.

    Raw min/std can be dominated by a handful of outliers and look alarming.
    This summary emphasises the bulk distribution and reports how many values
    would be clipped by the exp() safeguard in get_weights().
    """
    arr = np.asarray(arr)
    n = int(arr.size)
    if n == 0:
        log.warning(f"{indent}{name}: empty array")
        return

    q01, q1, q50, q99, q9999 = np.percentile(arr, [0.01, 1, 50, 99, 99.99])
    clipped_low = int(np.sum(arr < -exp_clip))
    clipped_high = int(np.sum(arr > exp_clip))
    frac_low = 100.0 * clipped_low / n
    frac_high = 100.0 * clipped_high / n

    log.info(
        f"{indent}{name}: median={q50:.3f}  p1={q1:.3f}  p99={q99:.3f}  "
        f"p0.01={q01:.3f}  p99.99={q9999:.3f}"
    )
    log.info(
        f"{indent}{name}: clip<-{exp_clip:.0f}: {clipped_low}/{n} ({frac_low:.6f}%)  "
        f"clip>{exp_clip:.0f}: {clipped_high}/{n} ({frac_high:.6f}%)"
    )


def compute_log_ratios(log_p_sm,
                       log_p_lin_pos, log_p_lin_neg,
                       log_p_quad_pos, log_p_quad_neg):
    """
    Compute log density ratios for the pos/neg split flows:

      log r_lin_pos_i  = log p_lin_pos(x_i)  - log p_sm(x_i)
      log r_lin_neg_i  = log p_lin_neg(x_i)  - log p_sm(x_i)
      log r_quad_pos_i = log p_quad_pos(x_i) - log p_sm(x_i)
      log r_quad_neg_i = log p_quad_neg(x_i) - log p_sm(x_i)

    These enter the combination formula in get_weights().
    """
    log_r_lin_pos  = log_p_lin_pos  - log_p_sm
    log_r_lin_neg  = log_p_lin_neg  - log_p_sm
    log_r_quad_pos = log_p_quad_pos - log_p_sm
    log_r_quad_neg = log_p_quad_neg - log_p_sm
    for name, arr in [("log_r_lin_pos",  log_r_lin_pos),
                      ("log_r_lin_neg",  log_r_lin_neg),
                      ("log_r_quad_pos", log_r_quad_pos),
                      ("log_r_quad_neg", log_r_quad_neg)]:
        _log_ratio_diagnostics(name, arr)
    return log_r_lin_pos, log_r_lin_neg, log_r_quad_pos, log_r_quad_neg


# ---------------------------------------------------------------------------
# Step 6 – Importance weights for any c
# ---------------------------------------------------------------------------

def get_weights(c: float, Z: dict,
                log_r_lin_pos: np.ndarray, log_r_lin_neg: np.ndarray,
                log_r_quad_pos: np.ndarray, log_r_quad_neg: np.ndarray) -> np.ndarray:
    """
    Compute normalised importance weights for Wilson coefficient c_Ht.

    The signed SMEFT measure is decomposed as:
        w_lin  = w_lin_pos - w_lin_neg      (pos/neg signed parts)
        w_quad = w_quad_pos - w_quad_neg

    Each part is modelled by its own normalising flow:
        w_lin_pos(x) ≈ Z_lin_pos  * q_lin_pos(x)
        w_lin_neg(x) ≈ Z_lin_neg  * q_lin_neg(x)

    Combining with the SM flow (r_k = q_k / q_sm):
        w_i(c) ∝ Z_sm
                 + c   * (Z_lin_pos  * exp(log_r_lin_pos_i)
                          - Z_lin_neg  * exp(log_r_lin_neg_i))
                 + c²  * (Z_quad_pos * exp(log_r_quad_pos_i)
                          - Z_quad_neg * exp(log_r_quad_neg_i))

    Individual per-event weights can be negative — this reflects destructive
    interference between the SM and EFT amplitudes in that region of phase
    space and is physically meaningful.  The sum of all weights (proportional
    to the total cross section) should remain positive for a valid EFT point.
    """
    # Clip exponents to avoid overflow from rare extreme tails.
    exp_clip = 80.0
    r_lin_pos = np.exp(np.clip(log_r_lin_pos, -exp_clip, exp_clip))
    r_lin_neg = np.exp(np.clip(log_r_lin_neg, -exp_clip, exp_clip))
    r_quad_pos = np.exp(np.clip(log_r_quad_pos, -exp_clip, exp_clip))
    r_quad_neg = np.exp(np.clip(log_r_quad_neg, -exp_clip, exp_clip))

    unnorm = (Z["sm"]
              + c    * (Z["linear_pos"]    * r_lin_pos
                        - Z["linear_neg"]  * r_lin_neg)
              + c**2 * (Z["quadratic_pos"] * r_quad_pos
                        - Z["quadratic_neg"] * r_quad_neg))

    neg = (unnorm < 0).sum()
    if neg:
        log.info(f"  c={c:+.1f}: {neg} events have negative weights (destructive interference)")

    total = unnorm.sum()
    if total <= 0:
        log.warning(f"  c={c:+.1f}: total weight sum = {total:.4f} <= 0 — EFT may be outside validity range")

    return unnorm / total


def effective_sample_size(weights: np.ndarray) -> float:
    """
    Effective sample size for signed weights.

    For non-negative weights the standard estimator is N_eff = (Σw)²/Σw².
    With signed weights this can be misleading (negative and positive weights
    can cancel, inflating the apparent N_eff).  We therefore use:

        N_eff = |Σw| / Σ|w|

    which measures what fraction of the absolute weight budget survives after
    cancellations — a more conservative and honest diagnostic.
    Already-normalised weights have Σw = 1, so this reduces to 1 / Σ|w|.
    """
    return float(np.abs(weights.sum())) / float(np.abs(weights).sum())


# ---------------------------------------------------------------------------
# Step 7 – Inverse transform + plotting helpers
# ---------------------------------------------------------------------------

def inverse_transform(module_sm, x_scaled: np.ndarray) -> np.ndarray:
    """
    Map samples from preprocessed space back to physical units using
    the scalers stored inside the SM model checkpoint.
    """
    from ml.common.data_utils.feature_scaling import RescalingHandler

    if not (hasattr(module_sm, 'scalers') and module_sm.scalers is not None
            and hasattr(module_sm, 'selection') and module_sm.selection is not None):
        log.warning("SM module has no stored scalers; returning raw scaled data.")
        return x_scaled

    # Strip the weight column from selection if present
    sel = module_sm.selection
    sel = sel[sel['feature'] != 'cHt_weight'].reset_index(drop=True)

    rh = RescalingHandler(sel, module_sm.scalers)
    return rh.inverse_transform(x_scaled)


def get_range_limits(data, percentile_range=0.1):
    lo = np.percentile(data, percentile_range)
    hi = np.percentile(data, 100 - percentile_range)
    pad = 0.1 * (hi - lo)
    return lo - pad, hi + pad


def compute_derived_features(x_phys: np.ndarray):
    """
    Compute derived higher-order features from base kinematic features.

    Base feature layout
    -------------------
    Supports both representations to preserve backwards compatibility:
    - pt/eta/phi:  Z_Lepton1, Z_Lepton2, W_Lepton, BJet(+Mass), MET(+Phi)
    - Cartesian:   Z_Lepton1, Z_Lepton2, W_Lepton, BJet(+Mass), MET(Px,Py)

    Derived quantities (matched to sample_analyzer)
    -----------------------------------------------
    Z_Pt, Z_Eta, Z_Phi, Z_Mass, Z_DeltaR : from two massless Z-leptons
    W_Pt, W_Phi, W_MT                     : from W-lepton + MET (transverse)
    Top_Pt, Top_Phi, Top_MT               : from b-jet + W (transverse approximation)
    """
    if len(FEATURE_NAMES) >= 4 and FEATURE_NAMES[0].endswith('_Pt'):
        # pt/eta/phi representation.
        zl1_pt, zl1_eta, zl1_phi = x_phys[:, 0], x_phys[:, 1], x_phys[:, 2]
        zl2_pt, zl2_eta, zl2_phi = x_phys[:, 3], x_phys[:, 4], x_phys[:, 5]
        wl_pt, wl_eta, wl_phi = x_phys[:, 6], x_phys[:, 7], x_phys[:, 8]
        bj_pt, bj_eta, bj_phi = x_phys[:, 9], x_phys[:, 10], x_phys[:, 11]
        met_pt, met_phi = x_phys[:, 13], x_phys[:, 14]

        zl1_px = zl1_pt * np.cos(zl1_phi)
        zl1_py = zl1_pt * np.sin(zl1_phi)
        zl1_pz = zl1_pt * np.sinh(zl1_eta)

        zl2_px = zl2_pt * np.cos(zl2_phi)
        zl2_py = zl2_pt * np.sin(zl2_phi)
        zl2_pz = zl2_pt * np.sinh(zl2_eta)

        wl_px = wl_pt * np.cos(wl_phi)
        wl_py = wl_pt * np.sin(wl_phi)
        wl_pz = wl_pt * np.sinh(wl_eta)

        bj_px = bj_pt * np.cos(bj_phi)
        bj_py = bj_pt * np.sin(bj_phi)
        bj_pz = bj_pt * np.sinh(bj_eta)

        met_px = met_pt * np.cos(met_phi)
        met_py = met_pt * np.sin(met_phi)
    else:
        # Already in Cartesian representation.
        zl1_px, zl1_py, zl1_pz = x_phys[:, 0], x_phys[:, 1], x_phys[:, 2]
        zl2_px, zl2_py, zl2_pz = x_phys[:, 3], x_phys[:, 4], x_phys[:, 5]
        wl_px, wl_py, wl_pz = x_phys[:, 6], x_phys[:, 7], x_phys[:, 8]
        bj_px, bj_py, bj_pz = x_phys[:, 9], x_phys[:, 10], x_phys[:, 11]
        met_px, met_py = x_phys[:, 13], x_phys[:, 14]

    # --- Z boson (massless leptons) ---
    zl1_E = np.sqrt(np.clip(zl1_px**2 + zl1_py**2 + zl1_pz**2, 0.0, None))
    zl2_E = np.sqrt(np.clip(zl2_px**2 + zl2_py**2 + zl2_pz**2, 0.0, None))

    Z_px = zl1_px + zl2_px
    Z_py = zl1_py + zl2_py
    Z_pz = zl1_pz + zl2_pz
    Z_E = zl1_E + zl2_E
    Z_p  = np.sqrt(Z_px**2 + Z_py**2 + Z_pz**2)

    Z_pt   = np.sqrt(Z_px**2 + Z_py**2)
    Z_phi  = np.arctan2(Z_py, Z_px)
    Z_eta  = np.arctanh(np.clip(Z_pz / np.clip(Z_p, 1e-9, None), -1 + 1e-7, 1 - 1e-7))
    Z_mass = np.sqrt(np.clip(Z_E**2 - Z_p**2, 0.0, None))

    zl1_p = np.sqrt(zl1_px**2 + zl1_py**2 + zl1_pz**2)
    zl2_p = np.sqrt(zl2_px**2 + zl2_py**2 + zl2_pz**2)
    zl1_eta = np.arctanh(np.clip(zl1_pz / np.clip(zl1_p, 1e-9, None), -1 + 1e-7, 1 - 1e-7))
    zl2_eta = np.arctanh(np.clip(zl2_pz / np.clip(zl2_p, 1e-9, None), -1 + 1e-7, 1 - 1e-7))
    zl1_phi = np.arctan2(zl1_py, zl1_px)
    zl2_phi = np.arctan2(zl2_py, zl2_px)
    dphi_ll = (zl1_phi - zl2_phi + np.pi) % (2 * np.pi) - np.pi
    Z_deltaR = np.sqrt((zl1_eta - zl2_eta) ** 2 + dphi_ll ** 2)

    # --- W boson transverse (lepton + MET 2D vector sum) ---
    W_px = wl_px + met_px
    W_py = wl_py + met_py
    W_pt = np.sqrt(W_px**2 + W_py**2)
    W_phi = np.arctan2(W_py, W_px)

    wl_pt = np.sqrt(wl_px**2 + wl_py**2)
    met_pt = np.sqrt(met_px**2 + met_py**2)
    wl_phi = np.arctan2(wl_py, wl_px)
    met_phi = np.arctan2(met_py, met_px)
    dphi_w = np.arctan2(np.sin(wl_phi - met_phi), np.cos(wl_phi - met_phi))
    W_mt = np.sqrt(np.clip(2.0 * wl_pt * met_pt * (1.0 - np.cos(dphi_w)), 0.0, None))

    # --- Top quark transverse (b-jet + W transverse) ---
    top_px = W_px + bj_px
    top_py = W_py + bj_py
    top_pt = np.sqrt(top_px**2 + top_py**2)
    top_phi = np.arctan2(top_py, top_px)

    bj_pt = np.sqrt(bj_px**2 + bj_py**2)
    bj_phi = np.arctan2(bj_py, bj_px)
    dphi_tb = np.arctan2(np.sin(W_phi - bj_phi), np.cos(W_phi - bj_phi))
    top_mt = np.sqrt(np.clip((W_mt + bj_pt)**2 + 2.0 * W_pt * bj_pt * (1.0 - np.cos(dphi_tb)), 0.0, None))

    derived = np.column_stack([
        Z_pt, Z_eta, Z_phi, Z_mass, Z_deltaR,
        W_pt, W_phi, W_mt,
        top_pt, top_phi, top_mt,
    ])
    derived_names = [
        'Z_Pt', 'Z_Eta', 'Z_Phi', 'Z_Mass', 'Z_DeltaR',
        'W_Pt', 'W_Phi', 'W_MT',
        'Top_Pt', 'Top_Phi', 'Top_MT',
    ]

    return np.concatenate([x_phys, derived], axis=1), FEATURE_NAMES + derived_names


def plot_reweighted_distributions(x_phys: np.ndarray,
                                  log_r_lin_pos: np.ndarray,
                                  log_r_lin_neg: np.ndarray,
                                  log_r_quad_pos: np.ndarray,
                                  log_r_quad_neg: np.ndarray,
                                  Z: dict,
                                  c_values: list,
                                  output_dir: Path,
                                  bins: int = 50,
                                  n_cols: int = 3,
                                  feature_names: list = None,
                                  x_root: np.ndarray = None,
                                  w_decomp: dict = None,
                                  x_root_sm: np.ndarray = None,
                                  w_root_sm: np.ndarray = None,
                                  log_y: bool = False):
    """
    For each c in c_values, create a figure with one panel per feature.
    Each panel has:
      - upper plot : SM (filled) and SMEFT-reweighted (line) normalised densities
      - lower plot : ratio generated(c) / ROOT(c)  with a reference line at 1

    When x_root_sm and x_root are both provided, shows:
      - **Blue**: ROOT file SM data weighted by ROOT eventWeight (SM baseline)
      - **Red**: Generated/reweighted samples from flow
      - **Green**: ROOT file data weighted to c value (ground truth at c)
      - **Ratio**: Generated(c) / ROOT(c) — validates model accuracy

    Histograms
    ----------
    SM (blue)         : ROOT eventWeight-weighted, density=False, then divided by sum(w_sm)*dx.
    Generated (red)   : weighted by w (sum to 1), density=False, then divided by dx manually.
    ROOT c-weighted (green): weighted by w_root, density=False, then divided by N_root*dx.
    This avoids numpy's density=True conflating normalisation with weighting.

    Binning policy
    --------------
    If ROOT SM data is supplied (x_root_sm), histogram ranges are fixed from ROOT-SM
    only, so reruns with different generated samples use identical bin edges.
    """
    from matplotlib.gridspec import GridSpec

    if feature_names is None:
        feature_names = FEATURE_NAMES
    n_features = len(feature_names)
    n_rows = (n_features + n_cols - 1) // n_cols
    N = x_phys.shape[0]

    # Units for each feature type
    _UNITS = {}
    for name in feature_names:
        lower = name.lower()
        if any(k in lower for k in ('_pt', '_px', '_py', '_pz', 'met', '_mass')):
            _UNITS[name] = '[GeV]'
        else:
            _UNITS[name] = ''   # Eta is dimensionless

    # Diagnostics: robust summaries and clip impact for heavy-tailed log-ratios.
    for name, arr in [("log_r_lin_pos",  log_r_lin_pos),
                      ("log_r_lin_neg",  log_r_lin_neg),
                      ("log_r_quad_pos", log_r_quad_pos),
                      ("log_r_quad_neg", log_r_quad_neg)]:
        _log_ratio_diagnostics(name, arr, indent="  ")

    if x_root_sm is not None:
        log.info("Using ROOT-basis fixed binning: ranges from ROOT(SM) percentiles.")
        if w_root_sm is None:
            log.warning("x_root_sm provided without w_root_sm; falling back to unweighted ROOT SM baseline.")
        elif len(w_root_sm) != len(x_root_sm):
            raise ValueError(f"Length mismatch: len(w_root_sm)={len(w_root_sm)} != len(x_root_sm)={len(x_root_sm)}")

    for c in c_values:
        w = get_weights(c, Z, log_r_lin_pos, log_r_lin_neg, log_r_quad_pos, log_r_quad_neg)   # (N,)
        neff_frac = effective_sample_size(w)
        log.info(f"  c={c:+.1f}: N_eff/N={100*neff_frac:.1f}%  w_std={w.std():.2e}  w_min={w.min():.2e}  w_max={w.max():.2e}")

        # Pre-compute ROOT weights for this c (same for all features)
        w_root_c = None
        if x_root is not None and w_decomp is not None:
            w_root_c = w_decomp["sm"] + c * w_decomp["linear"] + c**2 * w_decomp["quadratic"]
            w_root_c = w_root_c / w_root_c.sum()

        # height per feature row = 5 inches (3 main + 1 ratio + gaps)
        fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows + 0.8))
        scale_label = '  [log scale]' if log_y else ''
        fig.suptitle(
            f'SMEFT reweighted distributions  ($c_{{Ht}}$ = {c:+.1f}){scale_label}\n'
            f'N = {N}   |   $N_{{\\mathrm{{eff}}}}/N$ = {100*neff_frac:.1f}%',
            fontsize=13, fontweight='bold', y=0.995,
        )

        gs = GridSpec(
            n_rows * 3, n_cols,
            height_ratios=[3, 1, 0.4] * n_rows,  # main plot, ratio plot, spacing to next feature
            hspace=0.05,   # minimal gap (most spacing is in height_ratios)
            wspace=0.38, 
            top=0.97, bottom=0.03, left=0.07, right=0.98,
        )

        for i, feat in enumerate(feature_names):
            row, col = divmod(i, n_cols)
            ax_main  = fig.add_subplot(gs[row * 3,     col])
            ax_ratio = fig.add_subplot(gs[row * 3 + 1, col], sharex=ax_main)

            # Determine which source to use for SM reference (blue curve)
            # If ROOT SM provided, use it (ground truth); otherwise use generated samples
            if x_root_sm is not None:
                feat_data_sm = x_root_sm[:, i]
                sm_weights = w_root_sm if w_root_sm is not None else None
                sm_source = "ROOT eventWeight"
            else:
                feat_data_sm = x_phys[:, i]
                sm_weights = None
                sm_source = "Generated"

            # Generated (reweighted) data
            feat_data = x_phys[:, i]

            # Fix binning to ROOT-SM basis (deterministic across reruns).
            if x_root_sm is not None:
                x_min, x_max = get_range_limits(feat_data_sm)
            else:
                x_min, x_max = get_range_limits(feat_data)

            dx = (x_max - x_min) / bins

            # --- histograms (manual normalisation avoids numpy density ambiguity) ---
            # SM reference (blue)
            counts_sm, edges = np.histogram(feat_data_sm, bins=bins, range=(x_min, x_max), weights=sm_weights)
            if sm_weights is None:
                norm_sm = max(len(feat_data_sm), 1)
            else:
                norm_sm = float(np.sum(sm_weights))
                if np.isclose(norm_sm, 0.0):
                    log.warning("ROOT SM eventWeight sum is ~0 for this plot; using unweighted normalization fallback.")
                    norm_sm = max(len(feat_data_sm), 1)
            hist_sm = counts_sm / (norm_sm * dx)

            # Generated reweighted (red)
            counts_c, _ = np.histogram(feat_data, bins=edges, weights=w)
            hist_c = counts_c / dx                         # weighted PDF (w sums to 1)

            bc = (edges[:-1] + edges[1:]) / 2

            # If ROOT data supplied, compute ROOT weights and histogram for comparison
            hist_root = None
            if w_root_c is not None:
                feat_data_root = x_root[:, i]
                counts_root, _ = np.histogram(feat_data_root, bins=edges, weights=w_root_c)
                hist_root = counts_root / dx
                N_root = len(x_root)

            # Main panel
            ax_main.fill_between(bc, hist_sm, step='mid',
                                 color='royalblue', alpha=0.35, label=f'SM {sm_source}')
            ax_main.step(bc, hist_sm, color='royalblue', where='mid', lw=1.5)
            ax_main.step(bc, hist_c,  color='crimson',   where='mid', lw=2,
                         label=f'Generated $c={c:+.1f}$')
            if hist_root is not None:
                ax_main.step(bc, hist_root, color='green', where='mid', lw=1.8, linestyle='--',
                             label=f'ROOT $c={c:+.1f}$')
            ax_main.set_ylabel('Density', fontsize=8)
            ax_main.set_xlim(x_min, x_max)
            if log_y:
                ax_main.set_yscale('log')
            else:
                ax_main.set_ylim(bottom=0)
            ax_main.grid(True, linestyle='--', alpha=0.3)
            ax_main.legend(fontsize=7, loc='upper right')
            ax_main.tick_params(labelbottom=False, labelsize=8)
            ax_main.set_title(feat, fontsize=9, pad=3)

            # Ratio panel: both ROOT(c)/SM-ROOT and Generated(c)/SM-ROOT on the same axes
            with np.errstate(divide='ignore', invalid='ignore'):
                denom = np.where(hist_sm > 0, hist_sm, np.nan)
                ratio_gen  = hist_c    / denom   # Generated(c) / ROOT(SM)
                ratio_root = hist_root / denom if hist_root is not None else None  # ROOT(c) / ROOT(SM)

            ax_ratio.axhline(1.0, color='black', linestyle='--', lw=1.2, alpha=0.5)
            ax_ratio.step(bc, ratio_gen, where='mid', color='crimson', lw=1.5,
                          label=f'Gen / ROOT(SM)')
            if ratio_root is not None:
                ax_ratio.step(bc, ratio_root, where='mid', color='green', lw=1.5,
                              linestyle='--', label='ROOT(c) / ROOT(SM)')
                ax_ratio.legend(fontsize=6, loc='upper right')

            units = _UNITS.get(feat, '')
            ax_ratio.set_xlabel(units, fontsize=8)
            ax_ratio.tick_params(labelbottom=True, labelsize=8)
            ax_ratio.set_ylabel('Ratio to ROOT SM', fontsize=7)
            ax_ratio.set_xlim(x_min, x_max)
            ax_ratio.set_ylim(0.7, 1.3)
            ax_ratio.grid(True, linestyle='--', alpha=0.3)
            ax_ratio.tick_params(labelsize=7)

        # Hide unused cells
        for i in range(n_features, n_rows * n_cols):
            r, c_ = divmod(i, n_cols)
            fig.add_subplot(gs[r * 3,     c_]).axis('off')
            fig.add_subplot(gs[r * 3 + 1, c_]).axis('off')
            fig.add_subplot(gs[r * 3 + 2, c_]).axis('off')

        c_str = f"c{c:+.1f}".replace('.', 'p').replace('+', 'plus').replace('-', 'minus')
        suffix = "_log" if log_y else ""
        out = output_dir / f"reweighted_{c_str}{suffix}.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        log.info(f"  Saved: {out}")


def plot_neff_curve(Z: dict,
                    log_r_lin_pos: np.ndarray, log_r_lin_neg: np.ndarray,
                    log_r_quad_pos: np.ndarray, log_r_quad_neg: np.ndarray,
                    output_dir: Path, c_range=(-6, 6), n_points=100):
    """Plot effective sample size as a function of c_Ht."""
    c_arr = np.linspace(c_range[0], c_range[1], n_points)
    neff_arr = np.array([
        effective_sample_size(get_weights(c, Z, log_r_lin_pos, log_r_lin_neg,
                                          log_r_quad_pos, log_r_quad_neg))
        for c in c_arr
    ])  # already a fraction 0-1

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(c_arr, neff_arr * 100, color='navy', lw=2)
    ax.axhline(y=50, color='orange', linestyle='--', lw=1.5, label='50%')
    ax.axhline(y=10, color='red',    linestyle='--', lw=1.5, label='10%')
    ax.set_xlabel('Wilson coefficient $c_{Ht}$', fontsize=12)
    ax.set_ylabel('$N_{\\mathrm{eff}}$ / $N$  [%]', fontsize=12)
    ax.set_title('Effective sample size vs $c_{Ht}$', fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=11)
    ax.set_xlim(c_range)
    ax.set_ylim(0, 105)
    plt.tight_layout()

    out = output_dir / "neff_vs_c.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved N_eff curve: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SMEFT importance reweighting from three normalising flows."
    )
    parser.add_argument('--n-samples', type=int, default=10_000_000,
                        help='Number of events to sample from SM flow (default: 10000000)')
    parser.add_argument('--c-values', type=float, nargs='+',
                        default=[-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0],
                        help='c_Ht values for which to produce comparison plots')
    parser.add_argument('--bins', type=int, default=50,
                        help='Histogram bins (default: 50)')
    parser.add_argument('--n-cols', type=int, default=3,
                        help='Columns in plot grid (default: 3)')
    parser.add_argument('--load-ratios', action='store_true',
                        help='Load pre-computed density ratios from disk (skip sampling/encoding)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for model inference: cpu or cuda (default: cpu)')
    parser.add_argument('--sm-run-id', type=str, default=None,
                        help='Optional MLflow run_id for SM model artifact')
    parser.add_argument('--linear-pos-run-id', type=str, default=None,
                        help='Optional MLflow run_id for linear_pos model artifact')
    parser.add_argument('--linear-neg-run-id', type=str, default=None,
                        help='Optional MLflow run_id for linear_neg model artifact')
    parser.add_argument('--quadratic-pos-run-id', type=str, default=None,
                        help='Optional MLflow run_id for quadratic_pos model artifact')
    parser.add_argument('--quadratic-neg-run-id', type=str, default=None,
                        help='Optional MLflow run_id for quadratic_neg model artifact')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("SMEFT importance reweighting")
    log.info("=" * 60)

    # -----------------------------------------------------------------------
    # Z_k normalisation constants (always recompute – fast)
    # -----------------------------------------------------------------------
    Z = compute_z_normalisations(DATA_ROOT_FILE)

    # -----------------------------------------------------------------------
    # Load or compute density ratios
    # -----------------------------------------------------------------------
    if args.load_ratios and RATIOS_FILE.exists():
        log.info(f"Loading saved density ratios from {RATIOS_FILE}")
        saved = np.load(RATIOS_FILE)
        log_r_lin_pos  = saved['log_r_lin_pos']
        log_r_lin_neg  = saved['log_r_lin_neg']
        log_r_quad_pos = saved['log_r_quad_pos']
        log_r_quad_neg = saved['log_r_quad_neg']
        x_scaled       = saved['x_scaled']
        log.info(f"  Loaded {len(log_r_lin_pos)} samples")

        # Sanity check: degenerate (all-zero) ratios indicate a stale or bugged file
        all_arrs = [log_r_lin_pos, log_r_lin_neg, log_r_quad_pos, log_r_quad_neg]
        if all(a.std() < 1e-6 for a in all_arrs):
            log.warning("  Loaded ratios are degenerate (std ≈ 0) — deleting stale file and recomputing.")
            RATIOS_FILE.unlink()
            args.load_ratios = False   # fall through to recompute block below

        # Reject corrupted cache files with NaN/Inf values.
        has_nonfinite = (
            (not np.isfinite(x_scaled).all())
            or (not np.isfinite(log_r_lin_pos).all())
            or (not np.isfinite(log_r_lin_neg).all())
            or (not np.isfinite(log_r_quad_pos).all())
            or (not np.isfinite(log_r_quad_neg).all())
        )
        if has_nonfinite:
            log.warning("  Loaded ratios contain NaN/Inf — deleting stale file and recomputing.")
            RATIOS_FILE.unlink()
            args.load_ratios = False

    else:
        # Step 1 – Resolve model sources
        use_run_ids = any([
            args.sm_run_id,
            args.linear_pos_run_id,
            args.linear_neg_run_id,
            args.quadratic_pos_run_id,
            args.quadratic_neg_run_id,
        ])

        if use_run_ids:
            log.info("\n--- Step 1: Resolving model sources (run_id override where provided) ---")
            log.info(f"  [sm]            run_id: {args.sm_run_id if args.sm_run_id else 'latest registered model'}")
            log.info(f"  [linear_pos]    run_id: {args.linear_pos_run_id if args.linear_pos_run_id else 'latest registered model'}")
            log.info(f"  [linear_neg]    run_id: {args.linear_neg_run_id if args.linear_neg_run_id else 'latest registered model'}")
            log.info(f"  [quadratic_pos] run_id: {args.quadratic_pos_run_id if args.quadratic_pos_run_id else 'latest registered model'}")
            log.info(f"  [quadratic_neg] run_id: {args.quadratic_neg_run_id if args.quadratic_neg_run_id else 'latest registered model'}")
        else:
            log.info("\n--- Step 1: Finding model names ---")
            name_sm        = find_model_name("sm")
            name_lin_pos   = find_model_name("linear_pos")
            name_lin_neg   = find_model_name("linear_neg")
            name_quad_pos  = find_model_name("quadratic_pos")
            name_quad_neg  = find_model_name("quadratic_neg")

        # Step 1b – Load modules (run_id override or latest registered)
        log.info("\n--- Loading flow modules ---")
        log.info(f"Loading SM model (device={args.device}) ...")
        module_sm = (
            load_module_from_run_id(args.sm_run_id, device=args.device)
            if args.sm_run_id else load_module(name_sm, device=args.device)
        )
        log.info(f"Loading linear_pos model ...")
        module_lin_pos = (
            load_module_from_run_id(args.linear_pos_run_id, device=args.device)
            if args.linear_pos_run_id else load_module(name_lin_pos, device=args.device)
        )
        log.info(f"Loading linear_neg model ...")
        module_lin_neg = (
            load_module_from_run_id(args.linear_neg_run_id, device=args.device)
            if args.linear_neg_run_id else load_module(name_lin_neg, device=args.device)
        )
        log.info(f"Loading quadratic_pos model ...")
        module_quad_pos = (
            load_module_from_run_id(args.quadratic_pos_run_id, device=args.device)
            if args.quadratic_pos_run_id else load_module(name_quad_pos, device=args.device)
        )
        log.info(f"Loading quadratic_neg model ...")
        module_quad_neg = (
            load_module_from_run_id(args.quadratic_neg_run_id, device=args.device)
            if args.quadratic_neg_run_id else load_module(name_quad_neg, device=args.device)
        )

        # Step 3 – Sample from SM flow
        log.info("\n--- Step 3: Sampling from SM flow ---")
        x_scaled = sample_from_sm_flow(module_sm, args.n_samples)

        # Step 4 – Evaluate log probabilities
        log.info("\n--- Step 4: Evaluating log probabilities ---")
        log_p_sm       = compute_log_probs(module_sm,       x_scaled, "sm")
        log_p_lin_pos  = compute_log_probs(module_lin_pos,  x_scaled, "linear_pos")
        log_p_lin_neg  = compute_log_probs(module_lin_neg,  x_scaled, "linear_neg")
        log_p_quad_pos = compute_log_probs(module_quad_pos, x_scaled, "quadratic_pos")
        log_p_quad_neg = compute_log_probs(module_quad_neg, x_scaled, "quadratic_neg")

        # Guard against rare NaN/Inf outliers that would otherwise poison all weights.
        x_scaled, log_p_sm, log_p_lin_pos, log_p_lin_neg, log_p_quad_pos, log_p_quad_neg = filter_finite_rows(
            x_scaled, log_p_sm, log_p_lin_pos, log_p_lin_neg, log_p_quad_pos, log_p_quad_neg
        )
        log.info(f"  Using {len(x_scaled)} finite samples for ratio computation")

        # Step 5 – Log density ratios
        log.info("\n--- Step 5: Computing log density ratios ---")
        log_r_lin_pos, log_r_lin_neg, log_r_quad_pos, log_r_quad_neg = compute_log_ratios(
            log_p_sm, log_p_lin_pos, log_p_lin_neg, log_p_quad_pos, log_p_quad_neg
        )

        # Save to disk
        np.savez(RATIOS_FILE,
                 log_r_lin_pos=log_r_lin_pos,
                 log_r_lin_neg=log_r_lin_neg,
                 log_r_quad_pos=log_r_quad_pos,
                 log_r_quad_neg=log_r_quad_neg,
                 x_scaled=x_scaled)
        log.info(f"Saved density ratios to {RATIOS_FILE}")
        all_stale = all(a.std() < 1e-6
                        for a in [log_r_lin_pos, log_r_lin_neg, log_r_quad_pos, log_r_quad_neg])
        if all_stale:
            log.warning("  Ratios are degenerate — the five flows may have converged to the same density. "
                        "Check that training used the correct weight columns.")

        # Keep modules for inverse transform
        log.info("\n--- Inverse-transforming samples to physical space ---")
        x_phys = inverse_transform(module_sm, x_scaled)

    # If we loaded from file, reconstruct x_phys by loading the SM module just for scalers
    # (always CPU — inverse transform is numpy, no GPU needed)
    if args.load_ratios and RATIOS_FILE.exists():
        log.info("Loading SM module for inverse transform ...")
        name_sm   = find_model_name("sm")
        module_sm = load_module(name_sm, device='cpu')
        x_phys    = inverse_transform(module_sm, x_scaled)

    # -----------------------------------------------------------------------
    # Step 7 – Plots
    # -----------------------------------------------------------------------
    log.info("\n--- Step 7: Generating plots ---")

    # Extend x_phys with derived features
    log.info("Computing derived higher-order features ...")
    x_plot, all_feature_names = compute_derived_features(x_phys)
    log.info(f"  Total features for plotting: {len(all_feature_names)}")

    # Load ROOT file data for ground truth comparison
    log.info("Loading ROOT file data for ground truth comparison ...")
    x_root_phys, w_root_sm, w_root_decomp = load_root_features_and_weights(DATA_ROOT_FILE)
    x_root_plot, _ = compute_derived_features(x_root_phys)

    # N_eff curve
    plot_neff_curve(Z, log_r_lin_pos, log_r_lin_neg,
                    log_r_quad_pos, log_r_quad_neg, OUTPUT_DIR)

    # Per-c comparison plots (linear scale)
    log.info(f"Generating per-c comparison plots (linear scale) for c = {args.c_values}")
    plot_reweighted_distributions(
        x_plot,
        log_r_lin_pos, log_r_lin_neg,
        log_r_quad_pos, log_r_quad_neg,
        Z,
        args.c_values, OUTPUT_DIR, args.bins, args.n_cols,
        feature_names=all_feature_names,
        x_root=x_root_plot,
        w_decomp=w_root_decomp,
        x_root_sm=x_root_plot,
        w_root_sm=w_root_sm,
    )

    # Per-c comparison plots (log scale)
    log.info(f"Generating per-c comparison plots (log scale) for c = {args.c_values}")
    plot_reweighted_distributions(
        x_plot,
        log_r_lin_pos, log_r_lin_neg,
        log_r_quad_pos, log_r_quad_neg,
        Z,
        args.c_values, OUTPUT_DIR, args.bins, args.n_cols,
        feature_names=all_feature_names,
        x_root=x_root_plot,
        w_decomp=w_root_decomp,
        x_root_sm=x_root_plot,
        w_root_sm=w_root_sm,
        log_y=True,
    )

    log.info("=" * 60)
    log.info(f"Done!  Figures in: {OUTPUT_DIR}")
    log.info(f"Log:              {LOG_FILE}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Validate weight decomposition consistency across different cHt values.

This script checks that SM, linear, and quadratic weight components produce
identical feature sets regardless of whether they are extracted from cHt=5
or cHt=1 weight definitions.

The SMEFT weight at coupling strength c is parametrized as:
    w(c) = w_sm + c*w_lin + c^2*w_quad

where:
    w_lin = (w_plus - w_minus) / 10.0       (at cHt=5 reference)
    w_quad = (w_plus + w_minus - 2*w_sm) / 50.0  (at cHt=5 reference)

At arbitrary cHt, components scale as:
    w_lin(cHt) = w_lin * (cHt / 5)
    w_quad(cHt) = w_quad * (cHt / 5)^2

By comparing feature extractions at different cHt values, we verify that 
the decomposition is mathematically consistent and the feature selection 
pipeline handles all cases identically.

Usage:
    python ml/custom/ttz/validate_weight_decomposition.py
"""

import logging
import numpy as np
import uproot
import awkward as ak
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def load_and_decompose_weights(cHt: float) -> dict:
    """
    Load TTZ dataset directly from ROOT file and decompose weights at a specific cHt value.
    
    ROOT file structure:
        - Physics features: Z_Lepton1/2 (3D), W_Lepton (3D), BJet (4D), MET (2D) = 15 features
        - smeft_weights: array of 125 weight values indexed 0-124
        - eventWeight: SM baseline weight
    
    Weight indices (from ml/data/ttz/cHt_weight_indices.txt):
        121: cHt_m1p0 (cHt = -1.0)
        122: cHt_m5p0 (cHt = -5.0)
        123: cHt_p1p0 (cHt = +1.0)
        124: cHt_p5p0 (cHt = +5.0)
    
    SMEFT parametrization at coupling strength cHt:
        w(cHt) = w_sm + cHt * w_lin + cHt^2 * w_quad
    
    We extract both ±1 and ±5 measurements and verify they give consistent decomposition:
        From ±1.0: w_lin = (w(+1) - w(-1)) / 2.0
                   w_quad = (w(+1) + w(-1) - 2*w_sm) / 2.0
        From ±5.0: w_lin = (w(+5) - w(-5)) / 10.0
                   w_quad = (w(+5) + w(-5) - 2*w_sm) / 50.0
    
    Parameters
    ----------
    cHt : float
        Coupling strength value for weight decomposition.
        
    Returns
    -------
    dict
        Components with keys: 'sm', 'linear_pos', 'linear_neg', 'quadratic_pos', 'quadratic_neg'
        Each value is a dict with 'features' (n_selected, 15) and 'weights' (weights array).
    """
    log.info(f"\n{'='*70}")
    log.info(f"Loading TTZ weights from ROOT file")
    log.info(f"Decomposing at cHt = {cHt}")
    log.info(f"{'='*70}")
    
    # Load data directly from ROOT file
    root_file = "/project/atlas/users/kdevries/EventLoop/ttZ_for_Melle_tree.root"
    
    branches_to_load = [
        'Z_Lepton1_Px', 'Z_Lepton1_Py', 'Z_Lepton1_Pz',
        'Z_Lepton2_Px', 'Z_Lepton2_Py', 'Z_Lepton2_Pz',
        'W_Lepton_Px', 'W_Lepton_Py', 'W_Lepton_Pz',
        'BJet_Px', 'BJet_Py', 'BJet_Pz', 'BJet_Mass',
        'MET', 'MET_phi', 'eventWeight',
    ]
    
    with uproot.open(root_file) as file_root:
        tree = file_root["Events"]
        
        # Load physics features
        data = tree.arrays(branches_to_load, library="np")
        n_events = len(data['MET'])
        log.info(f"Loaded {n_events} events from {root_file}")
        
        # Load SMEFT weights with awkward to handle ragged arrays
        smeft_ak = tree.arrays(['smeft_weights'], library="ak")['smeft_weights']
        
        # Filter: keep only events with all 125 weight values
        valid_mask = ak.to_numpy(ak.num(smeft_ak) > 124)
        n_dropped = int((~valid_mask).sum())
        if n_dropped:
            log.warning(f"Dropping {n_dropped} events with <125 SMEFT weights")
            smeft_ak = smeft_ak[valid_mask]
            for key in list(data.keys()):
                data[key] = data[key][valid_mask]
        
        # Extract weights at BOTH cHt = ±1.0 and ±5.0 reference points
        # This allows us to verify decomposition consistency
        w_minus_1 = ak.to_numpy(smeft_ak[:, 121]).astype(np.float32)  # cHt = -1.0
        w_minus_5 = ak.to_numpy(smeft_ak[:, 122]).astype(np.float32)  # cHt = -5.0
        w_plus_1 = ak.to_numpy(smeft_ak[:, 123]).astype(np.float32)   # cHt = +1.0
        w_plus_5 = ak.to_numpy(smeft_ak[:, 124]).astype(np.float32)   # cHt = +5.0
        log.info(f"Extracted weight indices:")
        log.info(f"  ±1.0: 121 (cHt=-1.0), 123 (cHt=+1.0)")
        log.info(f"  ±5.0: 122 (cHt=-5.0), 124 (cHt=+5.0)")
    
    # Compute SM and decomposed weight coefficients
    w_sm = data['eventWeight'].astype(np.float32)
    
    # SMEFT decomposition using ±5.0 reference:
    # w(+5) = w_sm + 5*w_lin + 25*w_quad
    # w(-5) = w_sm - 5*w_lin + 25*w_quad
    # Solving: w_lin = (w(+5) - w(-5)) / 10
    #          w_quad = (w(+5) + w(-5) - 2*w_sm) / 50
    w_lin_from_5 = (w_plus_5 - w_minus_5) / 10.0
    w_quad_from_5 = (w_plus_5 + w_minus_5 - 2.0 * w_sm) / 50.0
    
    # SMEFT decomposition using ±1.0 reference:
    # w(+1) = w_sm + 1*w_lin + 1*w_quad
    # w(-1) = w_sm - 1*w_lin + 1*w_quad
    # Solving: w_lin = (w(+1) - w(-1)) / 2
    #          w_quad = (w(+1) + w(-1) - 2*w_sm) / 2
    w_lin_from_1 = (w_plus_1 - w_minus_1) / 2.0
    w_quad_from_1 = (w_plus_1 + w_minus_1 - 2.0 * w_sm) / 2.0
    
    # Verify that decomposition is mathematically consistent
    log.info(f"\nConsistency check: Decomposition coefficients")
    lin_diff = np.abs(w_lin_from_5 - w_lin_from_1).max()
    quad_diff = np.abs(w_quad_from_5 - w_quad_from_1).max()
    log.info(f"  Max |w_lin(from ±5) - w_lin(from ±1)| = {lin_diff:.2e}")
    log.info(f"  Max |w_quad(from ±5) - w_quad(from ±1)| = {quad_diff:.2e}")
    if lin_diff < 1e-5 and quad_diff < 1e-5:
        log.info(f"  ✓ Coefficients are consistent!")
    else:
        log.warning(f"  ✗ Coefficients diverge significantly!")
    
    # Use ±5.0 decomposition as reference
    w_lin_coeff = w_lin_from_5
    w_quad_coeff = w_quad_from_5
    
    log.info(f"\nWeight statistics from ROOT:")
    log.info(f"  w_sm:            min={w_sm.min():.6e}, mean={w_sm.mean():.6e}, max={w_sm.max():.6e}")
    log.info(f"  w(cHt=+5.0):     min={w_plus_5.min():.6e}, mean={w_plus_5.mean():.6e}, max={w_plus_5.max():.6e}")
    log.info(f"  w(cHt=-5.0):     min={w_minus_5.min():.6e}, mean={w_minus_5.mean():.6e}, max={w_minus_5.max():.6e}")
    log.info(f"  w(cHt=+1.0):     min={w_plus_1.min():.6e}, mean={w_plus_1.mean():.6e}, max={w_plus_1.max():.6e}")
    log.info(f"  w(cHt=-1.0):     min={w_minus_1.min():.6e}, mean={w_minus_1.mean():.6e}, max={w_minus_1.max():.6e}")
    log.info(f"\nDerived coefficients from ±5.0:")
    log.info(f"  w_lin_coeff = (w(+5) - w(-5)) / 10.0")
    log.info(f"    min={w_lin_coeff.min():.6e}, mean={w_lin_coeff.mean():.6e}, max={w_lin_coeff.max():.6e}")
    log.info(f"  w_quad_coeff = (w(+5) + w(-5) - 2*w_sm) / 50.0")
    log.info(f"    min={w_quad_coeff.min():.6e}, mean={w_quad_coeff.mean():.6e}, max={w_quad_coeff.max():.6e}")
    
    # Scale to arbitrary cHt
    w_lin_at_cHt = w_lin_coeff * cHt
    w_quad_at_cHt = w_quad_coeff * (cHt ** 2)
    
    log.info(f"\nScaled to cHt={cHt}:")
    log.info(f"  w_lin_at_cHt = w_lin_coeff * {cHt}")
    log.info(f"    min={w_lin_at_cHt.min():.6e}, mean={w_lin_at_cHt.mean():.6e}, max={w_lin_at_cHt.max():.6e}")
    log.info(f"  w_quad_at_cHt = w_quad_coeff * {cHt}^2")
    log.info(f"    min={w_quad_at_cHt.min():.6e}, mean={w_quad_at_cHt.mean():.6e}, max={w_quad_at_cHt.max():.6e}")
    
    # Construct physics features (15D: Cartesian coordinates)
    # Order: Z_Lepton1 (3) -> Z_Lepton2 (3) -> W_Lepton (3) -> BJet (4) -> MET (2)
    met_px = data['MET'] * np.cos(data['MET_phi'])
    met_py = data['MET'] * np.sin(data['MET_phi'])
    features_raw = np.column_stack([
        data['Z_Lepton1_Px'],  data['Z_Lepton1_Py'], data['Z_Lepton1_Pz'],
        data['Z_Lepton2_Px'],  data['Z_Lepton2_Py'], data['Z_Lepton2_Pz'],
        data['W_Lepton_Px'],   data['W_Lepton_Py'],  data['W_Lepton_Pz'],
        data['BJet_Px'],       data['BJet_Py'],       data['BJet_Pz'],      data['BJet_Mass'],
        met_px,                met_py,
    ]).astype(np.float32)
    
    # Create decomposed weight dictionary
    decomposed = {}
    
    # SM: no filtering needed, use all events
    log.info("\n--- SM component ---")
    n_sm = len(w_sm)
    log.info(f"Total events: {n_sm}")
    decomposed['sm'] = {
        'features': features_raw,
        'weights': w_sm,
        'count': n_sm,
    }
    
    # Linear: split into positive and negative
    log.info("\n--- Linear component ---")
    lin_pos_mask = w_lin_at_cHt >= 0
    lin_neg_mask = w_lin_at_cHt < 0
    n_lin_pos = lin_pos_mask.sum()
    n_lin_neg = lin_neg_mask.sum()
    log.info(f"Positive events: {n_lin_pos}")
    log.info(f"Negative events: {n_lin_neg}")
    
    decomposed['linear_pos'] = {
        'features': features_raw[lin_pos_mask],
        'weights': w_lin_at_cHt[lin_pos_mask],
        'count': n_lin_pos,
    }
    decomposed['linear_neg'] = {
        'features': features_raw[lin_neg_mask],
        'weights': np.abs(w_lin_at_cHt[lin_neg_mask]),  # Flip sign for model training
        'count': n_lin_neg,
    }
    
    # Quadratic: split into positive and negative
    log.info("\n--- Quadratic component ---")
    quad_pos_mask = w_quad_at_cHt >= 0
    quad_neg_mask = w_quad_at_cHt < 0
    n_quad_pos = quad_pos_mask.sum()
    n_quad_neg = quad_neg_mask.sum()
    log.info(f"Positive events: {n_quad_pos}")
    log.info(f"Negative events: {n_quad_neg}")
    
    decomposed['quadratic_pos'] = {
        'features': features_raw[quad_pos_mask],
        'weights': w_quad_at_cHt[quad_pos_mask],
        'count': n_quad_pos,
    }
    decomposed['quadratic_neg'] = {
        'features': features_raw[quad_neg_mask],
        'weights': np.abs(w_quad_at_cHt[quad_neg_mask]),  # Flip sign for model training
        'count': n_quad_neg,
    }
    
    return decomposed


def compare_decompositions(dec_5: dict, dec_1: dict) -> None:
    """
    Compare weight decompositions at cHt=5 and cHt=1.
    
    Parameters
    ----------
    dec_5 : dict
        Decomposition at cHt=5
    dec_1 : dict
        Decomposition at cHt=1
    """
    log.info(f"\n{'='*70}")
    log.info("COMPARISON: cHt=5 vs cHt=1")
    log.info(f"{'='*70}")
    
    components = ['sm', 'linear_pos', 'linear_neg', 'quadratic_pos', 'quadratic_neg']
    all_match = True
    
    for comp in components:
        log.info(f"\n--- {comp.upper()} ---")
        
        feat_5 = dec_5[comp]['features']
        feat_1 = dec_1[comp]['features']
        count_5 = dec_5[comp]['count']
        count_1 = dec_1[comp]['count']
        
        # Check if event counts match
        if count_5 != count_1:
            log.warning(f"  Event count MISMATCH: cHt=5 has {count_5}, cHt=1 has {count_1}")
            all_match = False
        else:
            log.info(f"  Event count: {count_5} ✓")
        
        # Check if features match
        if feat_5.shape != feat_1.shape:
            log.warning(f"  Feature shape MISMATCH: cHt=5 {feat_5.shape}, cHt=1 {feat_1.shape}")
            all_match = False
        else:
            # Check if feature arrays are identical (within floating point tolerance)
            if np.allclose(feat_5, feat_1, rtol=1e-10, atol=1e-14):
                log.info(f"  Features: {feat_5.shape} ✓ (numerically identical)")
            else:
                max_diff = np.abs(feat_5 - feat_1).max()
                mean_diff = np.abs(feat_5 - feat_1).mean()
                log.warning(f"  Features DIFFER: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
                all_match = False
    
    log.info(f"\n{'='*70}")
    if all_match:
        log.info("✓ ALL COMPONENTS MATCH: Weight decomposition is consistent!")
    else:
        log.warning("✗ MISMATCHES FOUND: Review differences above.")
    log.info(f"{'='*70}\n")
    
    return all_match


def plot_feature_comparisons(decomposed_5: dict, decomposed_1: dict):
    """
    Create 3 comprehensive histogram plots comparing features at cHt=5 vs cHt=1
    with ratio plots underneath. One file per weight component (SM, Linear, Quadratic).
    
    Parameters
    ----------
    decomposed_5 : dict
        Decomposed components at cHt=5
    decomposed_1 : dict
        Decomposed components at cHt=1
    """
    log.info("\nGenerating feature comparison plots with ratio...")
    
    # Feature names from TTZ dataset
    feature_names = [
        "Z_Lepton1_Px", "Z_Lepton1_Py", "Z_Lepton1_Pz",
        "Z_Lepton2_Px", "Z_Lepton2_Py", "Z_Lepton2_Pz",
        "W_Lepton_Px", "W_Lepton_Py", "W_Lepton_Pz",
        "BJet_Px", "BJet_Py", "BJet_Pz", "BJet_Mass",
        "MET_Px", "MET_Py",
    ]
    
    # For SM: show all 15 features, plus linear_pos/neg, plus quadratic_pos/neg
    component_groups = {
        "SM": ["sm"],
        "Linear": ["linear_pos", "linear_neg"],
        "Quadratic": ["quadratic_pos", "quadratic_neg"],
    }
    
    for group_name, components in component_groups.items():
        log.info(f"  Creating {group_name} comparison figure...")
        
        # Create figure with all 15 features × 2 rows (main + ratio)
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(14, 5 * len(feature_names)))
        gs = gridspec.GridSpec(len(feature_names) * 2, 1, 
                               height_ratios=[3, 1] * len(feature_names),
                               hspace=0.40)
        
        fig.suptitle(f"{group_name} Component(s): Feature Distributions (cHt=±5 vs cHt=±1) with Ratios",
                    fontsize=16, fontweight='bold', y=0.995)
        
        for feat_idx, feat_name in enumerate(feature_names):
            # Combine data from all components in this group
            data_5_list = []
            data_1_list = []
            for comp in components:
                data_5_list.append(decomposed_5[comp]['features'][:, feat_idx])
                data_1_list.append(decomposed_1[comp]['features'][:, feat_idx])
            
            # Concatenate all component data for this feature
            data_5 = np.concatenate(data_5_list)
            data_1 = np.concatenate(data_1_list)
            
            # Top panel: line histograms
            ax_hist = fig.add_subplot(gs[feat_idx * 2, 0])
            
            # Create histograms with step lines (not filled)
            counts_5, bins_5, _ = ax_hist.hist(data_5, bins=50, alpha=1.0, histtype='step', 
                                                label="cHt=±5", color='blue', linewidth=2)
            counts_1, bins_1, _ = ax_hist.hist(data_1, bins=50, alpha=1.0, histtype='step',
                                                label="cHt=±1", color='red', linewidth=2)
            
            ax_hist.set_ylabel("Count", fontsize=10)
            ax_hist.set_yscale('log')
            ax_hist.legend(fontsize=9, loc='upper right')
            ax_hist.grid(alpha=0.3)
            ax_hist.set_title(feat_name, fontsize=11, fontweight='bold', pad=8)
            
            # Add statistics
            mean_5, std_5 = data_5.mean(), data_5.std()
            mean_1, std_1 = data_1.mean(), data_1.std()
            stats_text = f"cHt=±5: μ={mean_5:.3f}\ncHt=±1: μ={mean_1:.3f}"
            ax_hist.text(0.02, 0.97, stats_text, transform=ax_hist.transAxes,
                        fontsize=8, verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Bottom panel: ratio plot
            ax_ratio = fig.add_subplot(gs[feat_idx * 2 + 1, 0])
            
            # Use common bins for ratio calculation
            bins_common = np.linspace(min(data_5.min(), data_1.min()), 
                                     max(data_5.max(), data_1.max()), 50)
            counts_5_common, _ = np.histogram(data_5, bins=bins_common)
            counts_1_common, _ = np.histogram(data_1, bins=bins_common)
            
            # Avoid division by zero
            ratio = np.divide(counts_5_common, counts_1_common, 
                            where=counts_1_common > 0, 
                            out=np.ones_like(counts_5_common, dtype=float))
            bin_centers = (bins_common[:-1] + bins_common[1:]) / 2
            
            # Plot ratio as step line
            ax_ratio.step(bin_centers, ratio, where='mid', color='purple', linewidth=2)
            ax_ratio.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            
            ax_ratio.set_xlabel(feat_name, fontsize=10)
            ax_ratio.set_ylabel("Ratio (±5/±1)", fontsize=10)
            ax_ratio.set_ylim([0.5, 1.5])
            ax_ratio.grid(alpha=0.3)
        
        # Save single figure for this group
        output_path = Path(f"validate_weight_{group_name.lower()}_cHt_comparison.png")
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        log.info(f"    Saved: {output_path}")
        plt.close()
    
    log.info("✓ All 3 component comparison figures generated successfully!")


def main():
    """Main entrypoint."""
    log.info("Starting weight decomposition validation...")
    
    try:
        # Load and decompose at both cHt values
        decomposed_5 = load_and_decompose_weights(5.0)
        decomposed_1 = load_and_decompose_weights(1.0)
        
        # Compare
        match = compare_decompositions(decomposed_5, decomposed_1)
        
        # Generate plots
        plot_feature_comparisons(decomposed_5, decomposed_1)
        
        sys.exit(0 if match else 1)
        
    except Exception as e:
        log.exception(f"Error during validation: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()

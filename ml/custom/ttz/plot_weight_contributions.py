#!/usr/bin/env python
"""
Plot ttZ data with SM, linear, and quadratic SMEFT weight contributions.

This script loads the ttZ data directly from the ROOT file and creates
histograms showing the differences between SM, linear, and quadratic 
weight contributions for all features. The sum of weights for each 
contribution is displayed at the top of each plot.

Usage:
    python ml/custom/ttz/plot_weight_contributions.py
    python ml/custom/ttz/plot_weight_contributions.py --output plots/weights.png
    python ml/custom/ttz/plot_weight_contributions.py --bins 40
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT_FILE = "/project/atlas/users/kdevries/EventLoop/ttZ_for_Melle_tree.root"
VARIABLES_JSON = PROJECT_ROOT / "ml/data/ttz/variables.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "ml/custom/ttz/figures/weight_contributions.png"

# Weight indices in smeft_weights array
IDX_CHT_MINUS5 = 121  # cHt = -5.0
IDX_CHT_PLUS5 = 123   # cHt = +5.0


def load_ttz_data(root_file_path):
    """
    Load ttZ data directly from ROOT file with all weight information.
    Includes both the 15 base kinematic features and derived higher-order
    features (Z, W boson, top kinematics) stored directly in the ROOT tree.

    Returns
    -------
    features : np.ndarray
        Array of shape (n_events, n_features) with all kinematic features
    feature_names : list of str
        Names of all features
    weights_sm : np.ndarray
        SM weights (eventWeight)
    weights_linear : np.ndarray
        Linear SMEFT term weights
    weights_quadratic : np.ndarray
        Quadratic SMEFT term weights
    """
    log.info(f"Loading data from {root_file_path}")

    # Base input features (Cartesian coordinates, same as training)
    base_branches = [
        'Z_Lepton1_Px', 'Z_Lepton1_Py', 'Z_Lepton1_Pz',
        'Z_Lepton2_Px', 'Z_Lepton2_Py', 'Z_Lepton2_Pz',
        'W_Lepton_Px',  'W_Lepton_Py',  'W_Lepton_Pz',
        'BJet_Px',      'BJet_Py',      'BJet_Pz',      'BJet_Mass',
        'MET_Px',       'MET_Py',
    ]

    # Higher-order / derived features stored in the ROOT tree
    derived_branches = [
        'Z_pt',      'Z_eta',    'Z_mass',   'Z_deltaR',
        'W_boson_pt', 'W_boson_eta',
        'top_pt',    'top_mass', 'top_eta',  'top_lep_rapidity',
        'Z_over_top_pt',
    ]

    feature_branches = base_branches + derived_branches

    root_base_branches = [
        'Z_Lepton1_Px', 'Z_Lepton1_Py', 'Z_Lepton1_Pz',
        'Z_Lepton2_Px', 'Z_Lepton2_Py', 'Z_Lepton2_Pz',
        'W_Lepton_Px',  'W_Lepton_Py',  'W_Lepton_Pz',
        'BJet_Px',      'BJet_Py',      'BJet_Pz',      'BJet_Mass',
        'MET',          'MET_phi',
    ]

    # Weight and selection branches
    weight_branches = ['eventWeight', 'smeft_weights']
    selection_branches = ['Jet_Pt']

    all_branches = root_base_branches + derived_branches + weight_branches + selection_branches

    with uproot.open(root_file_path) as root_file:
        tree = root_file["Events"]
        data = tree.arrays(all_branches, library="np")

    n_events = len(data['eventWeight'])
    log.info(f"Loaded {n_events} events")

    # ---- Filter: keep only events with exactly 3 jets ----
    n_jets = np.array([len(j) for j in data['Jet_Pt']])
    mask = n_jets == 3
    n_selected = mask.sum()
    log.info(f"Jet multiplicity filter (n_jets == 3): {n_selected} / {n_events} events kept ({100*n_selected/n_events:.1f}%)")

    log.info(f"Base features   : {len(base_branches)}")
    log.info(f"Derived features: {len(derived_branches)}")

    # Stack all features into one array, applying the jet filter.
    # MET_Px/MET_Py are derived from ROOT branches (MET, MET_phi).
    met_px = data['MET'][mask] * np.cos(data['MET_phi'][mask])
    met_py = data['MET'][mask] * np.sin(data['MET_phi'][mask])

    feature_columns = [
        data['Z_Lepton1_Px'][mask], data['Z_Lepton1_Py'][mask], data['Z_Lepton1_Pz'][mask],
        data['Z_Lepton2_Px'][mask], data['Z_Lepton2_Py'][mask], data['Z_Lepton2_Pz'][mask],
        data['W_Lepton_Px'][mask], data['W_Lepton_Py'][mask], data['W_Lepton_Pz'][mask],
        data['BJet_Px'][mask], data['BJet_Py'][mask], data['BJet_Pz'][mask], data['BJet_Mass'][mask],
        met_px, met_py,
    ]
    # Append derived branches after the 15 base features in their original order.
    feature_columns.extend([data[branch][mask] for branch in derived_branches])
    features = np.column_stack(feature_columns).astype(np.float32)

    # Extract weights (with jet filter applied)
    w_sm = data['eventWeight'][mask]
    smeft_weights = np.stack(data['smeft_weights'][mask])
    w_plus = smeft_weights[:, IDX_CHT_PLUS5]   # cHt = +5.0
    w_minus = smeft_weights[:, IDX_CHT_MINUS5]  # cHt = -5.0
    
    # Compute weight contributions:
    # w(cHt) = w_sm + cHt * w_lin + cHt^2 * w_quad
    # At cHt = ±5.0:
    #   w(+5) = w_sm + 5*w_lin + 25*w_quad
    #   w(-5) = w_sm - 5*w_lin + 25*w_quad
    # Solving for w_lin and w_quad:
    weights_sm = w_sm
    weights_linear = (w_plus - w_minus) / 2.0
    weights_quadratic = (w_plus + w_minus - 2.0 * w_sm) / 2.0
    
    # Decompose into positive and negative components
    # Linear: split based on sign (show |weight| for negative)
    weights_linear_pos = np.where(weights_linear >= 0, weights_linear, 0)
    weights_linear_neg = np.where(weights_linear < 0, -weights_linear, 0)
    
    # Quadratic: split based on sign (show |weight| for negative)
    weights_quadratic_pos = np.where(weights_quadratic >= 0, weights_quadratic, 0)
    weights_quadratic_neg = np.where(weights_quadratic < 0, -weights_quadratic, 0)
    
    sum_weights_linear = (w_plus.sum() - w_minus.sum()) / 2.0
    sum_weights_quadratic = (w_plus.sum() + w_minus.sum() - 2.0 * w_sm.sum()) / 2.0
    log.info(f"Features shape: {features.shape}")
    log.info(f"SM weights sum: {weights_sm.sum():.2f}")
    log.info(f"Linear pos weights sum: {weights_linear_pos.sum():.2f}")
    log.info(f"Linear neg weights sum: {weights_linear_neg.sum():.2f}")
    log.info(f"Linear total sum: {(weights_linear_pos.sum() - weights_linear_neg.sum()):.2f}")
    log.info(f"Quadratic pos weights sum: {weights_quadratic_pos.sum():.2f}")
    log.info(f"Quadratic neg weights sum: {weights_quadratic_neg.sum():.2f}")
    log.info(f"Quadratic total sum: {(weights_quadratic_pos.sum() + weights_quadratic_neg.sum()):.2f}")
    log.info(f"Check linear weight sum from w_plus/w_minus: {sum_weights_linear:.2f}")
    log.info(f"Check quadratic weight sum from w_plus/w_minus: {sum_weights_quadratic:.2f}")
    
    return features, feature_branches, weights_sm, weights_linear_pos, weights_linear_neg, weights_quadratic_pos, weights_quadratic_neg


def get_range_limits(data, percentile_range=0.1):
    """Calculate range limits excluding extreme outliers."""
    lower = np.percentile(data, percentile_range)
    upper = np.percentile(data, 100 - percentile_range)
    data_range = upper - lower
    # Add padding
    lower = lower - 0.1 * data_range
    upper = upper + 0.1 * data_range
    return lower, upper


def plot_weight_contributions(features, feature_names, weights_sm, weights_linear_pos, 
                              weights_linear_neg, weights_quadratic_pos, weights_quadratic_neg, 
                              output_path, bins=50, n_cols=3):
    """
    Create comparison plots showing SM, linear (+/-), and quadratic (+/-) weight contributions.
    
    Parameters
    ----------
    features : np.ndarray
        Feature data array (n_events, n_features)
    feature_names : list of str
        Names of features
    weights_sm : np.ndarray
        SM weights
    weights_linear_pos : np.ndarray
        Linear SMEFT weights (positive region)
    weights_linear_neg : np.ndarray
        Linear SMEFT weights (negative region, as absolute values)
    weights_quadratic_pos : np.ndarray
        Quadratic SMEFT weights (positive region)
    weights_quadratic_neg : np.ndarray
        Quadratic SMEFT weights (negative region, as absolute values)
    output_path : str or Path
        Path to save the figure
    bins : int
        Number of histogram bins
    n_cols : int
        Number of columns in plot grid
    """
    log.info("Creating weight contribution plots with pos/neg decomposition...")
    
    n_features = len(feature_names)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Calculate total weight sums for display
    sum_sm = weights_sm.sum()
    sum_linear_pos = weights_linear_pos.sum()
    sum_linear_neg = weights_linear_neg.sum()
    sum_quadratic_pos = weights_quadratic_pos.sum()
    sum_quadratic_neg = weights_quadratic_neg.sum()
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    
    fig.suptitle(
        f'ttZ Feature Distributions: SMEFT Weight Contributions (pos/neg decomposed)\n'
        f'SM: {sum_sm:.3f}  |  Lin(+): {sum_linear_pos:.3f}  |  Lin(−): {sum_linear_neg:.3f}  |  Quad(+): {sum_quadratic_pos:.4f}  |  Quad(−): {sum_quadratic_neg:.4f}',
        fontsize=12, fontweight='bold', y=0.995
    )
    
    for i, feature in enumerate(feature_names):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Get feature data
        feature_data = features[:, i]
        
        # Calculate histogram range
        x_min, x_max = get_range_limits(feature_data)
        
        # Create weighted histograms for each contribution
        hist_sm, bin_edges = np.histogram(
            feature_data, bins=bins, range=(x_min, x_max), 
            density=True, weights=weights_sm
        )
        hist_linear_pos, _ = np.histogram(
            feature_data, bins=bin_edges, 
            density=True, weights=weights_linear_pos
        )
        hist_linear_neg, _ = np.histogram(
            feature_data, bins=bin_edges, 
            density=True, weights=weights_linear_neg
        )
        hist_quadratic_pos, _ = np.histogram(
            feature_data, bins=bin_edges, 
            density=True, weights=weights_quadratic_pos
        )
        hist_quadratic_neg, _ = np.histogram(
            feature_data, bins=bin_edges, 
            density=True, weights=weights_quadratic_neg
        )
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plot all five contributions
        ax.step(bin_centers, hist_sm, color='blue', label='SM', 
                where='mid', lw=2, alpha=0.8)
        ax.step(bin_centers, hist_linear_pos, color='red', label='Linear (+)', 
                where='mid', lw=2, alpha=0.8)
        ax.step(bin_centers, hist_linear_neg, color='red', linestyle='--', label='Linear (−)', 
                where='mid', lw=2, alpha=0.8)
        ax.step(bin_centers, hist_quadratic_pos, color='green', label='Quadratic (+)', 
                where='mid', lw=2, alpha=0.8)
        ax.step(bin_centers, hist_quadratic_neg, color='green', linestyle='--', label='Quadratic (−)', 
                where='mid', lw=2, alpha=0.8)
        
        # Formatting
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('Normalized density', fontsize=10)
        ax.set_xlim(x_min, x_max)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(fontsize=9, loc='best')
        ax.tick_params(labelsize=9)
    
    # Hide empty subplots
    for i in range(n_features, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    log.info(f"Saved plot to {output_path}")
    
    # Also create log scale version
    log_output_path = output_path.parent / (output_path.stem + "_log" + output_path.suffix)
    
    # Update all axes to log scale
    for i in range(n_features):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        ax.set_yscale('log')
        ax.set_ylabel('Normalized density (log)', fontsize=10)
    
    plt.savefig(log_output_path, dpi=150, bbox_inches='tight')
    log.info(f"Saved log-scale plot to {log_output_path}")
    
    plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Plot ttZ feature distributions split by SMEFT weight contributions"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output path for plot (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins (default: 50)"
    )
    parser.add_argument(
        "--n-cols",
        type=int,
        default=4,
        help="Number of columns in plot grid (default: 4)"
    )
    args = parser.parse_args()
    
    log.info("=" * 60)
    log.info("ttZ SMEFT Weight Contribution Plotter (pos/neg decomposed)")
    log.info("=" * 60)
    
    # Load data
    features, feature_names, weights_sm, weights_linear_pos, weights_linear_neg, \
        weights_quadratic_pos, weights_quadratic_neg = \
        load_ttz_data(DATA_ROOT_FILE)
    
    # Create plots
    plot_weight_contributions(
        features, feature_names,
        weights_sm, weights_linear_pos, weights_linear_neg,
        weights_quadratic_pos, weights_quadratic_neg,
        args.output, args.bins, args.n_cols
    )
    
    log.info("=" * 60)
    log.info("Done!")
    log.info("=" * 60)


if __name__ == "__main__":
    main()

"""Plotting utilities for comparing SM and SMEFT models and data."""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


def get_range_limits(data, percentile_range=0.1):
    """Calculate range limits excluding extreme outliers."""
    lower = np.percentile(data, percentile_range)
    upper = np.percentile(data, 100 - percentile_range)
    data_range = upper - lower
    # Add padding
    lower = lower - 0.1 * data_range
    upper = upper + 0.1 * data_range
    return lower, upper


def plot_sm_vs_smeft_comparison(sm_real_data, sm_gen_data, smeft_real_data, smeft_gen_data,
                                sm_real_weights, smeft_real_weights,
                                selected_features, selection, variables, output_path, 
                                bins=50, n_cols=3, use_log_scale=False):
    """
    Plot comparison of SM and SMEFT for both real and generated data.
    
    Each subplot shows 4 histograms:
    - SM real data (blue solid) - weighted by SM eventWeight
    - SM generated data (blue dotted) - unweighted
    - SMEFT real data (red solid) - weighted by cHt=5.0
    - SMEFT generated data (red dotted) - unweighted
    
    Parameters
    ----------
    sm_real_data : np.ndarray
        SM weighted real data
    sm_gen_data : np.ndarray
        Generated samples from SM model
    smeft_real_data : np.ndarray
        SMEFT (cHt=5.0) weighted real data
    smeft_gen_data : np.ndarray
        Generated samples from SMEFT model
    sm_real_weights : np.ndarray
        SM event weights for real data
    smeft_real_weights : np.ndarray
        SMEFT event weights for real data
    selected_features : list
        List of feature names
    selection : pd.DataFrame
        Feature selection info
    variables : dict
        Variables configuration
    output_path : str
        Path to save figure
    bins : int
        Number of histogram bins
    n_cols : int
        Number of columns in plot grid
    use_log_scale : bool
        Whether to use log scale for y-axis
    """
    # Calculate grid dimensions
    n_features = len(selected_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Create figure
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    outer_gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.3)
    
    for idx, feature in enumerate(selected_features):
        row = idx // n_cols
        col = idx % n_cols
        
        # Create subplot with main plot and ratio plot
        inner_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[row, col], 
                                          hspace=0.05, height_ratios=[3, 1])
        ax_main = fig.add_subplot(inner_gs[0])
        ax_ratio = fig.add_subplot(inner_gs[1], sharex=ax_main)
        
        # Get feature data
        feature_idx = selected_features.index(feature)
        sm_real_feature = sm_real_data[:, feature_idx]
        sm_gen_feature = sm_gen_data[:, feature_idx]
        smeft_real_feature = smeft_real_data[:, feature_idx]
        smeft_gen_feature = smeft_gen_data[:, feature_idx]
        
        # Calculate histogram range using all data
        x_min = min(get_range_limits(sm_real_feature)[0],
                   get_range_limits(sm_gen_feature)[0],
                   get_range_limits(smeft_real_feature)[0],
                   get_range_limits(smeft_gen_feature)[0])
        x_max = max(get_range_limits(sm_real_feature)[1],
                   get_range_limits(sm_gen_feature)[1],
                   get_range_limits(smeft_real_feature)[1],
                   get_range_limits(smeft_gen_feature)[1])
        
        # Create histograms - apply weights to real data
        hist_sm_real, bin_edges = np.histogram(sm_real_feature, bins=bins, 
                                               range=(x_min, x_max), density=True,
                                               weights=sm_real_weights)
        hist_sm_gen, _ = np.histogram(sm_gen_feature, bins=bin_edges, density=True)
        hist_smeft_real, _ = np.histogram(smeft_real_feature, bins=bin_edges, density=True,
                                          weights=smeft_real_weights)
        hist_smeft_gen, _ = np.histogram(smeft_gen_feature, bins=bin_edges, density=True)
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plot all four distributions
        ax_main.step(bin_centers, hist_sm_real, color='blue', label='SM Real', 
                    where='mid', lw=1.0, linestyle='-')
        ax_main.step(bin_centers, hist_sm_gen, color='blue', label='SM Generated', 
                    where='mid', lw=1.0, linestyle=':', alpha=0.9)
        ax_main.step(bin_centers, hist_smeft_real, color='red', label='cHt=5.0 Real', 
                    where='mid', lw=1.0, linestyle='-')
        ax_main.step(bin_centers, hist_smeft_gen, color='red', label='cHt=5.0 Generated', 
                    where='mid', lw=1.0, linestyle=':', alpha=0.9)
        
        # Formatting main plot
        if use_log_scale:
            ax_main.set_yscale('log')
            ax_main.set_ylabel('Density (log)', fontsize=10)
        else:
            y_max = max(np.max(hist_sm_real), np.max(hist_sm_gen),
                       np.max(hist_smeft_real), np.max(hist_smeft_gen)) * 1.1
            ax_main.set_ylabel('Density', fontsize=10)
            ax_main.set_ylim(0, y_max)
        
        ax_main.set_xlim(x_min, x_max)
        ax_main.grid(True, linestyle='--', alpha=0.5)
        ax_main.legend(fontsize=7, loc='best')
        ax_main.set_title(feature, fontsize=10, pad=5)
        ax_main.tick_params(labelbottom=False, labelsize=9)
        
        # Ratio plot: All distributions relative to SM real data
        ratio_sm_gen = np.divide(hist_sm_gen, hist_sm_real, 
                                out=np.ones_like(hist_sm_gen), where=hist_sm_real!=0)
        ratio_smeft_real = np.divide(hist_smeft_real, hist_sm_real, 
                                     out=np.ones_like(hist_smeft_real), where=hist_sm_real!=0)
        ratio_smeft_gen = np.divide(hist_smeft_gen, hist_sm_real, 
                                    out=np.ones_like(hist_smeft_gen), where=hist_sm_real!=0)
        
        ax_ratio.step(bin_centers, ratio_sm_gen, color='blue', where='mid', lw=1.0, 
                     label='SM Gen', linestyle=':', alpha=0.9)
        ax_ratio.step(bin_centers, ratio_smeft_real, color='red', where='mid', lw=1.0, 
                     label='cHt=5.0 Real', linestyle='-')
        ax_ratio.step(bin_centers, ratio_smeft_gen, color='red', where='mid', lw=1.0, 
                     label='cHt=5.0 Gen', linestyle=':', alpha=0.9)
        ax_ratio.axhline(y=1, color='gray', linestyle='--', lw=1, alpha=0.7, label='SM Real')
        ax_ratio.set_xlabel(feature, fontsize=9)
        ax_ratio.set_ylabel('Ratio to SM Real', fontsize=8)
        ax_ratio.set_xlim(x_min, x_max)
        ax_ratio.set_ylim(0.5, 1.5)
        ax_ratio.grid(True, linestyle='--', alpha=0.5)
        ax_ratio.tick_params(labelsize=8)
        ax_ratio.legend(fontsize=6, loc='best')
    
    # Determine output filename
    if use_log_scale:
        save_path = output_path.replace('.png', '_log.png')
        scale_type = "log scale"
    else:
        save_path = output_path
        scale_type = "normal scale"
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved SM vs SMEFT comparison plot ({scale_type}) to: {save_path}")


def create_both_scale_plots(sm_real_data, sm_gen_data, smeft_real_data, smeft_gen_data,
                           sm_real_weights, smeft_real_weights,
                           selected_features, selection, variables, output_path, 
                           bins=50, n_cols=3):
    """Create both normal and log scale versions of the SM vs SMEFT comparison."""
    for use_log_scale in [False, True]:
        plot_sm_vs_smeft_comparison(
            sm_real_data, sm_gen_data, smeft_real_data, smeft_gen_data,
            sm_real_weights, smeft_real_weights,
            selected_features, selection, variables, output_path,
            bins, n_cols, use_log_scale
        )

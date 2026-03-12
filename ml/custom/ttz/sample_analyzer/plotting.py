"""Plotting utilities for ttZ sample analysis."""

import os
import numpy as np
import matplotlib.pyplot as plt


def get_range_limits(data, percentile_range=0.1):
    """Calculate range limits excluding extreme outliers."""
    lower = np.percentile(data, percentile_range)
    upper = np.percentile(data, 100 - percentile_range)
    data_range = upper - lower
    # Add padding
    lower = lower - 0.1 * data_range
    upper = upper + 0.1 * data_range
    return lower, upper


def plot_feature_comparison(real_data, generated_data, selected_features, selection, 
                           variables, output_path, bins=50, n_cols=3, real_weights=None, weight_type=None):
    """
    Plot comparison of all features between real and generated data with ratio plots.
    Creates both normal and log scale versions.
    
    Parameters
    ----------
    real_data : np.ndarray
        Real data array
    generated_data : np.ndarray
        Generated data array
    selected_features : list
        List of selected feature names
    selection : pd.DataFrame
        Feature selection DataFrame with type information
    variables : dict
        Variables configuration dictionary
    output_path : str
        Path to save figure (will create both normal and _log versions)
    bins : int
        Number of histogram bins
    n_cols : int
        Number of columns in plot grid
    real_weights : np.ndarray, optional
        Event weights for real data (e.g., cHt=5.0 SMEFT weights)
    weight_type : str, optional
        Weight type for labeling (e.g., 'linear_pos', 'quadratic_neg')
    """
    # Create both normal and log scale versions
    for use_log_scale in [False, True]:
        _create_feature_comparison_plot(
            real_data, generated_data, selected_features, selection,
            variables, output_path, bins, n_cols, use_log_scale, real_weights, weight_type
        )


def _create_feature_comparison_plot(real_data, generated_data, selected_features, selection, 
                                   variables, output_path, bins, n_cols, use_log_scale, real_weights=None, weight_type=None):
    """
    Plot comparison of all features between real and generated data with ratio plots.
    
    Parameters
    ----------
    real_data : np.ndarray
        Real data array
    generated_data : np.ndarray
        Generated data array
    selected_features : list
        List of selected feature names
    selection : pd.DataFrame
        Feature selection DataFrame with type information
    variables : dict
        Variables configuration dictionary
    output_path : str
        Path to save figure
    bins : int
        Number of histogram bins
    n_cols : int
        Number of columns in plot grid
    use_log_scale : bool
        Whether to use log scale
    real_weights : np.ndarray, optional
        Event weights for real data
    weight_type : str, optional
        Weight type for labeling
    """
    # Calculate grid dimensions
    n_features = len(selected_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Create figure with subplots
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    outer_gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.3)
    
    # Plot each feature
    for i, feature in enumerate(selected_features):
        row = i // n_cols
        col = i % n_cols
        
        # Create inner grid for this feature (main plot + ratio plot)
        inner_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[row, col], 
                                          height_ratios=[4, 1], hspace=0.08)
        
        ax_main = fig.add_subplot(inner_gs[0])
        ax_ratio = fig.add_subplot(inner_gs[1], sharex=ax_main)
        
        # Get data
        real_feature_data = real_data[:, i]
        generated_feature_data = generated_data[:, i]
        
        # Calculate histogram range
        x_min = min(get_range_limits(real_feature_data)[0], 
                   get_range_limits(generated_feature_data)[0])
        x_max = max(get_range_limits(real_feature_data)[1], 
                   get_range_limits(generated_feature_data)[1])
        
        # Create histograms
        # Apply weights to real data if provided (e.g., SMEFT cHt=5.0 weights)
        hist_real, bins_edges = np.histogram(real_feature_data, bins=bins, 
                                             range=(x_min, x_max), density=True,
                                             weights=real_weights)
        hist_gen, _ = np.histogram(generated_feature_data, bins=bins_edges, density=True)
        
        bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
        
        # Main plot — generate descriptive labels based on weight type
        if weight_type and weight_type != "unknown":
            # Create shortened weight type labels for compact display
            short_weight_map = {
                "sm": "sm",
                "linear": "lin",
                "linear_pos": "lin_pos",
                "linear_neg": "lin_neg",
                "quadratic": "qua",
                "quadratic_pos": "qua_pos",
                "quadratic_neg": "qua_neg",
                "full": "full",
            }
            short_type = short_weight_map.get(weight_type, weight_type)
            
            if weight_type.endswith('_neg'):
                label_real = f'MC data (weighted to {short_type}, |w|)'
            else:
                label_real = f'MC data (weighted to {short_type})'
        else:
            label_real = 'MC data (weighted)' if real_weights is not None else 'MC data (unweighted)'
        
        label_gen = 'Generated samples'
        ax_main.step(bin_centers, hist_real, color='blue', label=label_real, where='mid', lw=2)
        ax_main.step(bin_centers, hist_gen, color='red', label=label_gen, where='mid', lw=2)
        
        # Formatting main plot
        if use_log_scale:
            ax_main.set_yscale('log')
            ax_main.set_ylabel('Density (log scale)', fontsize=10)
        else:
            y_max = max(max(hist_real), max(hist_gen)) * 1.1
            ax_main.set_ylabel('Density', fontsize=10)
            ax_main.set_ylim(0, y_max)
        ax_main.set_xlim(x_min, x_max)
        ax_main.grid(True, linestyle='--', alpha=0.5)
        ax_main.legend(fontsize=8, loc='best')
        ax_main.set_title(feature, fontsize=10, pad=5)
        ax_main.tick_params(labelbottom=False, labelsize=9)
        
        # Ratio plot
        ratio = np.divide(hist_gen, hist_real, out=np.ones_like(hist_gen), where=hist_real!=0)
        
        ax_ratio.step(bin_centers, ratio, color='black', where='mid', lw=1.5)
        ax_ratio.axhline(y=1, color='gray', linestyle='--', lw=1, alpha=0.7)
        ax_ratio.set_xlabel(feature, fontsize=9)
        ax_ratio.set_ylabel('Gen/Real', fontsize=9)
        ax_ratio.set_xlim(x_min, x_max)
        ax_ratio.set_ylim(0.8, 1.2)
        ax_ratio.grid(True, linestyle='--', alpha=0.5)
        ax_ratio.tick_params(labelsize=8)
    
    # Determine output path based on scale type
    if use_log_scale:
        # Replace .png with _log.png
        save_path = output_path.replace('.png', '_log.png')
        scale_type = "log scale"
    else:
        save_path = output_path
        scale_type = "normal scale"
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved feature comparison plot ({scale_type}) to: {save_path}")

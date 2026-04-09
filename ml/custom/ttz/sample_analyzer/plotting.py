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


def _wrap_delta_phi(phi_a, phi_b):
    return np.arctan2(np.sin(phi_a - phi_b), np.cos(phi_a - phi_b))


def _weighted_hist(data, bins, value_range=None, weights=None):
    return np.histogram(data, bins=bins, range=value_range, density=True, weights=weights)


def _compute_binned_edges(real_vals, gen_vals, n_bins=4):
    pooled = np.concatenate([real_vals, gen_vals])
    edges = np.quantile(pooled, np.linspace(0.0, 1.0, n_bins + 1))
    # Guarantee strictly increasing edges to avoid empty-width bins.
    edges = np.maximum.accumulate(edges)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    return edges


def _plot_binned_delta_phi(real_dphi, gen_dphi, real_bin_var, gen_bin_var, real_weights, output_path, var_label):
    edges = _compute_binned_edges(real_bin_var, gen_bin_var, n_bins=4)
    phi_range = (-np.pi, np.pi)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    axes = axes.flatten()

    for i in range(4):
        lo, hi = edges[i], edges[i + 1]
        r_mask = (real_bin_var >= lo) & (real_bin_var < hi if i < 3 else real_bin_var <= hi)
        g_mask = (gen_bin_var >= lo) & (gen_bin_var < hi if i < 3 else gen_bin_var <= hi)

        if not np.any(r_mask) or not np.any(g_mask):
            axes[i].set_title(f"{var_label} in [{lo:.1f}, {hi:.1f}] (empty)")
            axes[i].grid(True, linestyle='--', alpha=0.5)
            continue

        rw = real_weights[r_mask] if real_weights is not None else None
        h_r, e = _weighted_hist(real_dphi[r_mask], bins=45, value_range=phi_range, weights=rw)
        h_g, _ = _weighted_hist(gen_dphi[g_mask], bins=e, weights=None)
        c = 0.5 * (e[:-1] + e[1:])

        axes[i].step(c, h_r, where='mid', color='blue', lw=1.8, label='Real')
        axes[i].step(c, h_g, where='mid', color='red', lw=1.8, label='Generated')
        axes[i].set_title(f"{var_label} in [{lo:.1f}, {hi:.1f}]")
        axes[i].grid(True, linestyle='--', alpha=0.5)
        axes[i].set_xlim(*phi_range)

    axes[0].legend(loc='best', fontsize=8)
    fig.supxlabel(r'$\Delta\phi_{\ell,\mathrm{MET}}$')
    fig.supylabel('Density')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved binned DeltaPhi diagnostics to: {output_path}")


def plot_correlation_diagnostics(real_data, generated_data, selected_features, output_path,
                                 real_weights=None, bins_1d=60, bins_2d=60):
    """Plot focused correlation diagnostics for Z and W systems.

    Produces:
      1) 1D overlays for Z lepton pair delta-eta and delta-phi, and W lepton-MET delta-phi.
      2) 2D maps for (delta-eta_ll, delta-phi_ll) in real and generated samples.
    """
    required = [
        'Z_Lepton1_Eta', 'Z_Lepton1_Phi',
        'Z_Lepton2_Eta', 'Z_Lepton2_Phi',
        'W_Lepton_Phi', 'W_Lepton_Pt',
        'MET', 'MET_Phi',
    ]
    missing = [name for name in required if name not in selected_features]
    if missing:
        print(f"Skipping correlation diagnostics; missing features: {missing}")
        return

    idx = {name: selected_features.index(name) for name in required}

    z1_eta_r = real_data[:, idx['Z_Lepton1_Eta']]
    z2_eta_r = real_data[:, idx['Z_Lepton2_Eta']]
    z1_phi_r = real_data[:, idx['Z_Lepton1_Phi']]
    z2_phi_r = real_data[:, idx['Z_Lepton2_Phi']]
    wl_phi_r = real_data[:, idx['W_Lepton_Phi']]
    wl_pt_r = real_data[:, idx['W_Lepton_Pt']]
    met_r = real_data[:, idx['MET']]
    met_phi_r = real_data[:, idx['MET_Phi']]

    z1_eta_g = generated_data[:, idx['Z_Lepton1_Eta']]
    z2_eta_g = generated_data[:, idx['Z_Lepton2_Eta']]
    z1_phi_g = generated_data[:, idx['Z_Lepton1_Phi']]
    z2_phi_g = generated_data[:, idx['Z_Lepton2_Phi']]
    wl_phi_g = generated_data[:, idx['W_Lepton_Phi']]
    wl_pt_g = generated_data[:, idx['W_Lepton_Pt']]
    met_g = generated_data[:, idx['MET']]
    met_phi_g = generated_data[:, idx['MET_Phi']]

    d_eta_ll_r = z1_eta_r - z2_eta_r
    d_eta_ll_g = z1_eta_g - z2_eta_g
    d_phi_ll_r = _wrap_delta_phi(z1_phi_r, z2_phi_r)
    d_phi_ll_g = _wrap_delta_phi(z1_phi_g, z2_phi_g)
    d_phi_wmet_r = _wrap_delta_phi(wl_phi_r, met_phi_r)
    d_phi_wmet_g = _wrap_delta_phi(wl_phi_g, met_phi_g)

    # W transverse observables used for focused diagnostics.
    w_px_r = wl_pt_r * np.cos(wl_phi_r) + met_r * np.cos(met_phi_r)
    w_py_r = wl_pt_r * np.sin(wl_phi_r) + met_r * np.sin(met_phi_r)
    w_pt_r = np.sqrt(w_px_r**2 + w_py_r**2)
    w_mt_r = np.sqrt(np.clip(2.0 * wl_pt_r * met_r * (1.0 - np.cos(d_phi_wmet_r)), 0.0, None))

    w_px_g = wl_pt_g * np.cos(wl_phi_g) + met_g * np.cos(met_phi_g)
    w_py_g = wl_pt_g * np.sin(wl_phi_g) + met_g * np.sin(met_phi_g)
    w_pt_g = np.sqrt(w_px_g**2 + w_py_g**2)
    w_mt_g = np.sqrt(np.clip(2.0 * wl_pt_g * met_g * (1.0 - np.cos(d_phi_wmet_g)), 0.0, None))

    # 1D correlation diagnostics
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=False)
    one_d_specs = [
        (d_eta_ll_r, d_eta_ll_g, r'$\Delta\eta_{\ell\ell}$', None),
        (d_phi_ll_r, d_phi_ll_g, r'$\Delta\phi_{\ell\ell}$', (-np.pi, np.pi)),
        (d_phi_wmet_r, d_phi_wmet_g, r'$\Delta\phi_{\ell,\mathrm{MET}}$', (-np.pi, np.pi)),
    ]

    for ax, (real_vals, gen_vals, label, value_range) in zip(axes, one_d_specs):
        if value_range is None:
            low = min(get_range_limits(real_vals)[0], get_range_limits(gen_vals)[0])
            high = max(get_range_limits(real_vals)[1], get_range_limits(gen_vals)[1])
            value_range = (low, high)

        hist_real, edges = _weighted_hist(real_vals, bins=bins_1d, value_range=value_range, weights=real_weights)
        hist_gen, _ = _weighted_hist(gen_vals, bins=edges, weights=None)
        centers = 0.5 * (edges[:-1] + edges[1:])

        ax.step(centers, hist_real, where='mid', color='blue', lw=2, label='Real')
        ax.step(centers, hist_gen, where='mid', color='red', lw=2, label='Generated')
        ax.set_ylabel('Density')
        ax.set_title(label)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='best', fontsize=9)

    axes[-1].set_xlabel('Value')
    fig.tight_layout()
    one_d_path = output_path.replace('.png', '_1d.png')
    fig.savefig(one_d_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved correlation diagnostics (1D) to: {one_d_path}")

    # 2D Z correlation maps
    eta_range = (
        min(np.percentile(d_eta_ll_r, 0.5), np.percentile(d_eta_ll_g, 0.5)),
        max(np.percentile(d_eta_ll_r, 99.5), np.percentile(d_eta_ll_g, 99.5)),
    )
    phi_range = (-np.pi, np.pi)

    H_real, xedges, yedges = np.histogram2d(
        d_eta_ll_r, d_phi_ll_r,
        bins=bins_2d,
        range=[eta_range, phi_range],
        density=True,
        weights=real_weights,
    )
    H_gen, _, _ = np.histogram2d(
        d_eta_ll_g, d_phi_ll_g,
        bins=[xedges, yedges],
        density=True,
    )

    fig2, ax2 = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True, sharey=True)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    vmax = max(np.nanmax(H_real), np.nanmax(H_gen))

    im0 = ax2[0].imshow(H_real.T, origin='lower', extent=extent, aspect='auto', vmin=0, vmax=vmax)
    ax2[0].set_title('Real: $\Delta\eta_{\ell\ell}$ vs $\Delta\phi_{\ell\ell}$')
    ax2[0].set_xlabel('$\Delta\eta_{\ell\ell}$')
    ax2[0].set_ylabel('$\Delta\phi_{\ell\ell}$')

    im1 = ax2[1].imshow(H_gen.T, origin='lower', extent=extent, aspect='auto', vmin=0, vmax=vmax)
    ax2[1].set_title('Generated: $\Delta\eta_{\ell\ell}$ vs $\Delta\phi_{\ell\ell}$')
    ax2[1].set_xlabel('$\Delta\eta_{\ell\ell}$')

    ratio = np.divide(H_gen, H_real, out=np.ones_like(H_gen), where=H_real > 0)
    im2 = ax2[2].imshow(ratio.T, origin='lower', extent=extent, aspect='auto', vmin=0.7, vmax=1.3, cmap='coolwarm')
    ax2[2].set_title('Generated / Real ratio')
    ax2[2].set_xlabel('$\Delta\eta_{\ell\ell}$')

    fig2.colorbar(im1, ax=ax2[:2], fraction=0.046, pad=0.04, label='Density')
    fig2.colorbar(im2, ax=ax2[2], fraction=0.046, pad=0.04, label='Ratio')
    fig2.tight_layout()
    two_d_path = output_path.replace('.png', '_2d.png')
    fig2.savefig(two_d_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved correlation diagnostics (2D) to: {two_d_path}")

    # Additional 2D: DeltaPhi(l, MET) vs W_MT.
    wmt_range = (
        min(np.percentile(w_mt_r, 0.5), np.percentile(w_mt_g, 0.5)),
        max(np.percentile(w_mt_r, 99.5), np.percentile(w_mt_g, 99.5)),
    )
    dphi_range = (-np.pi, np.pi)

    H2_real, x2, y2 = np.histogram2d(
        w_mt_r, d_phi_wmet_r,
        bins=bins_2d,
        range=[wmt_range, dphi_range],
        density=True,
        weights=real_weights,
    )
    H2_gen, _, _ = np.histogram2d(
        w_mt_g, d_phi_wmet_g,
        bins=[x2, y2],
        density=True,
    )

    fig3, ax3 = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True, sharey=True)
    extent2 = [x2[0], x2[-1], y2[0], y2[-1]]
    vmax2 = max(np.nanmax(H2_real), np.nanmax(H2_gen))

    im30 = ax3[0].imshow(H2_real.T, origin='lower', extent=extent2, aspect='auto', vmin=0, vmax=vmax2)
    ax3[0].set_title(r'Real: $W_{MT}$ vs $\Delta\phi_{\ell,\mathrm{MET}}$')
    ax3[0].set_xlabel(r'$W_{MT}$')
    ax3[0].set_ylabel(r'$\Delta\phi_{\ell,\mathrm{MET}}$')

    im31 = ax3[1].imshow(H2_gen.T, origin='lower', extent=extent2, aspect='auto', vmin=0, vmax=vmax2)
    ax3[1].set_title(r'Generated: $W_{MT}$ vs $\Delta\phi_{\ell,\mathrm{MET}}$')
    ax3[1].set_xlabel(r'$W_{MT}$')

    ratio2 = np.divide(H2_gen, H2_real, out=np.ones_like(H2_gen), where=H2_real > 0)
    im32 = ax3[2].imshow(ratio2.T, origin='lower', extent=extent2, aspect='auto', vmin=0.7, vmax=1.3, cmap='coolwarm')
    ax3[2].set_title('Generated / Real ratio')
    ax3[2].set_xlabel(r'$W_{MT}$')

    fig3.colorbar(im31, ax=ax3[:2], fraction=0.046, pad=0.04, label='Density')
    fig3.colorbar(im32, ax=ax3[2], fraction=0.046, pad=0.04, label='Ratio')
    fig3.tight_layout()
    two_d_wmt_path = output_path.replace('.png', '_wmt_2d.png')
    fig3.savefig(two_d_wmt_path, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"Saved W_MT-DeltaPhi diagnostics (2D) to: {two_d_wmt_path}")

    # Additional 1D slices: DeltaPhi(l, MET) across W_MT and W_Pt bins.
    binned_wmt_path = output_path.replace('.png', '_dphi_binned_wmt.png')
    _plot_binned_delta_phi(
        d_phi_wmet_r, d_phi_wmet_g,
        w_mt_r, w_mt_g,
        real_weights,
        binned_wmt_path,
        var_label=r'$W_{MT}$',
    )

    binned_wpt_path = output_path.replace('.png', '_dphi_binned_wpt.png')
    _plot_binned_delta_phi(
        d_phi_wmet_r, d_phi_wmet_g,
        w_pt_r, w_pt_g,
        real_weights,
        binned_wpt_path,
        var_label=r'$W_{Pt}$',
    )

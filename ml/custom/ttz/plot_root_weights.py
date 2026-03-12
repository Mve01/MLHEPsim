#!/usr/bin/env python
"""
Plot kinematic distributions from ROOT file weighted by SMEFT weights.

Loads physics features and cHt weights directly from the ROOT file and
produces histograms of each kinematic variable, overlaying c_Ht = -5, -1, 0, 1, 5.
No flow models or sampling involved.

Usage
-----
    python ml/custom/ttz/plot_root_weights.py
    python ml/custom/ttz/plot_root_weights.py --bins 50
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import uproot
import awkward as ak

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / f"plot_root_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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

OUTPUT_DIR = PROJECT_ROOT / "ml" / "custom" / "ttz" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_BRANCHES = [
    'Z_Lepton1_Pt', 'Z_Lepton1_Eta', 'Z_Lepton1_Phi',
    'Z_Lepton2_Pt', 'Z_Lepton2_Eta', 'Z_Lepton2_Phi',
    'W_Lepton_Pt',  'W_Lepton_Eta',  'W_Lepton_Phi',
    'BJet_Pt',      'BJet_Eta',      'BJet_Phi',     'BJet_Mass',
    'MET',          'MET_phi',
]

UNITS = {
    'Pt': '[GeV]', 'pt': '[GeV]', 'MET': '[GeV]', 'Mass': '[GeV]', 'mass': '[GeV]',
    'Phi': '[rad]', 'phi': '[rad]',
}


def compute_derived_features(x: np.ndarray):
    """
    Compute derived higher-order features from the 15 base kinematic features.
    Mirrors compute_derived_features() in smeft_reweighting.py.
    Returns (x_extended, all_feature_names).
    """
    zl1_pt, zl1_eta, zl1_phi = x[:, 0],  x[:, 1],  x[:, 2]
    zl2_pt, zl2_eta, zl2_phi = x[:, 3],  x[:, 4],  x[:, 5]
    wl_pt,              wl_phi = x[:, 6],               x[:, 8]
    bj_pt,              bj_phi = x[:, 9],               x[:, 11]
    met,    met_phi            = x[:, 13], x[:, 14]

    # Z boson (massless leptons)
    zl1_px = zl1_pt * np.cos(zl1_phi);  zl1_py = zl1_pt * np.sin(zl1_phi)
    zl1_pz = zl1_pt * np.sinh(zl1_eta); zl1_E  = zl1_pt * np.cosh(zl1_eta)
    zl2_px = zl2_pt * np.cos(zl2_phi);  zl2_py = zl2_pt * np.sin(zl2_phi)
    zl2_pz = zl2_pt * np.sinh(zl2_eta); zl2_E  = zl2_pt * np.cosh(zl2_eta)

    Z_px = zl1_px + zl2_px;  Z_py = zl1_py + zl2_py
    Z_pz = zl1_pz + zl2_pz;  Z_E  = zl1_E  + zl2_E
    Z_p  = np.sqrt(Z_px**2 + Z_py**2 + Z_pz**2)

    Z_pt   = np.sqrt(Z_px**2 + Z_py**2)
    Z_eta  = np.arctanh(np.clip(Z_pz / np.clip(Z_p, 1e-9, None), -1 + 1e-7, 1 - 1e-7))
    Z_mass = np.sqrt(np.clip(Z_E**2 - Z_p**2, 0.0, None))

    dphi_ll = (zl1_phi - zl2_phi + np.pi) % (2 * np.pi) - np.pi
    Z_deltaR = np.sqrt((zl1_eta - zl2_eta)**2 + dphi_ll**2)

    # W boson transverse (lepton + MET 2D vector sum)
    W_px = wl_pt * np.cos(wl_phi) + met * np.cos(met_phi)
    W_py = wl_pt * np.sin(wl_phi) + met * np.sin(met_phi)
    W_boson_pt = np.sqrt(W_px**2 + W_py**2)

    # Top quark transverse (b-jet + W transverse)
    bj_px = bj_pt * np.cos(bj_phi)
    bj_py = bj_pt * np.sin(bj_phi)
    top_pt = np.sqrt((bj_px + W_px)**2 + (bj_py + W_py)**2)

    # Z / top pT ratio
    Z_over_top_pt = Z_pt / np.clip(top_pt, 1e-3, None)

    derived = np.column_stack([Z_pt, Z_eta, Z_mass, Z_deltaR,
                               W_boson_pt, top_pt, Z_over_top_pt])
    derived_names = ['Z_pt', 'Z_eta', 'Z_mass', 'Z_deltaR',
                     'W_boson_pt', 'top_pt', 'Z_over_top_pt']

    return np.concatenate([x, derived], axis=1), FEATURE_BRANCHES + derived_names

# smeft_weights indices for each cHt value (from ml/data/ttz/cHt_weight_indices.txt)
CHT_INDICES = {
    -5.0: 122,
    -1.0: 121,
     1.0: 123,
     5.0: 124,
}

COLORS = {
    -5.0: '#d62728',  # red
    -1.0: '#ff7f0e',  # orange
     0.0: '#2ca02c',  # green  (SM baseline)
     1.0: '#1f77b4',  # blue
     5.0: '#9467bd',  # purple
}


def feature_unit(name: str) -> str:
    for key, unit in UNITS.items():
        if key in name:
            return unit
    return ''


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bins', type=int, default=50,
                        help='Number of histogram bins (default: 50)')
    parser.add_argument('--n-cols', type=int, default=3,
                        help='Number of columns in the figure (default: 3)')
    args = parser.parse_args()

    log.info("="*70)
    log.info("ROOT Kinematic Distribution Plotter")
    log.info("="*70)
    log.info(f"ROOT file : {DATA_ROOT_FILE}")
    log.info(f"bins      : {args.bins}")

    # ------------------------------------------------------------------
    # Load data from ROOT file
    # ------------------------------------------------------------------
    log.info("\nLoading data from ROOT file...")
    branches = FEATURE_BRANCHES + ['eventWeight']
    with uproot.open(DATA_ROOT_FILE) as f:
        tree = f["Events"]
        data_np = tree.arrays(branches, library='np')
        smeft_ak = tree.arrays(['smeft_weights'], library='ak')['smeft_weights']

    log.info(f"  Total events: {len(data_np['eventWeight'])}")

    # Drop events with fewer than 125 SMEFT weights (ragged array guard)
    valid_mask = ak.to_numpy(ak.num(smeft_ak) > 124)
    n_dropped = int((~valid_mask).sum())
    if n_dropped:
        log.warning(f"  Dropping {n_dropped} events with <125 SMEFT weights")
    for key in data_np:
        data_np[key] = data_np[key][valid_mask]
    smeft_filtered = smeft_ak[valid_mask]

    # Feature matrix: (N, 15)
    x = np.column_stack([data_np[b] for b in FEATURE_BRANCHES]).astype(np.float64)
    N = len(x)
    log.info(f"  Events after filter: {N:,}")

    # Compute derived higher-order features
    x, all_feature_names = compute_derived_features(x)
    log.info(f"  Features: {len(FEATURE_BRANCHES)} base + {len(all_feature_names) - len(FEATURE_BRANCHES)} derived = {len(all_feature_names)} total")

    # Build weights dict; SM uses eventWeight, others use smeft_weights columns
    weights = {0.0: data_np['eventWeight'].astype(np.float64)}
    for c_val, idx in CHT_INDICES.items():
        weights[c_val] = ak.to_numpy(smeft_filtered[:, idx]).astype(np.float64)

    # Log weight statistics
    log.info("\nWeight statistics:")
    for c in sorted(weights.keys()):
        w = weights[c]
        neg = (w < 0).sum()
        log.info(f"  c={c:+.1f}  neg={neg:>8,} ({100*neg/len(w):.2f}%)  "
                 f"mean={w.mean():.4e}  std={w.std():.4e}")

    # Normalise each weight array to sum to 1 so histograms are comparable densities
    weights_norm = {c: w / w.sum() for c, w in weights.items()}

    # ------------------------------------------------------------------
    # Plot: one figure, one panel per feature, all c-values overlaid
    # ------------------------------------------------------------------
    log.info("\nCreating plots...")
    n_feats = len(all_feature_names)
    n_cols  = args.n_cols
    n_rows  = (n_feats + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows + 1.2))

    # Build legend handles first, place them in a header row above the plots
    c_sorted = sorted(weights_norm.keys())
    handles = [plt.Line2D([0], [0], color=COLORS[c],
                          lw=2.5, linestyle='--' if c < 0 else '-',
                          label=f'c_Ht = {c:+.1f}{"  (SM)" if c == 0.0 else ""}')
               for c in c_sorted]

    fig.suptitle(
        'Kinematic distributions from ROOT file (cHt-weighted)',
        fontsize=14, fontweight='bold', y=0.998,
    )

    # Legend placed just below the title, above all panels
    fig.legend(handles=handles, loc='upper center', ncol=len(c_sorted),
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.965),
               handlelength=2.2, columnspacing=1.5)

    gs = GridSpec(n_rows, n_cols, hspace=0.45, wspace=0.35,
                  top=0.93, bottom=0.05, left=0.07, right=0.98)

    for fi, feat in enumerate(all_feature_names):
        row, col = divmod(fi, n_cols)
        ax = fig.add_subplot(gs[row, col])

        feat_data = x[:, fi]
        lo = np.percentile(feat_data, 0.1)
        hi = np.percentile(feat_data, 99.9)
        edges = np.linspace(lo, hi, args.bins + 1)
        bc = (edges[:-1] + edges[1:]) / 2
        dx = edges[1] - edges[0]

        for c in c_sorted:
            w_n = weights_norm[c]
            counts, _ = np.histogram(feat_data, bins=edges, weights=w_n)
            density = counts / dx
            linestyle = '--' if c < 0 else '-'
            ax.step(bc, density, where='mid', lw=1.8,
                    color=COLORS[c], linestyle=linestyle,
                    label=f'c={c:+.1f}')

        unit = feature_unit(feat)
        ax.set_title(feat, fontsize=9, pad=3)
        ax.set_xlabel(unit, fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle='--', alpha=0.3)

    # Hide unused panels
    for fi in range(n_feats, n_rows * n_cols):
        row, col = divmod(fi, n_cols)
        fig.add_subplot(gs[row, col]).axis('off')

    out = OUTPUT_DIR / "root_weighted_distributions.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    log.info(f"\nSaved: {out}")
    plt.close()

    log.info("Done!")


if __name__ == '__main__':
    main()

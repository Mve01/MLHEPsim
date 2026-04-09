#!/usr/bin/env python
"""
Compare ROOT data histograms under three weight treatments:
  1. Raw weights (as computed from ROOT file)
  2. Negative weights clamped to 0
  3. Negative weights clamped to 0 AND weights > 5 capped at 5

Runs for both linear and quadratic decompositions.
Plots each feature with three overlaid histograms for visual comparison.

Usage
-----
    python ml/custom/ttz/test_weight_clipping.py
    python ml/custom/ttz/test_weight_clipping.py --c 5.0
    python ml/custom/ttz/test_weight_clipping.py --n-events 200000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import uproot
import awkward as ak

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

FILE_PATH  = "/project/atlas/users/kdevries/EventLoop/ttZ_for_Melle_tree.root"
IDX_PLUS   = 124  # cHt = +5.0
IDX_MINUS  = 122  # cHt = -5.0
C_NORM     = 5.0  # normalisation constant matching the index choice

FEATURE_NAMES = [
    'Z_Lepton1_Px', 'Z_Lepton1_Py', 'Z_Lepton1_Pz',
    'Z_Lepton2_Px', 'Z_Lepton2_Py', 'Z_Lepton2_Pz',
    'W_Lepton_Px',  'W_Lepton_Py',  'W_Lepton_Pz',
    'BJet_Px',      'BJet_Py',      'BJet_Pz',      'BJet_Mass',
    'MET_Px',       'MET_Py',
]

UNITS = {}
for name in FEATURE_NAMES:
    lower = name.lower()
    if any(k in lower for k in ('_pt', '_px', '_py', '_pz', 'met', '_mass')):
        UNITS[name] = '[GeV]'
    else:
        UNITS[name] = ''


def load_data(n_events):
    branches = FEATURE_NAMES + ['eventWeight']
    print(f"Loading {n_events if n_events else 'all'} events from ROOT file...")
    with uproot.open(FILE_PATH) as f:
        tree = f["Events"]
        kw = dict(entry_stop=n_events) if n_events else {}
        data = tree.arrays(branches, library="np", **kw)
        smeft_ak = tree.arrays(['smeft_weights'], library="ak", **kw)['smeft_weights']

        valid = ak.to_numpy(ak.num(smeft_ak) > IDX_PLUS)
        smeft_ak = smeft_ak[valid]
        for k in data:
            data[k] = data[k][valid]

        w_plus  = ak.to_numpy(smeft_ak[:, IDX_PLUS]).astype(np.float64)
        w_minus = ak.to_numpy(smeft_ak[:, IDX_MINUS]).astype(np.float64)
        w_sm    = data['eventWeight'].astype(np.float64)

    features = np.column_stack([data[f] for f in FEATURE_NAMES]).astype(np.float32)
    w_linear    = (w_plus - w_minus) / (2.0 * C_NORM)
    w_quadratic = (w_plus + w_minus - 2.0 * w_sm) / (C_NORM ** 2)

    print(f"  Loaded {len(features):,} events")
    return features, w_sm, w_linear, w_quadratic


def make_variants(w_raw, cap=5.0):
    """Return (raw, clamp_neg, clamp_neg_and_cap) normalised weight vectors."""
    def normalise(w):
        m = np.mean(w)
        return w / m if m != 0 else w

    w_norm = normalise(w_raw)

    w_clamp = np.abs(w_norm)                   # negatives → positive (abs)
    w_capped = np.clip(w_norm, None, cap)       # cap high values only, negatives kept

    return w_norm, w_clamp, w_capped


def plot_comparison(features, weights_raw, weight_label, c_value, cap, out_dir, bins=50):
    w_raw, w_clamp, w_capped = make_variants(weights_raw, cap=cap)

    # Print stats
    print(f"\n  [{weight_label}] weight stats after normalisation:")
    for name, w in [("raw", w_raw), ("abs", w_clamp), (f"cap_{cap}", w_capped)]:
        n_neg = int((w < 0).sum())
        print(f"    {name:12s}: mean={np.mean(w):.4f}  min={np.min(w):.4f}  "
              f"p99={np.percentile(w,99):.4f}  max={np.max(w):.4f}  n_neg={n_neg}")

    n_features = len(FEATURE_NAMES)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    from matplotlib.gridspec import GridSpec
    # Each feature gets 3 sub-rows: main, ratio_clamp, ratio_capped
    fig = plt.figure(figsize=(5 * n_cols, 6 * n_rows))
    fig.suptitle(
        f'{weight_label} weights  |  c = {c_value:+.1f}  |  '
        f'comparing raw vs clamp_neg vs cap_at_{cap}\n'
        f'N = {len(features):,}',
        fontsize=12, fontweight='bold', y=0.999,
    )
    gs = GridSpec(
        n_rows * 4, n_cols,
        height_ratios=[3, 1, 1, 0.4] * n_rows,
        hspace=0.05, wspace=0.38,
        top=0.97, bottom=0.03, left=0.07, right=0.98,
    )

    for i, feat in enumerate(FEATURE_NAMES):
        row, col = divmod(i, n_cols)
        ax_main   = fig.add_subplot(gs[row * 4,     col])
        ax_r1     = fig.add_subplot(gs[row * 4 + 1, col], sharex=ax_main)
        ax_r2     = fig.add_subplot(gs[row * 4 + 2, col], sharex=ax_main)

        x = features[:, i]
        lo, hi = np.percentile(x, 0.5), np.percentile(x, 99.5)
        dx = (hi - lo) / bins
        n = len(x)

        # Compute histograms for all three variants
        hists = {}
        edges = None
        for w, key in [(w_raw, 'raw'), (w_clamp, 'clamp'), (w_capped, 'capped')]:
            counts, e = np.histogram(x, bins=bins, range=(lo, hi), weights=w, density=True)
            if edges is None:
                edges = e
            hists[key] = counts / (dx * n)

        bc = (edges[:-1] + edges[1:]) / 2

        # Main panel
        for key, label, color, ls in [
            ('raw',    'raw',                  'royalblue', '-'),
            ('clamp',  '|w| (abs)',             'crimson',   '--'),
            ('capped', f'cap at {cap} (no clamp)', 'green', ':'),
        ]:
            ax_main.step(bc, hists[key], where='mid',
                         color=color, lw=1.8, linestyle=ls, label=label)
        ax_main.set_ylabel('Density', fontsize=8)
        ax_main.set_xlim(lo, hi)
        ax_main.set_ylim(bottom=0)
        ax_main.grid(True, linestyle='--', alpha=0.3)
        ax_main.legend(fontsize=7, loc='upper right')
        ax_main.tick_params(labelbottom=False, labelsize=8)
        ax_main.set_title(feat, fontsize=9, pad=3)

        # Ratio 1: clamp_neg / raw
        with np.errstate(divide='ignore', invalid='ignore'):
            r1 = np.where(np.abs(hists['raw']) > 0, hists['clamp'] / hists['raw'], np.nan)
            r2 = np.where(np.abs(hists['raw']) > 0, hists['capped'] / hists['raw'], np.nan)

        ax_r1.step(bc, r1, where='mid', color='crimson', lw=1.5)
        ax_r1.axhline(1.0, color='black', linestyle='--', lw=1.2, alpha=0.5)
        ax_r1.set_ylabel('abs/raw', fontsize=7)
        ax_r1.set_xlim(lo, hi)
        ax_r1.tick_params(labelbottom=False, labelsize=7)
        ax_r1.grid(True, linestyle='--', alpha=0.3)
        # Robust y-limits around 1
        finite = r1[np.isfinite(r1)]
        if len(finite):
            margin = max(0.05, 0.2 * np.std(finite))
            ax_r1.set_ylim(max(0, np.percentile(finite, 1) - margin),
                           np.percentile(finite, 99) + margin)

        ax_r2.step(bc, r2, where='mid', color='green', lw=1.5)
        ax_r2.axhline(1.0, color='black', linestyle='--', lw=1.2, alpha=0.5)
        units = UNITS.get(feat, '')
        ax_r2.set_xlabel(units, fontsize=8)
        ax_r2.set_ylabel(f'cap/raw', fontsize=7)
        ax_r2.set_xlim(lo, hi)
        ax_r2.tick_params(labelbottom=True, labelsize=7)
        ax_r2.grid(True, linestyle='--', alpha=0.3)
        finite2 = r2[np.isfinite(r2)]
        if len(finite2):
            margin2 = max(0.05, 0.2 * np.std(finite2))
            ax_r2.set_ylim(max(0, np.percentile(finite2, 1) - margin2),
                           np.percentile(finite2, 99) + margin2)

    plt.tight_layout(rect=[0, 0, 1, 0.997])
    out_path = out_dir / f"weight_clip_test_{weight_label}_c{c_value:+.1f}.png"
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--c', type=float, default=5.0, help='Wilson coefficient value for histogram weighting')
    p.add_argument('--cap', type=float, default=20.0, help='Cap value for the third weight variant (default 20.0)')
    p.add_argument('--n-events', type=int, default=None, help='Max events to load (default: all)')
    p.add_argument('--bins', type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = PROJECT_ROOT / "ml" / "custom" / "ttz" / "figures" / "weight_clip_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    features, w_sm, w_linear, w_quadratic = load_data(args.n_events)

    # Scale decomposed weights by c and c² to get the actual event weight at value c
    # w(c) = w_sm + c*w_linear + c²*w_quadratic
    # But what we pass to the flow is each decomposition component individually.
    # So we show each component separately.
    print("\n=== Linear weight component ===")
    plot_comparison(features, w_linear, 'linear', args.c, args.cap, out_dir, args.bins)

    print("\n=== Quadratic weight component ===")
    plot_comparison(features, w_quadratic, 'quadratic', args.c, args.cap, out_dir, args.bins)

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()

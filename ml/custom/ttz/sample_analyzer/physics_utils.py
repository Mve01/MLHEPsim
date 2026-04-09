"""Physics utilities for calculating derived quantities from TTZ kinematics.

Supports two base representations via feature names:
- pt/eta/phi
- Cartesian (Px/Py/Pz)
"""

import numpy as np


def _eta_from_xyz(px, py, pz):
    p = np.sqrt(px**2 + py**2 + pz**2)
    return np.arctanh(np.clip(pz / np.clip(p, 1e-9, None), -1 + 1e-7, 1 - 1e-7))


def _has(feature_names, name):
    return name in feature_names


def _phi_from_features(data, feature_names, phi_name):
    if _has(feature_names, phi_name):
        return data[:, feature_names.index(phi_name)]
    raise KeyError(f"Missing phi feature: expected {phi_name}")


def _get_particle_xyz(data, feature_names, prefix):
    px_name, py_name, pz_name = f"{prefix}_Px", f"{prefix}_Py", f"{prefix}_Pz"
    if _has(feature_names, px_name) and _has(feature_names, py_name) and _has(feature_names, pz_name):
        return (
            data[:, feature_names.index(px_name)],
            data[:, feature_names.index(py_name)],
            data[:, feature_names.index(pz_name)],
        )

    pt_name, eta_name = f"{prefix}_Pt", f"{prefix}_Eta"
    if not (_has(feature_names, pt_name) and _has(feature_names, eta_name)):
        raise KeyError(f"Missing kinematics for {prefix}: expected cartesian or ({pt_name}, {eta_name}, phi)")

    phi = _phi_from_features(data, feature_names, f"{prefix}_Phi")
    pt = data[:, feature_names.index(pt_name)]
    eta = data[:, feature_names.index(eta_name)]
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return px, py, pz


def _get_met_xy(data, feature_names):
    if _has(feature_names, 'MET_Px') and _has(feature_names, 'MET_Py'):
        return data[:, feature_names.index('MET_Px')], data[:, feature_names.index('MET_Py')]

    if not _has(feature_names, 'MET'):
        raise KeyError("Missing MET magnitude feature 'MET'")
    met = data[:, feature_names.index('MET')]
    phi = _phi_from_features(data, feature_names, 'MET_Phi')
    return met * np.cos(phi), met * np.sin(phi)


def calculate_delta_r(px1, py1, pz1, px2, py2, pz2):
    """Calculate DeltaR using cartesian components for two particles."""
    eta1 = _eta_from_xyz(px1, py1, pz1)
    eta2 = _eta_from_xyz(px2, py2, pz2)
    phi1 = np.arctan2(py1, px1)
    phi2 = np.arctan2(py2, px2)

    delta_eta = eta1 - eta2
    delta_phi = np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))
    return np.sqrt(delta_eta**2 + delta_phi**2)


def calculate_z_kinematics(data, feature_names):
    """Calculate Z kinematics from Z_Lepton1 and Z_Lepton2."""
    px1, py1, pz1 = _get_particle_xyz(data, feature_names, 'Z_Lepton1')
    px2, py2, pz2 = _get_particle_xyz(data, feature_names, 'Z_Lepton2')

    # Massless leptons
    e1 = np.sqrt(np.clip(px1**2 + py1**2 + pz1**2, 0.0, None))
    e2 = np.sqrt(np.clip(px2**2 + py2**2 + pz2**2, 0.0, None))

    px_z = px1 + px2
    py_z = py1 + py2
    pz_z = pz1 + pz2
    e_z = e1 + e2

    z_pt = np.sqrt(px_z**2 + py_z**2)
    z_phi = np.arctan2(py_z, px_z)
    z_eta = _eta_from_xyz(px_z, py_z, pz_z)

    p2 = px_z**2 + py_z**2 + pz_z**2
    z_mass = np.sqrt(np.clip(e_z**2 - p2, 0.0, None))

    z_delta_r = calculate_delta_r(px1, py1, pz1, px2, py2, pz2)

    return {
        'Z_Pt': z_pt,
        'Z_Eta': z_eta,
        'Z_Phi': z_phi,
        'Z_Mass': z_mass,
        'Z_DeltaR': z_delta_r,
    }


def calculate_w_kinematics(data, feature_names):
    """Calculate W transverse kinematics from W_Lepton and MET."""
    lep_px, lep_py, _ = _get_particle_xyz(data, feature_names, 'W_Lepton')
    met_px, met_py = _get_met_xy(data, feature_names)

    w_px = lep_px + met_px
    w_py = lep_py + met_py

    lep_pt = np.sqrt(lep_px**2 + lep_py**2)
    met_pt = np.sqrt(met_px**2 + met_py**2)
    w_pt = np.sqrt(w_px**2 + w_py**2)
    w_phi = np.arctan2(w_py, w_px)

    lep_phi = np.arctan2(lep_py, lep_px)
    met_phi = np.arctan2(met_py, met_px)
    dphi = np.arctan2(np.sin(lep_phi - met_phi), np.cos(lep_phi - met_phi))
    w_mt = np.sqrt(np.clip(2 * lep_pt * met_pt * (1 - np.cos(dphi)), 0.0, None))

    return {
        'W_Pt': w_pt,
        'W_Phi': w_phi,
        'W_MT': w_mt,
    }


def calculate_top_kinematics(data, feature_names):
    """Calculate top transverse kinematics from BJet, W_Lepton, and MET."""
    bjet_px, bjet_py, _ = _get_particle_xyz(data, feature_names, 'BJet')

    w_kin = calculate_w_kinematics(data, feature_names)
    w_pt = w_kin['W_Pt']
    w_phi = w_kin['W_Phi']
    w_mt = w_kin['W_MT']

    w_px = w_pt * np.cos(w_phi)
    w_py = w_pt * np.sin(w_phi)

    top_px = w_px + bjet_px
    top_py = w_py + bjet_py
    top_pt = np.sqrt(top_px**2 + top_py**2)
    top_phi = np.arctan2(top_py, top_px)

    bjet_pt = np.sqrt(bjet_px**2 + bjet_py**2)
    bjet_phi = np.arctan2(bjet_py, bjet_px)
    dphi = np.arctan2(np.sin(w_phi - bjet_phi), np.cos(w_phi - bjet_phi))
    top_mt = np.sqrt(np.clip((w_mt + bjet_pt)**2 + 2 * w_pt * bjet_pt * (1 - np.cos(dphi)), 0.0, None))

    return {
        'Top_Pt': top_pt,
        'Top_Phi': top_phi,
        'Top_MT': top_mt,
    }
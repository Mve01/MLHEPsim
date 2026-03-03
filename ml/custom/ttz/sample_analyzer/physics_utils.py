"""Physics utilities for calculating derived quantities from particle kinematics."""

import numpy as np


# Particle masses in GeV
Z_MASS = 91.1876
W_MASS = 80.379
TOP_MASS = 172.76


def calculate_delta_r(eta1, phi1, eta2, phi2):
    """
    Calculate DeltaR angular distance between two particles.
    
    DeltaR = sqrt(delta_eta^2 + delta_phi^2)
    
    Parameters
    ----------
    eta1, phi1 : array-like
        Pseudorapidity and azimuthal angle of particle 1
    eta2, phi2 : array-like
        Pseudorapidity and azimuthal angle of particle 2
        
    Returns
    -------
    delta_r : array-like
        Angular distance
    """
    delta_eta = eta1 - eta2
    delta_phi = phi1 - phi2
    
    # Wrap delta_phi to [-pi, pi]
    delta_phi = np.arctan2(np.sin(delta_phi), np.cos(delta_phi))
    
    return np.sqrt(delta_eta**2 + delta_phi**2)


def calculate_z_kinematics(data, feature_names):
    """
    Calculate Z boson kinematics from pre-identified Z_Lepton1 and Z_Lepton2.

    Uses cylindrical coordinates (Pt, Eta, Phi) directly from data.
    Leptons are assumed massless: E = Pt*cosh(eta), Pz = Pt*sinh(eta).
    Returns Z_Pt, Z_Eta, Z_Phi, Z_Mass, and Z_DeltaR.

    Parameters
    ----------
    data : np.ndarray
        Data array with shape (n_events, n_features)
    feature_names : list
        List of feature names corresponding to data columns

    Returns
    -------
    z_kinematics : dict
        Dictionary with keys: 'Z_Pt', 'Z_Eta', 'Z_Phi', 'Z_Mass', 'Z_DeltaR'
        Each value is an array of shape (n_events,)
    """
    pt1  = data[:, feature_names.index('Z_Lepton1_Pt')]
    eta1 = data[:, feature_names.index('Z_Lepton1_Eta')]
    phi1 = data[:, feature_names.index('Z_Lepton1_Phi')]
    pt2  = data[:, feature_names.index('Z_Lepton2_Pt')]
    eta2 = data[:, feature_names.index('Z_Lepton2_Eta')]
    phi2 = data[:, feature_names.index('Z_Lepton2_Phi')]

    # Massless lepton 4-vectors
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    E1  = pt1 * np.cosh(eta1)

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    E2  = pt2 * np.cosh(eta2)

    # Z 4-vector (sum of two leptons)
    E_z  = E1  + E2
    px_z = px1 + px2
    py_z = py1 + py2
    pz_z = pz1 + pz2

    z_pt  = np.sqrt(px_z**2 + py_z**2)
    z_phi = np.arctan2(py_z, px_z)

    p_tot = np.sqrt(px_z**2 + py_z**2 + pz_z**2)
    z_eta = np.where(
        p_tot > np.abs(pz_z),
        0.5 * np.log((p_tot + pz_z) / (p_tot - pz_z)),
        0.0
    )

    m_squared = E_z**2 - (px_z**2 + py_z**2 + pz_z**2)
    z_mass = np.sqrt(np.maximum(m_squared, 0))

    z_delta_r = calculate_delta_r(eta1, phi1, eta2, phi2)

    return {
        'Z_Pt':     z_pt,
        'Z_Eta':    z_eta,
        'Z_Phi':    z_phi,
        'Z_Mass':   z_mass,
        'Z_DeltaR': z_delta_r,
    }


def calculate_w_kinematics(data, feature_names):
    """
    Calculate W boson kinematics from pre-identified W_Lepton and MET.

    Uses cylindrical coordinates (Pt, Eta, Phi) directly from data.
    MET is stored as scalar magnitude (MET) and azimuthal angle (MET_Phi).

    In ttZ events with 3 leptons:
    - 2 leptons come from Z decay (Z_Lepton1, Z_Lepton2)
    - 1 lepton comes from W decay (W_Lepton) along with neutrino
    - The neutrino escapes detection -> appears as MET

    Parameters
    ----------
    data : np.ndarray
        Data array of shape (n_events, n_features)
    feature_names : list
        List of feature names corresponding to columns in data

    Returns
    -------
    dict
        Dictionary with keys:
        - W_Pt:  Transverse momentum of W (GeV)
        - W_Phi: Azimuthal angle of W (radians)
        - W_MT:  Transverse mass of W (GeV)

    Notes
    -----
    Transverse mass: MT = sqrt(2 * pt_lep * MET * (1 - cos(dphi)))
    Expected peak around 80.4 GeV.
    """
    lep_pt  = data[:, feature_names.index('W_Lepton_Pt')]
    lep_phi = data[:, feature_names.index('W_Lepton_Phi')]
    met     = data[:, feature_names.index('MET')]
    met_phi = data[:, feature_names.index('MET_Phi')]

    # W transverse momentum (vector sum of lepton and MET)
    w_px = lep_pt * np.cos(lep_phi) + met * np.cos(met_phi)
    w_py = lep_pt * np.sin(lep_phi) + met * np.sin(met_phi)

    w_pt  = np.sqrt(w_px**2 + w_py**2)
    w_phi = np.arctan2(w_py, w_px)

    # Transverse mass
    dphi = np.arctan2(np.sin(lep_phi - met_phi), np.cos(lep_phi - met_phi))
    w_mt = np.sqrt(2 * lep_pt * met * (1 - np.cos(dphi)))

    return {
        'W_Pt':  w_pt,
        'W_Phi': w_phi,
        'W_MT':  w_mt,
    }


def calculate_top_kinematics(data, feature_names):
    """
    Calculate top quark kinematics from W boson and BJet.

    Uses cylindrical coordinates (Pt, Eta, Phi) directly from data.

    In ttZ events:
    - One top decays to W + b-jet
    - The W decays to lepton + neutrino (reconstructed above)
    - BJet is the pre-identified b-tagged jet from this top decay

    Parameters
    ----------
    data : np.ndarray
        Data array of shape (n_events, n_features)
    feature_names : list
        List of feature names corresponding to columns in data

    Returns
    -------
    dict
        Dictionary with keys:
        - Top_Pt:  Transverse momentum of top (GeV)
        - Top_Phi: Azimuthal angle of top (radians)
        - Top_MT:  Transverse mass of top (GeV)

    Notes
    -----
    Expected peak around 172.8 GeV.
    """
    bjet_pt  = data[:, feature_names.index('BJet_Pt')]
    bjet_phi = data[:, feature_names.index('BJet_Phi')]

    # Get W kinematics (vectorized)
    w_kin = calculate_w_kinematics(data, feature_names)
    w_pt  = w_kin['W_Pt']
    w_phi = w_kin['W_Phi']
    w_mt  = w_kin['W_MT']

    # Top transverse momentum (vector sum of W and b-jet)
    top_px = w_pt * np.cos(w_phi) + bjet_pt * np.cos(bjet_phi)
    top_py = w_pt * np.sin(w_phi) + bjet_pt * np.sin(bjet_phi)

    top_pt  = np.sqrt(top_px**2 + top_py**2)
    top_phi = np.arctan2(top_py, top_px)

    # Transverse mass combining W and b-jet
    dphi = np.arctan2(np.sin(w_phi - bjet_phi), np.cos(w_phi - bjet_phi))
    top_mt = np.sqrt((w_mt + bjet_pt)**2 + 2 * w_pt * bjet_pt * (1 - np.cos(dphi)))

    return {
        'Top_Pt':  top_pt,
        'Top_Phi': top_phi,
        'Top_MT':  top_mt,
    }

"""Physics utilities for calculating derived quantities from particle kinematics."""

import numpy as np


# Particle masses in GeV
Z_MASS = 91.1876
W_MASS = 80.379
TOP_MASS = 172.76


def calculate_invariant_mass(pt1, eta1, phi1, pt2, eta2, phi2):
    """
    Calculate invariant mass of two particles from their pt, eta, phi.
    
    Assumes massless particles (good approximation for leptons).
    
    Parameters
    ----------
    pt1, eta1, phi1 : array-like
        Transverse momentum, pseudorapidity, azimuthal angle of particle 1
    pt2, eta2, phi2 : array-like
        Transverse momentum, pseudorapidity, azimuthal angle of particle 2
        
    Returns
    -------
    m_inv : array-like
        Invariant mass in GeV
    """
    # Convert to 4-vectors (assuming massless particles)
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    E1 = pt1 * np.cosh(eta1)
    
    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    E2 = pt2 * np.cosh(eta2)
    
    # 4-vector sum
    px_sum = px1 + px2
    py_sum = py1 + py2
    pz_sum = pz1 + pz2
    E_sum = E1 + E2
    
    # Invariant mass: m^2 = E^2 - p^2
    m_squared = E_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2)
    
    # Handle numerical errors that might make m_squared slightly negative
    m_squared = np.maximum(m_squared, 0)
    
    return np.sqrt(m_squared)


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


def find_z_candidate_leptons(data, feature_names):
    """
    Find the two leptons whose invariant mass is closest to the Z mass.
    
    For ttZ events, we have 3 leptons. This finds the pair that forms
    the Z boson candidate (closest to 91.2 GeV).
    
    Parameters
    ----------
    data : np.ndarray
        Data array with shape (n_events, n_features)
    feature_names : list
        List of feature names corresponding to data columns
        
    Returns
    -------
    z_lepton_indices : np.ndarray
        Array of shape (n_events, 2) with indices of the two leptons
        forming the Z candidate for each event
    """
    # Get lepton indices
    lepton_features = {
        1: {}, 2: {}, 3: {}
    }
    
    for i, name in enumerate(feature_names):
        if 'Lepton1' in name:
            if '_Pt' in name:
                lepton_features[1]['pt'] = i
            elif '_Eta' in name:
                lepton_features[1]['eta'] = i
            elif '_Phi' in name:
                lepton_features[1]['phi'] = i
        elif 'Lepton2' in name:
            if '_Pt' in name:
                lepton_features[2]['pt'] = i
            elif '_Eta' in name:
                lepton_features[2]['eta'] = i
            elif '_Phi' in name:
                lepton_features[2]['phi'] = i
        elif 'Lepton3' in name:
            if '_Pt' in name:
                lepton_features[3]['pt'] = i
            elif '_Eta' in name:
                lepton_features[3]['eta'] = i
            elif '_Phi' in name:
                lepton_features[3]['phi'] = i
    
    n_events = len(data)
    
    # Calculate invariant masses for all three lepton pairs
    mass_12 = calculate_invariant_mass(
        data[:, lepton_features[1]['pt']], data[:, lepton_features[1]['eta']], data[:, lepton_features[1]['phi']],
        data[:, lepton_features[2]['pt']], data[:, lepton_features[2]['eta']], data[:, lepton_features[2]['phi']]
    )
    
    mass_13 = calculate_invariant_mass(
        data[:, lepton_features[1]['pt']], data[:, lepton_features[1]['eta']], data[:, lepton_features[1]['phi']],
        data[:, lepton_features[3]['pt']], data[:, lepton_features[3]['eta']], data[:, lepton_features[3]['phi']]
    )
    
    mass_23 = calculate_invariant_mass(
        data[:, lepton_features[2]['pt']], data[:, lepton_features[2]['eta']], data[:, lepton_features[2]['phi']],
        data[:, lepton_features[3]['pt']], data[:, lepton_features[3]['eta']], data[:, lepton_features[3]['phi']]
    )
    
    # Find which pair is closest to Z mass for each event
    diff_12 = np.abs(mass_12 - Z_MASS)
    diff_13 = np.abs(mass_13 - Z_MASS)
    diff_23 = np.abs(mass_23 - Z_MASS)
    
    # Stack differences and find minimum
    diffs = np.stack([diff_12, diff_13, diff_23], axis=1)
    min_pair_idx = np.argmin(diffs, axis=1)
    
    # Map indices to lepton pairs: 0->(1,2), 1->(1,3), 2->(2,3)
    z_lepton_indices = np.zeros((n_events, 2), dtype=int)
    z_lepton_indices[min_pair_idx == 0] = [1, 2]
    z_lepton_indices[min_pair_idx == 1] = [1, 3]
    z_lepton_indices[min_pair_idx == 2] = [2, 3]
    
    return z_lepton_indices, lepton_features


def calculate_z_kinematics(data, feature_names):
    """
    Calculate Z boson candidate kinematics from the two leptons closest to Z mass.
    
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
    z_lepton_indices, lepton_features = find_z_candidate_leptons(data, feature_names)
    
    n_events = len(data)
    z_pt = np.zeros(n_events)
    z_eta = np.zeros(n_events)
    z_phi = np.zeros(n_events)
    z_mass = np.zeros(n_events)
    z_delta_r = np.zeros(n_events)
    
    # Calculate Z kinematics for each event
    for i in range(n_events):
        lep1_idx, lep2_idx = z_lepton_indices[i]
        
        # Get lepton kinematics
        pt1 = data[i, lepton_features[lep1_idx]['pt']]
        eta1 = data[i, lepton_features[lep1_idx]['eta']]
        phi1 = data[i, lepton_features[lep1_idx]['phi']]
        
        pt2 = data[i, lepton_features[lep2_idx]['pt']]
        eta2 = data[i, lepton_features[lep2_idx]['eta']]
        phi2 = data[i, lepton_features[lep2_idx]['phi']]
        
        # Calculate 4-vector components (assuming massless leptons)
        px1 = pt1 * np.cos(phi1)
        py1 = pt1 * np.sin(phi1)
        pz1 = pt1 * np.sinh(eta1)
        E1 = pt1 * np.cosh(eta1)
        
        px2 = pt2 * np.cos(phi2)
        py2 = pt2 * np.sin(phi2)
        pz2 = pt2 * np.sinh(eta2)
        E2 = pt2 * np.cosh(eta2)
        
        # Z 4-vector
        px_z = px1 + px2
        py_z = py1 + py2
        pz_z = pz1 + pz2
        E_z = E1 + E2
        
        # Calculate Z kinematics
        z_pt[i] = np.sqrt(px_z**2 + py_z**2)
        z_phi[i] = np.arctan2(py_z, px_z)
        
        # Eta from pz and pt
        p_z_mag = np.sqrt(px_z**2 + py_z**2 + pz_z**2)
        z_eta[i] = 0.5 * np.log((p_z_mag + pz_z) / (p_z_mag - pz_z)) if p_z_mag != pz_z else 0.0
        
        # Invariant mass
        m_squared = E_z**2 - (px_z**2 + py_z**2 + pz_z**2)
        z_mass[i] = np.sqrt(max(m_squared, 0))
        
        # DeltaR between the two leptons
        z_delta_r[i] = calculate_delta_r(eta1, phi1, eta2, phi2)
    
    return {
        'Z_Pt': z_pt,
        'Z_Eta': z_eta,
        'Z_Phi': z_phi,
        'Z_Mass': z_mass,
        'Z_DeltaR': z_delta_r
    }


def calculate_w_kinematics(data, feature_names):
    """
    Calculate W boson kinematics from the third lepton (not part of Z) and MET.
    
    In ttZ events with 3 leptons:
    - 2 leptons come from Z decay
    - 1 lepton comes from W decay (along with neutrino)
    - The neutrino escapes detection → appears as MET
    
    This function identifies the non-Z lepton and reconstructs the W boson.
    
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
        - W_Pt: Transverse momentum of W (GeV)
        - W_Phi: Azimuthal angle of W (radians)
        - W_MT: Transverse mass of W (GeV) - doesn't require pz reconstruction
        
    Notes
    -----
    We calculate the transverse mass MT instead of full invariant mass because
    we don't have enough information to reconstruct the longitudinal momentum
    of the neutrino (only MET gives transverse components).
    
    Transverse mass is commonly used in W reconstruction:
    MT = sqrt(2 * pt_lep * MET * (1 - cos(dphi)))
    
    For W reconstruction: expects peak around 80.4 GeV
    """
    n_events = data.shape[0]
    
    # Initialize output arrays
    w_pt = np.zeros(n_events)
    w_phi = np.zeros(n_events)
    w_mt = np.zeros(n_events)
    
    # Get lepton indices for all 3 leptons
    lepton_features = []
    for lep_num in [1, 2, 3]:
        lepton_features.append({
            'pt': feature_names.index(f'Lepton{lep_num}_Pt'),
            'eta': feature_names.index(f'Lepton{lep_num}_Eta'),
            'phi': feature_names.index(f'Lepton{lep_num}_Phi'),
        })
    
    # Get MET indices
    met_idx = feature_names.index('MET')
    met_phi_idx = feature_names.index('MET_Phi')
    
    # First identify which leptons form the Z candidate (closest to Z mass)
    z_candidates, lepton_features = find_z_candidate_leptons(data, feature_names)
    
    # For each event
    for i in range(n_events):
        # Get the non-Z lepton (the one not in z_candidates)
        # z_candidates contains lepton numbers (1, 2, 3), not array indices
        z_lep1, z_lep2 = z_candidates[i]
        all_leptons = {1, 2, 3}
        w_lepton_num = list(all_leptons - {z_lep1, z_lep2})[0]
        
        # Get W lepton kinematics (convert lepton number to index)
        lep_pt = data[i, lepton_features[w_lepton_num]['pt']]
        lep_phi = data[i, lepton_features[w_lepton_num]['phi']]
        
        # Get MET
        met = data[i, met_idx]
        met_phi = data[i, met_phi_idx]
        
        # Calculate W transverse momentum (vector sum)
        lep_px = lep_pt * np.cos(lep_phi)
        lep_py = lep_pt * np.sin(lep_phi)
        met_px = met * np.cos(met_phi)
        met_py = met * np.sin(met_phi)
        
        w_px = lep_px + met_px
        w_py = lep_py + met_py
        
        w_pt[i] = np.sqrt(w_px**2 + w_py**2)
        w_phi[i] = np.arctan2(w_py, w_px)
        
        # Calculate transverse mass
        dphi = lep_phi - met_phi
        # Wrap dphi to [-pi, pi]
        dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
        
        w_mt[i] = np.sqrt(2 * lep_pt * met * (1 - np.cos(dphi)))
    
    return {
        'W_Pt': w_pt,
        'W_Phi': w_phi,
        'W_MT': w_mt,  # Transverse mass (most relevant for W reconstruction)
    }


def calculate_top_kinematics(data, feature_names):
    """
    Calculate top quark kinematics from W boson and b-jet (Jet1).
    
    In ttZ events:
    - One top decays to W + b-jet
    - The W decays to lepton + neutrino (reconstructed above)
    - Jet1 is assumed to be the b-jet from this top decay
    
    This gives us a hadronic top candidate reconstruction.
    
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
        - Top_Pt: Transverse momentum of top (GeV)
        - Top_Phi: Azimuthal angle of top (radians)
        - Top_MT: Transverse mass of top (GeV)
        
    Notes
    -----
    Similar to W, we use transverse mass because we don't have full 
    neutrino information. The transverse mass is calculated from:
    - W transverse momentum (from lepton + MET)
    - b-jet transverse momentum
    
    For top quark: expects peak around 172.8 GeV
    """
    n_events = data.shape[0]
    
    # Initialize output arrays
    top_pt = np.zeros(n_events)
    top_phi = np.zeros(n_events)
    top_mt = np.zeros(n_events)
    
    # Get W kinematics (already calculates W from non-Z lepton + MET)
    w_kinematics = calculate_w_kinematics(data, feature_names)
    w_pt = w_kinematics['W_Pt']
    w_phi = w_kinematics['W_Phi']
    w_mt = w_kinematics['W_MT']
    
    # Get Jet1 (b-jet) indices
    jet1_pt_idx = feature_names.index('Jet1_Pt')
    jet1_phi_idx = feature_names.index('Jet1_Phi')
    
    # For each event
    for i in range(n_events):
        # Get b-jet kinematics
        bjet_pt = data[i, jet1_pt_idx]
        bjet_phi = data[i, jet1_phi_idx]
        
        # Get W kinematics for this event
        w_pt_i = w_pt[i]
        w_phi_i = w_phi[i]
        
        # Calculate top transverse momentum (vector sum of W and b-jet)
        w_px = w_pt_i * np.cos(w_phi_i)
        w_py = w_pt_i * np.sin(w_phi_i)
        bjet_px = bjet_pt * np.cos(bjet_phi)
        bjet_py = bjet_pt * np.sin(bjet_phi)
        
        top_px = w_px + bjet_px
        top_py = w_py + bjet_py
        
        top_pt[i] = np.sqrt(top_px**2 + top_py**2)
        top_phi[i] = np.arctan2(top_py, top_px)
        
        # Calculate transverse mass
        # Using approximation: MT_top ≈ sqrt(MT_W^2 + pT_b^2) + pT_W
        # This is a simplified transverse mass for the 3-body system
        dphi = w_phi_i - bjet_phi
        dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
        
        # Transverse mass combining W and b-jet
        top_mt[i] = np.sqrt((w_mt[i] + bjet_pt)**2 + 2 * w_pt_i * bjet_pt * (1 - np.cos(dphi)))
    
    return {
        'Top_Pt': top_pt,
        'Top_Phi': top_phi,
        'Top_MT': top_mt,  # Transverse mass (commonly used for top reconstruction)
    }

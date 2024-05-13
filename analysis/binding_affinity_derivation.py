import numpy as np


def fit_params(kplus, kminus, pplus, pminus):
    """
    Returns the correct eta_tilde and eta for our model construction
    """

    eta = -(np.log(kplus) - np.log(kminus)) / (np.log(pplus) - np.log(pminus))
    # eta_tilde = kplus / (np.power(pplus, -eta))
    eta_tilde = kminus / (np.power(pminus, -eta))
    return eta, eta_tilde


def get_Kd_from_tulip_probs(tulip_probs, eta, eta_tilde):
    """
    Tulip probs is an unscaled probability.
    """
    return eta_tilde * np.power(tulip_probs, -eta)


def get_pbind_from_Kd(Kds, T_conc):
    return 1 / (1 + (Kds / T_conc))


def get_pbind_from_tulip_probs(tulip_probs, eta, eta_tilde, T_conc):
    """
    Provided you have fit eta and eta_tilde, get your pbind values.
    """

    Kds = get_Kd_from_tulip_probs(tulip_probs, eta, eta_tilde)
    return get_pbind_from_Kd(Kds, T_conc)


def fit_params_and_get_pbind_from_tulip_probs(tulip_probs, kplus, kminus, T_conc):
    """
    Get pbind directly in one fell swoop (assuming the points to fit, pplus and pminus are just the max and min of your input probabilities).
    """
    pplus = max(tulip_probs)
    pminus = min(tulip_probs)
    eta, eta_tilde = fit_params(kplus, kminus, pplus, pminus)
    return get_pbind_from_tulip_probs(tulip_probs, eta, eta_tilde, T_conc)

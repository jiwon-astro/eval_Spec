#======================
# Spectrum extraction
#=======================

import numpy as np
from numpy.polynomial.chebyshev import chebval

from .tracing import trace_spectrum, trace_fit

Ncol = 1340
FWHM_AP = 10
step_AP = 10
N_AP = Ncol//step_AP

def extract_profile(data, texp, trace_center=[], params=False, width=5,
                    nspec=1, bound=(0, Ncol), deg=6, sigma=2.0, n_iter=5, fwhm_ap = FWHM_AP, step_ap = step_AP,
                    rdn = 0.0, gain = 1.0, fpn = 0.0):
    """Extract 1D spectrum by averaging a fixed-width aperture around a traced center.

    Parameters
    ----------
    data : 2D ndarray
        Image (rows=spatial, cols=dispersion).
    texp : float
        Exposure time [s] (used to return per-second rates).
    bound : (yi, yf)
        Row range used for tracing.
    nspec : int
        Number of spectra to extract.
    width : int
        Half-width of extraction window in pixels.
    trace_center : ndarray or None
        If provided, center row for each column (shape (nspec, ncol)).
        If None, trace and fit automatically.
    deg, sigma, n_iter : tracing fit controls
    rdn, gain, fpn : detector parameters for a photometric error estimation

    Returns
    -------
    profile : ndarray, shape (nspec, ncol, 2*width+1)
        Cutouts around the trace (per second).
    spec : ndarray, shape (nspec, ncol)
        Mean counts in the aperture (per second).
    e_spec : ndarray, shape (nspec, ncol)
        1-sigma uncertainty (per second).
    trace_center : ndarray, shape (nspec, ncol)
        Trace center used for extraction.
    """
    nrow, ncol = data.shape
    pixel = np.arange(ncol)

    if trace_center is None or (isinstance(trace_center, (list, tuple)) and len(trace_center) == 0):
        xpos, trace, _fwhm = trace_spectrum(
            data, bound=bound, nspec=nspec, fwhm_ap=fwhm_ap, step_ap=step_ap
        )
        params, _masks = trace_fit(xpos, trace, nspec=nspec, sigma=sigma, n_iter=n_iter, deg=deg)
        trace_center = np.array([np.around(chebval(pixel, params[i])) for i in range(nspec)], dtype=int)

    trace_center = np.asarray(trace_center, dtype=int)
    if trace_center.shape != (nspec, ncol):
        raise ValueError(f"trace_center must have shape (nspec, ncol)=({nspec},{ncol}), got {trace_center.shape}")

    profile = np.zeros((nspec, ncol, 2 * width + 1), dtype=float)
    spec = np.zeros((nspec, ncol), dtype=float)
    e_spec = np.zeros((nspec, ncol), dtype=float)

    for i in range(nspec):
        for j in range(ncol):
            yc = trace_center[i, j]
            lb = max(yc - width, 0)
            ub = min(yc + width + 1, nrow)
            # pad if at edges
            cut = data[lb:ub, j]
            if cut.size < 2 * width + 1:
                pad_before = max(0, (yc - width) - lb)
                pad_after = (2 * width + 1) - (pad_before + cut.size)
                cut = np.pad(cut, (pad_before, pad_after), mode="edge")
            profile[i, j, :] = cut

        # photometric error (ADU units): RN^2 + Poisson + fixed-pattern term
        var_profile = rdn**2 + profile[i] / max(gain, 1e-12) + (fpn * profile[i]) ** 2
        spec[i] = np.mean(profile[i], axis=1)
        e_spec[i] = np.sqrt(np.sum(var_profile, axis=1) / (2 * width + 1))

    # return per-second
    profile /= texp
    spec /= texp
    e_spec /= texp
    return profile, spec, e_spec, trace_center
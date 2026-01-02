#==============================
# Trace finding and fitting
#==============================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from numpy.polynomial.chebyshev import chebfit, chebval
from scipy.signal import find_peaks

from astropy.modeling.models import Gaussian1D
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clip

Ncol = 1340
FWHM_AP = 10
step_AP = 10
N_AP = Ncol//step_AP

def peak_detection(x, nspec = 1, bound = (0, Ncol), fwhm_ap = FWHM_AP):
    """Find `nspec` strongest peaks in 1D profile `x` within `bound` (row indices).

    Returns
    -------
    flag : bool
        True if at least `nspec` peaks found.
    idx : ndarray
        Peak indices (sorted).
    """
    noise = np.std(x)
    idx, _ = find_peaks(x, distance=fwhm_ap, height=2 * noise)
    idx = idx[(bound[0] < idx) & (idx < bound[1])]
    if len(idx) < nspec:
        return False, np.zeros(nspec)
    idx = idx[np.argsort(x[idx])[::-1]][:nspec]
    idx = np.sort(idx)
    return True, idx

def trace_spectrum(data, nspec = 1, bound = (0, Ncol), fwhm_ap = FWHM_AP, step_ap = step_AP):
    """Trace spectra (fiber centers) along dispersion axis.

    Parameters
    ----------
    data : 2D ndarray
        Image (rows=spatial, cols=dispersion).
    bound : (y0, y1)
        Row search window.
    nspec : int
        Number of traces to find.
    fwhm_ap : int
        Approx. FWHM in pixels for peak separation and fit window.
    step_ap : int
        Column bin size used to build spatial profiles along dispersion axis.

    Returns
    -------
    xpos : ndarray
        Sampled column positions.
    trace : ndarray, shape (nspec, len(xpos))
        Fitted center rows for each trace.
    fwhm : ndarray, shape (nspec, len(xpos))
        Fitted FWHM for each trace.
    """
    nrow, ncol = data.shape
    n_ap = ncol // step_ap
    xpos = []
    ypos = np.arange(-3 * fwhm_ap, 3 * fwhm_ap + 1)

    trace = np.zeros((1, nspec))
    trace_fwhm = []
    center_pix_new = np.zeros(nspec)

    fitter = LevMarLSQFitter()

    for ii in range(n_ap - 1):
        fwhm_tmp = []
        y = np.mean(data[:, ii * step_ap : (ii + 1) * step_ap], axis=1)
        flag, idx = peak_detection(y, nspec, bound, fwhm_ap=fwhm_ap)
        if not flag:
            continue

        xpos.append(step_ap * ii)
        for jj in range(nspec):
            peak_pix = int(idx[jj])
            lb, ub = peak_pix - 3 * fwhm_ap, peak_pix + 3 * fwhm_ap + 1
            lb = max(lb, 0)
            ub = min(ub, nrow)

            y_cropped = y[lb:ub]
            # local coordinate centered at peak_pix
            local = np.arange(lb - peak_pix, ub - peak_pix)

            g_init = Gaussian1D(
                amplitude=y[peak_pix],
                mean=0,
                stddev=gaussian_fwhm_to_sigma * fwhm_ap,
                bounds={
                    "amplitude": (0, 2 * y[peak_pix]),
                    "mean": (-3 * fwhm_ap, 3 * fwhm_ap),
                    "stddev": (1e-5, fwhm_ap * gaussian_fwhm_to_sigma),
                },
            )
            fitted = fitter(g_init, local, y_cropped)
            center_pix_new[jj] = fitted.mean.value + peak_pix
            fwhm_tmp.append(fitted.fwhm)

        trace = np.vstack((trace, center_pix_new))
        trace_fwhm.append(fwhm_tmp)

    if len(xpos) == 0:
        raise RuntimeError("No valid trace points found. Check `bound`, `nspec`, and SNR.")

    return np.array(xpos), trace[1:, :].T, np.array(trace_fwhm).T

def trace_fit(xpos, trace, nspec = 1, sigma = 2.0, n_iter = 5, deg = 4):
    """Fit trace centers with a Chebyshev polynomial and sigma-clip outliers."""
    xpos = np.asarray(xpos)
    trace = np.asarray(trace)

    params, masks = [], []
    for i in range(nspec):
        p_trace = chebfit(xpos, trace[i], deg=deg)
        for _ in range(n_iter):
            res_mask = sigma_clip(trace[i] - chebval(xpos, p_trace), sigma=sigma).mask
            p_trace = chebfit(xpos[~res_mask], trace[i][~res_mask], deg=deg)
        params.append(p_trace)
        masks.append(res_mask)
    return params, masks

def trace_profile(
    data, nspec = 1, bound = (0, Ncol), plot = True, fit = False, label = None,
    sigma = 2.0, n_iter = 5, deg = 4, fwhm_ap = FWHM_AP, step_ap = step_AP):
    
    nrow, ncol = data.shape
    pixel = np.arange(ncol)
    
    # trace spectrum
    xpos, trace, fwhm = trace_spectrum(data, nspec=nspec, bound=bound, fwhm_ap=fwhm_ap, step_ap=step_ap)
    
    params = None
    masks = None
    if fit:
        params, masks = trace_fit(xpos, trace, nspec=nspec, sigma=sigma, n_iter=n_iter, deg=deg)

    if plot:
        gs = GridSpec(nrows=2, ncols=1, height_ratios=[3, 1])
        plt.figure(figsize=(10, 5))

        plt.subplot(gs[0])
        if label:
            plt.title(label)
        for i in range(nspec):
            plt.plot(xpos, trace[i] - bound[0], marker="+", ls="None")
        plt.imshow(data[bound[0] : bound[1]], aspect="auto", origin="lower")

        plt.subplot(gs[1])
        for i in range(nspec):
            plt.plot(xpos, trace[i], marker="+", ls="None")
            if fit and params is not None and masks is not None:
                plt.plot(pixel, chebval(pixel, params[i]), color="k", lw=1)
                plt.plot(xpos[masks[i]], trace[i][masks[i]], marker="x", ls="None", color="r")
        plt.xlabel("Dispersion Axis")
        plt.ylabel("Spatial Axis")
        plt.xlim(0, ncol)
        plt.tight_layout()

    if fit:
        return xpos, trace, fwhm, params
    return xpos, trace, fwhm

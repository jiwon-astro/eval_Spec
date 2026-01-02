import numpy as np

def gaussian(x, A, mu, sigma):
    """1D Gaussian."""
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

def Gaussian2D(xy, A, x0, y0, sx, sy):
    """Axis-aligned 2D Gaussian (no rotation)."""
    x, y = xy
    g = A * np.exp(-(((x - x0) ** 2) / (2 * sx**2) + ((y - y0) ** 2) / (2 * sy**2)))
    return g.ravel()

def Gaussian2D_tilt(xy, A, x0, y0, sx, sy, theta):
    """Rotated 2D Gaussian."""
    x, y = xy
    a = np.cos(theta) ** 2 / (2 * sx**2) + np.sin(theta) ** 2 / (2 * sy**2)
    b = -np.sin(2 * theta) / (4 * sx**2) + np.sin(2 * theta) / (4 * sy**2)
    c = np.sin(theta) ** 2 / (2 * sx**2) + np.cos(theta) ** 2 / (2 * sy**2)
    g = A * np.exp(-(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2))
    return g.ravel()

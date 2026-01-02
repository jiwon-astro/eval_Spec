#=======================
# Unit helpers.
#=======================
# Internal convention: wavelength arrays are in **nm**.
# If a file/header provides Å, convert with `angstrom_to_nm`.

import numpy as np

WAVELENGTH_UNIT = "nm"

def angstrom_to_nm(x):
    """Convert wavelength(s) in Å to nm."""
    return np.asarray(x, dtype=float) * 0.1

def nm_to_angstrom(x):
    """Convert wavelength(s) in nm to Å."""
    return np.asarray(x, dtype=float) * 10.0
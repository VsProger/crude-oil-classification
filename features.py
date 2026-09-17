"""
Feature definitions shared by training and serving.

This module exists so that the transformer pickled inside model.pkl resolves to
a stable import path. Defining it in the training script instead would pickle a
reference to __main__ and break loading anywhere else.
"""

import numpy as np
import pandas as pd

# Density and API are excluded on purpose: the label is thresholded API gravity
# and API is a closed-form function of density, so either one leaks the answer.
FEATURES = [
    "viscosity", "pour_point", "flash_point",
    "saturates", "aromatics", "resins", "asphaltenes", "sulfur",
]

CLASSES = ["heavy", "medium", "light"]


def log_viscosity(X):
    """Viscosity spans seven orders of magnitude; model it on a log scale."""
    X = pd.DataFrame(X, columns=FEATURES).copy()
    X["viscosity"] = np.log10(X["viscosity"].clip(lower=1e-3))
    return X

import numpy as np
from scipy.interpolate import CubicSpline


def interpolate_data(x, y, ratio=15, log=False, num_points=None):
    """
    Returns:
        X, Y(X)
    """
    if log:
        if num_points == None:
            num_points = len(x) * ratio
        X = np.geomspace(x.min(), x.max(), num_points)
        _Y = CubicSpline(x, y)
        return X, _Y(X)

    X = np.linspace(x.min(), x.max(), len(x) * ratio)
    _Y = CubicSpline(x, y)

    return X, _Y(X)

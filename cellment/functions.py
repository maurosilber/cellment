import numpy as np
from scipy import ndimage, stats, interpolate


def normalize_gradient(grad):
    """Normalizes a gradient vector.

    Parameters
    ----------
    grad : numpy.array
        The input gradient.

    Returns
    -------
    numpy.array
        Normalized gradient.

    """
    norm_grad = np.linalg.norm(grad, axis=0)
    cond = norm_grad > 0
    grad[:, cond] = grad[:, cond] / norm_grad[cond]
    return grad


def normalized_gradient(image):
    """Calculates the normalized gradient of an image.

    Parameters
    ----------
    image : numpy.array
        Input image.

    Returns
    -------
    numpy.array
        The normalized gradient of the image.
    """
    grad = np.array(np.gradient(image.astype(float)))
    grad = normalize_gradient(grad)
    return grad


def silver_mountain_operator(input, sigma, size):
    """Applies the Silver Mountain Operator (SMO) to the input.

    Parameters
    ----------
    input : numpy.array
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    size : int or tuple of int
        Averaging window size.

    Returns
    -------
    numpy.array
    """
    size = ndimage._ni_support._normalize_sequence(size, input.ndim)
    input = ndimage.gaussian_filter(input.astype(float), sigma=sigma)
    norm_grad = normalized_gradient(input)
    sliding_mean = ndimage.uniform_filter(norm_grad, size=(1, *size))
    mod = np.linalg.norm(sliding_mean, axis=0)
    return mod


class Empiric_RV(stats.rv_continuous):
    """Returns a random variable from an array of observations.

    data : numpy.array
        Array of observations. If it is multidimensional, it will be flattened.
    """
    def __init__(self, data):
        super().__init__()
        self.x = np.linspace(0, 1, data.size, endpoint=False)
        self.y = np.sort(data, axis=None)
        kwargs = {'copy': False, 'assume_sorted': True}
        self._cdf = interpolate.interp1d(self.y, self.x, bounds_error=False, fill_value=(0., 1.), **kwargs)
        self._ppf = interpolate.interp1d(self.x, self.y, **kwargs)

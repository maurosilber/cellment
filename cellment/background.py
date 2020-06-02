import numpy as np

from .functions import HistogramRV, smo
from .functions import smo_rv as _smo_rv


def smo_mask(image, sigma, size, threshold=0.1, smo_rv=None):
    """Returns the mask of (some) background noise.

    Parameters
    ----------
    image : numpy.ma.MaskedArray
        Image. If there are saturated pixels, they should be masked.
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel.
    size : int or sequence of int
        Averaging window size.
    threshold : float
        Percentile value [0, 1] for the SMO distribution.
    smo_rv : HistogramRV, optional
        Distribution of SMO. If not given, it is computed.
        It saves time to precompute it if this function is
        called multiple times with fixed sigma and size.

    Returns
    -------
    mask : numpy.array

    Notes
    -----
    Sigma and size are scale parameters, and should be less than the typical cell size.
    """
    image = np.ma.asarray(image)
    smo_image = smo(image, sigma, size)
    if smo_rv is None:
        smo_rv = _smo_rv(image.ndim, sigma, size)
    threshold = smo_rv.ppf(threshold)
    return (smo_image < threshold) & ~image.mask


def bg_rv(image, sigma, size, threshold=0.1, smo_rv=None):
    """Returns the distribution of background noise.

    Use self.median() to get the median value,
    or self.ppf(percentile) to calculate any other desired value.

    Parameters
    ----------
    image : numpy.array or numpy.ma.MaskedArray
        Image. If there are saturated pixels, they should be masked.
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel.
    size : int or tuple of int
        Averaging window size.
    threshold : float
        Percentile value [0, 1] for the SMO distribution.
    smo_rv : HistogramRV, optional
        Distribution of SMO. If not given, it is computed.
        It saves time to precompute it if this function is
        called multiple times with fixed sigma and size.

    Returns
    -------
    HistogramRV
        Subclass of scipy.stats.rv_histogram.

    Notes
    -----
    Sigma and size are scale parameters, and should be less than the typical cell size.
    """
    mask = smo_mask(image, sigma, size, threshold=threshold, smo_rv=smo_rv)
    bg = image[mask]
    return HistogramRV.from_data(bg)

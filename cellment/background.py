import numpy as np

from .functions import silver_mountain_operator, HistogramRV


def smo_rv(im_shape, sigma, size):
    """Generates a random variable of the SMO operator for a given sigma and size.

    Parameters
    ----------
    im_shape : tuple
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    size : int or tuple of int
        Averaging window size.

    Returns
    -------
    HistogramRV
        Subclass of scipy.stats.rv_continuous.
    """
    im = np.random.normal(size=im_shape)
    smo = silver_mountain_operator(im, sigma, size)
    return HistogramRV(smo)


def smo_mask(im, sigma, size, threshold=0.1):
    """Returns the mask of (some) background noise.

    Parameters
    ----------
    im : numpy.array
        Image
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    size : int or tuple of int
        Averaging window parameter.
    threshold : float
        Percentile value [0, 1] for the SMO distribution.

    Returns
    -------
    HistogramRV
        Subclass of scipy.stats.rv_continuous.

    Notes
    -----
    Sigma and size are scale parameters, and should be less than the typical cell size.
    """
    smo = silver_mountain_operator(im, sigma, size)
    threshold = smo_rv(im.shape, sigma, size).ppf(threshold)
    return smo < threshold


def bg_rv(im, sigma, size, threshold=0.1):
    """Returns the distribution of background noise.

    Use self.median() to get the median value,
    or self.ppf(percentile) to calculate any other desired value.

    Parameters
    ----------
    im : numpy.array
        Image
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    size : int or tuple of int
        Averaging window parameter.
    threshold : float
        Percentile value [0, 1] for the SMO distribution.

    Returns
    -------
    HistogramRV
        Subclass of scipy.stats.rv_continuous.

    Notes
    -----
    Sigma and size are scale parameters, and should be less than the typical cell size.
    """
    mask = smo_mask(im, sigma, size, threshold=threshold)
    return HistogramRV(im[mask])

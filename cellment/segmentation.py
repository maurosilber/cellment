from skimage import morphology

from . import tracking


def threshold_segmentation(image, threshold, size=10):
    """Segment an image with a global threshold and binary opening.

    Parameters
    ----------
    image : numpy.array
        Image to segment.
    threshold : scalar or numpy.array
    size : int
        Size of a disk-shaped element to perform a binary opening.

    Returns
    -------
    numpy.array
        An array of labels.
    """
    mask = image > threshold
    mask = morphology.binary_opening(mask, morphology.disk(size), out=mask)
    return morphology.label(mask.astype(int))


def multi_threshold_segmentation(image, thresholds, bg_rv=None, size=10):
    """Performs a segmentation at multiple thresholds to try to split merged cells.

    A base segmentation is performed at the first threshold,
    and subsequent segmentations used to split merged cells.

    Parameters
    ----------
    image : numpy.array
        Image to segment.
    thresholds : Sequence of scalar or numpy.arrays
        If a bg_rv is supplied, thresholds are percentiles of the
        background distribution.
    bg_rv : stats.rv_continuous, optional
        Background random variable. Needs to implement ppf method.
    size : int
        Size of a disk-shaped element to perform a binary opening.

    Returns
    -------
    numpy.array
        An array of labels.
    """
    if bg_rv is not None:
        thresholds = map(bg_rv.ppf, thresholds)
    labels = tuple(threshold_segmentation(image, t, size=size) for t in thresholds)  # Segmentation
    graph = tracking.Labels_graph.from_labels_stack(labels)
    tracking.split_nodes(labels, graph, tuple(image for _ in range(len(labels))), 1000, 0.05)
    return labels[0]

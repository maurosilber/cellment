import networkx as nx
import numpy as np
from skimage import segmentation
from skimage.filters import gaussian
from skimage.morphology import binary_dilation, square


def color_mapping(labeled_images):
    graph = nx.Graph()
    for labeled_image in labeled_images:
        mask = labeled_image == 0  # Background mask

        # Expand labels to cover background area
        watershed = gaussian((~mask).astype(float), 10)
        watershed = segmentation.watershed(-watershed, labeled_image)

        # Find connected labels
        for label in np.unique(watershed):
            mask = watershed == label
            mask = binary_dilation(mask, square(2))  # Expand mask
            for connected_label in np.unique(watershed[mask]):
                graph.add_edge(label, connected_label)

    return nx.coloring.greedy_color(graph)


def color_labeled(labeled_image, color_map):
    out = labeled_image.copy()
    for label, color in color_map.items():
        out[labeled_image == label] = color + 1  # 0 is background

    return out

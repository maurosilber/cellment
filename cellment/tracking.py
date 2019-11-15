from collections import namedtuple
import functools

from more_itertools import pairwise
import numpy as np
from scipy import ndimage
from skimage import morphology
from skimage.segmentation import watershed
import networkx as nx

Node = namedtuple('Node', ['time', 'label'])


def count_labels(label_image, exclude_zero=False):
    """Calculate area for each label."""
    start = 1 if exclude_zero else 0
    counts = np.bincount(label_image.flat)[start:]
    labels = np.nonzero(counts)[0]
    return dict(zip(labels + start, counts[labels]))


def intersection_between_labels(label_image_1, label_image_2, labels_1=None, exclude_zero=False):
    """Calculates intersections of labels_stack between two pairs of labeled images.

    Parameters
    ----------
    label_image_1, label_image_2
        Image of labels_stack.

    Returns
    -------
    dict
        For each label_1, there's a dictionary of labels_2 and the size of their intersection.
        {x_label_1: {y_label_1: intersection_size, ...}, ...}
    """
    start = 1 if exclude_zero else 0

    if labels_1 is None:
        labels_1 = np.arange(start, 1 + label_image_1.max())  # Faster than np.unique
    else:
        if exclude_zero:
            labels_1 = labels_1[np.nonzero(labels_1)]

    if labels_1.size == 0:
        return {}

    hist = ndimage.labeled_comprehension(label_image_2, label_image_1, labels_1, np.bincount, object, None)

    out = {}
    for label_1, h in zip(labels_1, hist):
        if h is None:
            continue
        ix = np.nonzero(h[start:])[0] + start
        if ix.size > 0:
            out[label_1] = dict(zip(ix, h[ix]))
    return out


class Labels_graph(nx.DiGraph):
    """Time-like DAG of connected labels_stack.
    Each Node is a namedtuple of (time, label).

    Parameters
    ----------
    labels_stack : numpy.array
        Axis 0 corresponds to the temporal dimension.
    """

    @classmethod
    def from_labels_stack(cls, labels_stack):
        graph = cls()
        # Nodes
        for time, label_image in enumerate(labels_stack):
            graph.add_nodes_from_label_image(time, label_image)
        # Edges
        for (time_a, label_image_a), (time_b, label_image_b) in pairwise(enumerate(labels_stack)):
            graph.add_edges_from_label_image(time_a, label_image_a, time_b, label_image_b)

        return graph

    def add_nodes_from_label_image(self, t, label_image):
        for label, area in count_labels(label_image, exclude_zero=True).items():
            node_props = {'area': area}
            self.add_node(Node(t, label), **node_props)

    def add_edges_from_label_image(self, time_a, label_image_a, time_b, label_image_b):
        for label_a, intersections in intersection_between_labels(label_image_a, label_image_b,
                                                                  exclude_zero=True).items():
            for label_b, intersection in intersections.items():
                edge_props = {'area': intersection}
                self.add_edge(Node(time_a, label_a), Node(time_b, label_b), **edge_props)

    def times(self):
        """Returns all time indexes."""
        return (node.time for node in self.nodes)

    def is_timelike_chain(self):
        """Returns True if the graph has only one node per time."""
        return len(set(self.times())) == len(self)

    def __eq__(self, other):
        def equal(x, y):
            return x == y

        return nx.is_isomorphic(self, other, node_match=equal, edge_match=equal)


def decompose(graph):
    """Decomposes a graph into disconnected components."""
    cc = nx.connected_components(graph.to_undirected(as_view=True))
    return (graph.subgraph(c) for c in cc)


def merged_nodes(graph):
    for node, deg in graph.in_degree:
        if deg > 1:
            return node, 'in'

    for node, deg in graph.out_degree:
        if deg > 1:
            return node, 'out'

    return None, None


def split_nodes(labels_stack, graph, image_stack, area_threshold, edge_threshold):
    while True:
        trim_nodes(graph, labels_stack, area_threshold)
        trim_edges(graph, edge_threshold)
        node, mode = merged_nodes(graph)
        if node is None:
            break
        else:
            split_node(graph, labels_stack, node, mode, image_stack[node.time])


def trim_nodes(graph, labels_stack, area_threshold):
    nodes_to_remove = [node for node, area in graph.nodes(data='area') if area < area_threshold]
    for node in nodes_to_remove:
        graph.remove_node(node)
        labels_stack[node.time][labels_stack[node.time] == node.label] = 0


def trim_edges(graph, edge_threshold):
    def ratio(edge):
        return graph.edges[edge]['area'] / graph.nodes[edge[1]]['area']

    for node in filter(lambda x: graph.out_degree(x) > 1, graph.nodes):
        edges = sorted(graph.edges(node), key=ratio, reverse=True)
        edges_to_remove = list(filter(lambda edge: ratio(edge) < edge_threshold, edges[1:]))
        graph.remove_edges_from(edges_to_remove)


def split_node(graph, labels_stack, node, mode, image):
    """Graph and labels_stack are modified in-place."""
    if mode == 'in':
        nodes = list(graph.predecessors(node))
        label_image_2 = labels_stack[node.time - 1]
    elif mode == 'out':
        nodes = list(graph.successors(node))
        label_image_2 = labels_stack[node.time + 1]
    else:
        raise NotImplementedError('Mode must be either "in" or "out"')

    # Split with watershed
    watershed_labels, s = split_with_watershed(node, image, labels_stack[node.time], nodes, label_image_2)

    # Relabel label_stack
    for new_label, n in enumerate(nodes, labels_stack[node.time].max() + 1):
        labels_stack[node.time][s][watershed_labels == n.label] = new_label

    # Modify graph
    graph.remove_node(node)
    graph.add_nodes_from_label_image(node.time, labels_stack[node.time])
    if node.time > 0:
        graph.add_edges_from_label_image(node.time - 1, labels_stack[node.time - 1],
                                         node.time, labels_stack[node.time])
    if node.time + 1 < len(labels_stack):
        graph.add_edges_from_label_image(node.time, labels_stack[node.time],
                                         node.time + 1, labels_stack[node.time + 1])


def split_with_watershed(node, image, label_image, nodes, label_image_2):
    mask = label_image == node.label
    full_mask = functools.reduce(np.logical_or, (label_image_2 == n.label for n in nodes), mask)
    s = ndimage.find_objects(full_mask)[0]  # Find slice of mask

    image, mask = image[s], mask[s]
    markers = np.zeros_like(mask, int)
    for n in nodes:
        label_mask = label_image_2[s] == n.label
        for size in (20, 15, 10, 5, 1):
            eroded_mask = morphology.binary_erosion(label_mask, morphology.square(size))
            eroded_mask &= mask
            if eroded_mask.any():
                break
        markers[eroded_mask] = n.label

    watershed_labels = watershed(-image, markers=markers, mask=mask)
    return watershed_labels, s

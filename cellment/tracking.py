import numpy as np
import networkx as nx
from skimage.measure import regionprops


def labels_graph(labels_stack) -> nx.DiGraph:
    """Returns a time-like DAG of connected labels.

    Node names are (time, label).

    Parameters
    ----------
    labels_stack : numpy.array
        Axis 0 corresponds to the temporal dimension.

    Returns
    -------
    networkx.DiGraph
    """
    graph = nx.DiGraph()

    # Nodes
    for t, labels in enumerate(labels_stack):
        for prop in regionprops(labels):
            if prop.label == 0:  # Exclude background
                continue
            node_props = {'area': prop.area}
            graph.add_node((t, prop.label), **node_props)

    # Edges
    for t, labels in enumerate(labels_stack[1:], 1):
        prev_labels = labels_stack[t - 1]
        for i in np.unique(prev_labels):
            if i == 0:  # Exclude background
                continue
            intersection = labels[prev_labels == i]
            for j in np.unique(intersection):
                if j == 0:  # Exclude background
                    continue
                edge_props = {'area': (intersection == j).sum()}
                graph.add_edge((t - 1, i), (t, j), **edge_props)

    return graph


def decompose(graph):
    """Decomposes a graph into disconnected components."""
    cc = nx.connected_components(graph.to_undirected(as_view=True))
    return (graph.subgraph(c) for c in cc)


def time_indices(graph):
    """Returns all time indexes."""
    return (t for t, n in graph.nodes)


def nodes_at_time(graph, time):
    """Returns all nodes existing at the given time index."""
    return ((t, n) for t, n in graph.nodes if t == time)


def get_timelike_chains(graph):
    """Returns timelike chains from graph.

    Only returns timelike chains where the number of coexisting labels,
    that is, having the same time index, equals the maximum number of
    coexisting labels.
    """
    count = np.bincount(list(time_indices(graph)))  # Number of coexisting labels
    times_most_elements = np.argwhere(count == count.max())[:, 0]
    time_chains = np.split(times_most_elements,
                           1 + np.argwhere(np.diff(times_most_elements) > 1)[:, 0])
    subgraphs = []
    for chain in time_chains:
        t0, t1 = chain[0], chain[-1]
        if t0 == t1:
            continue
        for t, n in nodes_at_time(graph, t0):
            chain = [(t, n)]
            while t < t1:
                t, n = next(graph.successors((t, n)))
                chain.append((t, n))
            subgraphs.append(graph.subgraph(chain))

    return subgraphs


def is_timelike_chain(graph):
    """Returns True if the graph has only one node per time."""
    return len(set(time_indices(graph))) == len(graph)

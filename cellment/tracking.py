import numpy as np
import networkx as nx
from skimage.measure import regionprops


def labels_graph(labels_stack) -> nx.DiGraph:
    """Returns a time-like DAG of connected labels.

    Parameters
    ----------
    labels_stack : numpy.array

    Returns
    -------
    networkx.DiGraph
    """
    graph = nx.DiGraph()

    # Nodes
    for t, labels in enumerate(labels_stack):
        for prop in regionprops(labels):
            if prop.label == 0:
                continue
            node_props = {'area': prop.area}
            graph.add_node((t, prop.label), **node_props)

    # Edges
    for t in range(len(labels_stack) - 1):
        prev_labels, labels = labels_stack[t:][:2]
        for i in np.unique(prev_labels):
            if i > 0:
                intersection = labels[prev_labels == i]
                for j in np.unique(intersection):
                    if j > 0:
                        edge_props = {'area': (intersection == j).sum()}
                        graph.add_edge((t, i), (t + 1, j), **edge_props)

    return graph


def decompose(graph):
    cc = nx.connected_components(graph.to_undirected(as_view=True))
    return (graph.subgraph(c) for c in cc)


def time_indices(graph):
    return (t for t, n in graph.nodes)


def nodes_at_time(graph, time):
    return ((t, n) for t, n in graph.nodes if t == time)


def get_chains(graph):
    count = np.bincount(list(time_indices(graph)))
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


def is_chain(graph):
    return len(set(time_indices(graph))) == len(graph)

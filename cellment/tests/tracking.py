import unittest

import numpy as np

from .. import tracking


class LabelsTest(unittest.TestCase):
    def test_counts(self):
        label_image = np.array([0, 0, 1, 1, 1, 2, 2])
        counts = {0: 2, 1: 3, 2: 2}
        self.assertDictEqual(
            tracking.count_labels(label_image, exclude_zero=False), counts
        )

        # Test excluding zero labels_stack
        counts.pop(0)
        self.assertDictEqual(
            tracking.count_labels(label_image, exclude_zero=True), counts
        )

    def test_intersections(self):
        label_image_1 = np.array([0, 0, 1, 1, 2, 2])
        label_image_2 = np.array([0, 3, 3, 4, 4, 0])
        intersections = {0: {0: 1, 3: 1}, 1: {3: 1, 4: 1}, 2: {0: 1, 4: 1}}
        self.assertDictEqual(
            tracking.intersection_between_labels(
                label_image_1, label_image_2, exclude_zero=False
            ),
            intersections,
        )

        # Test excluding zero labels_stack
        intersections.pop(0)
        for value in intersections.values():
            value.pop(0, None)
        self.assertDictEqual(
            tracking.intersection_between_labels(
                label_image_1, label_image_2, exclude_zero=True
            ),
            intersections,
        )


class MergedNodesTest(unittest.TestCase):
    def test_merged_nodes(self):
        merging_graph = tracking.Labels_graph()
        merging_graph.add_nodes_from(range(3))
        merging_graph.add_edges_from([(0, 2), (1, 2)])
        self.assertEqual(tracking.merged_nodes(merging_graph), (2, "in"))

        splitting_graph = tracking.Labels_graph()
        splitting_graph.add_nodes_from(range(3))
        splitting_graph.add_edges_from([(0, 1), (0, 2)])
        self.assertEqual(tracking.merged_nodes(splitting_graph), (0, "out"))

        graph = tracking.Labels_graph()
        graph.add_nodes_from(range(3))
        self.assertEqual(tracking.merged_nodes(graph), (None, None))


if __name__ == "__main__":
    unittest.main()

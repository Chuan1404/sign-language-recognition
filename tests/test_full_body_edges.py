import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

import config


GRAPH_PATH = Path(__file__).resolve().parents[1] / 'src/models/spatial_graph.py'


class FullBodyEdgesTests(unittest.TestCase):
    def check_layout(self, graph, removed, num_nodes):
        kept = [i for i in range(33) if i not in removed]
        count = len(kept)
        self.assertEqual(graph['remaining_nodes'], kept)
        pose_map = {old: new for new, old in enumerate(kept)}
        expected_pose = [(pose_map[a], pose_map[b]) for a, b in graph['_POSE_BONES']
                         if a in pose_map and b in pose_map]
        expected_hands = [(count + hand * config._N_HAND + a,
                           count + hand * config._N_HAND + b)
                          for hand in (0, 1) for a, b in graph['_HAND_BONES']]
        bridges = [(pose_map[elbow], count + side * config._N_HAND)
                   for side, elbow in enumerate((13, 14)) if elbow in pose_map]
        self.assertEqual(graph['FULL_BODY_EDGES'], expected_pose + expected_hands + bridges)
        self.assertEqual(graph['HAND_EDGES'],
                         [(hand * config._N_HAND + a, hand * config._N_HAND + b)
                          for hand in (0, 1) for a, b in graph['_HAND_BONES']])
        adjacency = graph['build_adjacency_from_edges'](graph['FULL_BODY_EDGES'], num_nodes)
        self.assertEqual(tuple(adjacency.shape), (num_nodes, num_nodes))
        for a, b in bridges + [(pose_map[0], pose_map[s]) for s in (11, 12)
                                if 0 in pose_map and s in pose_map]:
            self.assertEqual(adjacency[a, b].item(), 1)
            self.assertEqual(adjacency[b, a].item(), 1)
        for a, b in expected_hands:
            self.assertGreaterEqual(a, count)
            self.assertGreaterEqual(b, count)
            self.assertEqual(adjacency[a, b].item(), 1)
            self.assertEqual(adjacency[b, a].item(), 1)

    def test_current_pose_and_hand_offsets(self):
        self.check_layout(runpy.run_path(str(GRAPH_PATH)), config._REMOVE_POSE_IDX, config._NUM_NODE)

    def test_last_pose_node_is_counted_when_retained(self):
        removed = [i for i in range(33) if i not in (0, 9, 10, 32)]
        nodes = 4 + 2 * config._N_HAND
        with patch.object(config, '_REMOVE_POSE_IDX', removed), patch.object(config, '_NUM_NODE', nodes):
            graph = runpy.run_path(str(GRAPH_PATH))
        self.check_layout(graph, removed, nodes)


if __name__ == '__main__':
    unittest.main()

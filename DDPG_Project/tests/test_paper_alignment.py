"""Regression tests for the paper chapter-3 implementation."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from config import AppConfig
from algorithm.environment.cloud_fog_edge_env import CloudFogEdgeEnv
from algorithm.per_ddpg.ddpg_agent import PERDDPGAgent
from algorithm.runner.runner import PERDDPGRunner


class PaperAlignmentTests(unittest.TestCase):
    def make_config(self):
        config = AppConfig()
        config.env.edge_device_num = 6
        config.env.fog_node_num = 3
        config.env.max_episodes = 20
        config.env.seed = 42
        return config

    def test_algorithm_31_ranks_and_topological_sequence(self):
        env = CloudFogEdgeEnv(self.make_config().env)
        env.reset()
        self.assertTrue(all(task.rank > 0 for task in env.dag.subtasks))

        positions = {task_id: index for index, task_id in enumerate(env.list_seq)}
        for predecessor, successor, _ in env.dag.edges:
            self.assertLess(positions[predecessor], positions[successor])

    def test_dag_edge_count_matches_configured_density(self):
        config = self.make_config()
        config.env.edge_device_num = 8
        config.env.dag_edge_prob = 0.25
        env = CloudFogEdgeEnv(config.env)
        env.reset()
        possible_edges = 8 * 7 // 2
        self.assertEqual(
            len(env.dag.edges),
            round(config.env.dag_edge_prob * possible_edges),
        )

    def test_sparse_dag_can_have_multiple_entries(self):
        config = self.make_config()
        config.env.edge_device_num = 8
        config.env.dag_edge_prob = 0.0
        env = CloudFogEdgeEnv(config.env)
        env.reset()
        self.assertEqual(env.dag.entry_tasks, list(range(8)))

    def test_invalid_dag_density_is_rejected(self):
        config = self.make_config()
        config.env.dag_edge_prob = 1.1
        with self.assertRaisesRegex(ValueError, "dag_edge_prob"):
            CloudFogEdgeEnv(config.env)

    def test_state_matches_equation_323_dimensions(self):
        env = CloudFogEdgeEnv(self.make_config().env)
        state = env.reset()
        self.assertEqual(len(state), 4 + len(env.all_node_ids) + env.M)

    def test_action_contains_only_cloud_fog_and_current_local(self):
        env = CloudFogEdgeEnv(self.make_config().env)
        env.reset()
        task_id = env.list_seq[0]
        self.assertEqual(
            env.get_feasible_node_ids(),
            ["cloud_0", "fog_1", "fog_2", "fog_3", f"edge_{task_id}"],
        )
        self.assertEqual(env.get_action_dim(), env.M + 3)

    def test_average_fog_time_uses_best_uplink_rate(self):
        env = CloudFogEdgeEnv(self.make_config().env)
        env.reset()
        task = env.dag.subtasks[0]
        best_rate = float(np.max(env.tx_rates[0]))
        expected = task.data_size / best_rate + task.total_cycles / (
            env.config.fog_cpu * 1e9
        )
        self.assertAlmostEqual(task.T_fog, expected, places=10)

    def test_invalid_action_shape_and_nan_are_rejected(self):
        env = CloudFogEdgeEnv(self.make_config().env)
        env.reset()
        with self.assertRaisesRegex(ValueError, "action length"):
            env.step(np.zeros(env.get_action_dim() + 1))

        bad_action = np.zeros(env.get_action_dim())
        bad_action[0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            env.step(bad_action)

    def test_step_reports_requested_and_effective_resource(self):
        env = CloudFogEdgeEnv(self.make_config().env)
        env.reset()
        action = np.zeros(env.get_action_dim(), dtype=np.float32)
        action[-2] = 1.0
        action[-1] = 0.01
        env.step(action)

        info = env.last_step_info
        for key in [
            "requested_resource_ratio", "requested_cpu",
            "allocated_cpu", "effective_resource_ratio",
        ]:
            self.assertIn(key, info)
        self.assertEqual(
            info["resource_ratio"], info["requested_resource_ratio"]
        )

    def test_resource_reservations_respect_capacity(self):
        env = CloudFogEdgeEnv(self.make_config().env)
        env.reset()
        for _ in range(env.N):
            action = np.zeros(env.get_action_dim(), dtype=np.float32)
            action[0] = 1.0
            action[-1] = 0.5
            _, reward, _ = env.step(action)
            self.assertTrue(np.isfinite(reward))

        node = env.nodes["cloud_0"]
        self.assertEqual(len(node.reservations), env.N)
        for reservation in node.reservations:
            self.assertGreaterEqual(
                reservation["cpu"], env.dag.subtasks[reservation["task_id"]].min_cpu
            )
        event_times = sorted(
            {item["start"] for item in node.reservations}
            | {item["finish"] for item in node.reservations}
        )
        for timestamp in event_times[:-1]:
            used = node.cpu_total - node.available_cpu_at(timestamp)
            self.assertLessEqual(used, node.cpu_total + 1e-9)

    def test_zero_node_probabilities_fall_back_to_uniform(self):
        config = self.make_config()
        env = CloudFogEdgeEnv(config.env)
        agent = PERDDPGAgent(
            env.get_state_dim(), env.get_action_dim(), config.ddpg, config.env
        )
        action = np.zeros(env.get_action_dim(), dtype=np.float32)
        normalized = agent._normalize_exploratory_action(action)
        np.testing.assert_allclose(
            normalized[:-1], 1.0 / (env.M + 2), rtol=1e-6
        )

    def test_checkpoint_metadata_rejects_incompatible_schema(self):
        config = self.make_config()
        env = CloudFogEdgeEnv(config.env)
        agent = PERDDPGAgent(
            env.get_state_dim(), env.get_action_dim(), config.ddpg, config.env
        )
        with tempfile.TemporaryDirectory() as path:
            agent.save(path)
            metadata_path = Path(path) / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["schema_version"] = 1
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema"):
                agent.load(path)

    def test_frontend_displays_requested_and_effective_resource_ratio(self):
        template = (
            Path(__file__).resolve().parents[1] / "web" / "templates" / "index.html"
        ).read_text(encoding="utf-8")
        self.assertIn("requested_resource_ratio", template)
        self.assertIn("effective_resource_ratio", template)

    def test_real_training_crosses_batch_boundary(self):
        np.random.seed(42)
        torch.manual_seed(42)
        config = self.make_config()
        env = CloudFogEdgeEnv(config.env)
        agent = PERDDPGAgent(
            env.get_state_dim(), env.get_action_dim(), config.ddpg, config.env
        )
        runner = PERDDPGRunner(env, agent, config.env, config.ddpg)
        before = [parameter.detach().clone() for parameter in agent.actor.parameters()]

        for _ in range(12):
            runner.start_episode()
            while True:
                detail = runner.run_step(explore=True)
                if detail is None or detail.get("done"):
                    break

        self.assertGreaterEqual(len(agent.buffer), config.ddpg.batch_size)
        self.assertGreater(agent.train_step, 0)
        self.assertTrue(
            any(
                not torch.equal(old, new)
                for old, new in zip(before, agent.actor.parameters())
            )
        )
        self.assertTrue(np.isfinite(agent.buffer.tree.total_priority))


if __name__ == "__main__":
    unittest.main(verbosity=2)

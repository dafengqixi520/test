"""论文实验运行器的回归测试。"""
import unittest

from config import AppConfig

from experiments.paper_experiments import (
    aggregate_experiment_runs,
    aggregate_seed_results,
    ddpg_variants,
    measured_edge_density,
    parse_seeds,
)


class PaperExperimentTests(unittest.TestCase):
    def test_ddpg_ablation_matrix(self):
        self.assertEqual(
            ddpg_variants(),
            {
                "DDPG": (False, False),
                "DDPG+PER": (True, False),
                "DDPG+LD": (False, True),
                "PER-DDPG": (True, True),
            },
        )


    def test_seed_override_and_aggregate_statistics(self):
        self.assertEqual(parse_seeds(7, "42,43"), [7])
        self.assertEqual(parse_seeds(None, "42,43"), [42, 43])

        result = aggregate_seed_results({
            42: {"mean_makespan": 2.0, "samples": [1.5, 2.5]},
            43: {"mean_makespan": 4.0, "samples": [3.5, 4.5]},
        })
        self.assertEqual(result["seed_count"], 2)
        self.assertEqual(result["mean_makespan"], 3.0)
        self.assertEqual(result["std_makespan"], 1.0)
        self.assertEqual(result["samples"], [1.5, 2.5, 3.5, 4.5])


    def test_comparison_and_sweep_runs_are_aggregated_by_seed(self):
        comparison = aggregate_experiment_runs({
            42: {"PER-DDPG": {"mean_makespan": 2.0, "samples": [2.0]}},
            43: {"PER-DDPG": {"mean_makespan": 4.0, "samples": [4.0]}},
        })
        self.assertEqual(comparison["PER-DDPG"]["seed_count"], 2)
        self.assertEqual(comparison["PER-DDPG"]["mean_makespan"], 3.0)

        sweep = aggregate_experiment_runs({
            42: [{
                "parameter": 0.1,
                "configured_density": 0.1,
                "measured_density": 0.1,
                "algorithms": {
                    "PER-DDPG": {"mean_makespan": 2.0, "samples": [2.0]}
                },
            }],
            43: [{
                "parameter": 0.1,
                "configured_density": 0.1,
                "measured_density": 0.1,
                "algorithms": {
                    "PER-DDPG": {"mean_makespan": 4.0, "samples": [4.0]}
                },
            }],
        })
        self.assertEqual(sweep[0]["configured_density"], 0.1)
        self.assertEqual(
            sweep[0]["algorithms"]["PER-DDPG"]["seed_count"], 2
        )


    def test_measured_edge_density_uses_generated_edge_count(self):
        config = AppConfig()
        config.env.edge_device_num = 8
        config.env.dag_edge_prob = 0.25
        self.assertEqual(measured_edge_density(config), 0.25)


if __name__ == "__main__":
    unittest.main(verbosity=2)

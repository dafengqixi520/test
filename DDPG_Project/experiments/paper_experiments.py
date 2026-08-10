"""Run reproducible chapter-3 experiments without hard-coded paper results.

The output is generated from the local implementation. It must not be presented
as the paper authors' original data because the paper does not provide its raw
DAG samples, complete random seeds, or training logs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import AppConfig
from algorithm.environment.cloud_fog_edge_env import CloudFogEdgeEnv
from algorithm.per_ddpg.ddpg_agent import PERDDPGAgent
from algorithm.per_ddpg.noise import FixedGaussianNoise
from algorithm.runner.runner import PERDDPGRunner


RESOURCE_LEVELS = (0.25, 0.5, 0.75, 1.0)


def ddpg_variants():
    """返回用于独立验证 PER 与 LD-Noise 贡献的固定消融矩阵。"""
    return {
        "DDPG": (False, False),
        "DDPG+PER": (True, False),
        "DDPG+LD": (False, True),
        "PER-DDPG": (True, True),
    }


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def action_vector(env, node_id, ratio):
    action = np.zeros(env.get_action_dim(), dtype=np.float32)
    action[env.get_feasible_node_ids().index(node_id)] = 1.0
    action[-1] = float(ratio)
    return action


def run_rule_policy(config, policy, episodes, seed):
    cfg = copy.deepcopy(config)
    cfg.env.seed = seed
    env = CloudFogEdgeEnv(cfg.env)
    makespans = []
    rng = np.random.RandomState(seed)
    for _ in range(episodes):
        env.reset()
        while env.current_task_idx < env.N:
            task_id = env.list_seq[env.current_task_idx]
            if policy == "FLE":
                action = action_vector(env, f"edge_{task_id}", 1.0)
            elif policy == "RO":
                candidates = ["cloud_0", *[n.id for n in env.fog_nodes], f"edge_{task_id}"]
                action = action_vector(env, rng.choice(candidates), rng.uniform(0.25, 1.0))
            else:
                raise ValueError(policy)
            env.step(action)
        makespans.append(max(env.task_finish_times.values()))
    return summarize(makespans)


def train_ddpg(config, episodes, prioritized, adaptive_noise, seed):
    seed_everything(seed)
    env = CloudFogEdgeEnv(config.env)
    agent = PERDDPGAgent(env.get_state_dim(), env.get_action_dim(), config.ddpg, config.env)
    if not prioritized:
        agent.buffer.alpha = 0.0
        agent.buffer.beta = 0.0
        agent.buffer.beta_increment = 0.0
    if not adaptive_noise:
        agent.noise = FixedGaussianNoise(env.get_action_dim(), sigma=0.2)
    runner = PERDDPGRunner(env, agent, config.env, config.ddpg)
    for _ in range(episodes):
        runner.start_episode()
        while True:
            detail = runner.run_step(explore=True)
            if detail is None or detail.get("done"):
                break
    return agent, {
        "reward_curve": runner.episode_rewards,
        "makespan_curve": runner.episode_makespans,
        "train_updates": agent.train_step,
    }


def evaluate_agent(config, agent, episodes, seed):
    cfg = copy.deepcopy(config)
    cfg.env.seed = seed
    env = CloudFogEdgeEnv(cfg.env)
    makespans = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action, _ = agent.select_action(state, explore=False)
            state, _, done = env.step(action)
        makespans.append(max(env.task_finish_times.values()))
    return summarize(makespans)


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=400):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state):
        return self.net(state)


def decode_dqn_action(env, index):
    level_count = len(RESOURCE_LEVELS)
    node_option, level = divmod(int(index), level_count)
    task_id = env.list_seq[env.current_task_idx]
    if node_option == 0:
        node_id = "cloud_0"
    elif node_option <= env.M:
        node_id = f"fog_{node_option}"
    else:
        node_id = f"edge_{task_id}"
    return action_vector(env, node_id, RESOURCE_LEVELS[level])


def train_dqn(config, episodes, seed):
    seed_everything(seed)
    env = CloudFogEdgeEnv(config.env)
    action_dim = (env.M + 2) * len(RESOURCE_LEVELS)
    online = DQN(env.get_state_dim(), action_dim)
    target = copy.deepcopy(online)
    optimizer = torch.optim.Adam(online.parameters(), lr=0.001)
    replay = deque(maxlen=config.ddpg.buffer_size)
    batch_size = config.ddpg.batch_size
    gamma = config.ddpg.gamma
    updates = 0
    reward_curve, makespan_curve = [], []

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        epsilon = max(0.05, 1.0 - episode / max(1, episodes - 1))
        while not done:
            if np.random.random() < epsilon:
                index = np.random.randint(action_dim)
            else:
                with torch.no_grad():
                    index = int(online(torch.tensor(state).float().unsqueeze(0)).argmax(1))
            action = decode_dqn_action(env, index)
            next_state, reward, done = env.step(action)
            replay.append((state, index, reward, next_state, done))
            state = next_state
            episode_reward += reward

            if len(replay) >= batch_size:
                batch = random.sample(replay, batch_size)
                states = torch.tensor(np.array([x[0] for x in batch])).float()
                actions = torch.tensor([x[1] for x in batch]).long().unsqueeze(1)
                rewards = torch.tensor([x[2] for x in batch]).float().unsqueeze(1)
                next_states = torch.tensor(np.array([x[3] for x in batch])).float()
                dones = torch.tensor([x[4] for x in batch]).float().unsqueeze(1)
                current = online(states).gather(1, actions)
                with torch.no_grad():
                    expected = rewards + gamma * target(next_states).max(1, keepdim=True)[0] * (1 - dones)
                loss = F.mse_loss(current, expected)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), 1.0)
                optimizer.step()
                updates += 1
                if updates % 100 == 0:
                    target.load_state_dict(online.state_dict())

        reward_curve.append(episode_reward)
        makespan_curve.append(max(env.task_finish_times.values()))
    return online, {
        "reward_curve": reward_curve,
        "makespan_curve": makespan_curve,
        "train_updates": updates,
    }


def evaluate_dqn(config, network, episodes, seed):
    cfg = copy.deepcopy(config)
    cfg.env.seed = seed
    env = CloudFogEdgeEnv(cfg.env)
    makespans = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                index = int(network(torch.tensor(state).float().unsqueeze(0)).argmax(1))
            state, _, done = env.step(decode_dqn_action(env, index))
        makespans.append(max(env.task_finish_times.values()))
    return summarize(makespans)


def summarize(values):
    values = [float(v) for v in values]
    return {
        "mean_makespan": float(np.mean(values)),
        "std_makespan": float(np.std(values)),
        "samples": values,
    }


def parse_seeds(seed, seeds_text):
    """解析训练种子；显式单种子优先于种子列表。"""
    if seed is not None:
        return [int(seed)]
    values = [item.strip() for item in str(seeds_text).split(",")]
    seeds = [int(item) for item in values if item]
    if not seeds:
        raise ValueError("at least one seed is required")
    return list(dict.fromkeys(seeds))


def aggregate_seed_results(results_by_seed):
    """以独立训练种子的评估均值为统计样本。"""
    if not results_by_seed:
        raise ValueError("results_by_seed must not be empty")
    ordered = sorted(results_by_seed.items())
    seed_means = [float(result["mean_makespan"]) for _, result in ordered]
    samples = []
    for _, result in ordered:
        samples.extend(float(value) for value in result.get("samples", []))
    return {
        "mean_makespan": float(np.mean(seed_means)),
        "std_makespan": float(np.std(seed_means)),
        "seed_count": len(ordered),
        "samples": samples,
        "seed_results": {str(seed): result for seed, result in ordered},
    }

def aggregate_experiment_runs(runs_by_seed):
    """聚合比较实验或参数扫描的独立训练种子结果。"""
    if not runs_by_seed:
        raise ValueError("runs_by_seed must not be empty")
    ordered = sorted(runs_by_seed.items())
    first = ordered[0][1]
    if isinstance(first, dict):
        return {
            algorithm: aggregate_seed_results({
                seed: run[algorithm] for seed, run in ordered
            })
            for algorithm in first
        }

    output = []
    for point_index, first_group in enumerate(first):
        group = {
            key: value for key, value in first_group.items()
            if key != "algorithms"
        }
        group["algorithms"] = {
            algorithm: aggregate_seed_results({
                seed: run[point_index]["algorithms"][algorithm]
                for seed, run in ordered
            })
            for algorithm in first_group["algorithms"]
        }
        output.append(group)
    return output

def measured_edge_density(config):
    """生成一个配置对应的 DAG 并返回其实际边密度。"""
    env = CloudFogEdgeEnv(copy.deepcopy(config.env))
    env.reset()
    return float(env.dag.edge_density)

def compare_algorithms(config, train_episodes, eval_episodes, seed):
    result = {
        "FLE": run_rule_policy(config, "FLE", eval_episodes, seed + 10000),
        "RO": run_rule_policy(config, "RO", eval_episodes, seed + 10000),
    }
    for name, (prioritized, adaptive_noise) in ddpg_variants().items():
        agent, training = train_ddpg(
            config, train_episodes, prioritized, adaptive_noise, seed
        )
        result[name] = evaluate_agent(
            config, agent, eval_episodes, seed + 10000
        )
        result[name]["training"] = training

    dqn, dqn_train = train_dqn(config, train_episodes, seed)
    result["DQN"] = evaluate_dqn(config, dqn, eval_episodes, seed + 10000)
    result["DQN"]["training"] = dqn_train
    return result

def experiment_points(name, profile):
    full = profile == "full"
    return {
        "task-count": [4, 8, 12, 16, 20, 24] if full else [4, 8],
        "data-size": [350, 400, 450, 500, 550, 600] if full else [350, 600],
        "fog-count": [3, 4, 5, 6, 7, 8] if full else [3, 5],
        "edge-density": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] if full else [0.1, 0.6],
        "learning-rate": [
            (0.0001, 0.0002), (0.001, 0.002), (0.01, 0.02)
        ] if full else [(0.0001, 0.0002), (0.001, 0.002)],
        "batch-size": [16, 64, 256] if full else [16, 64],
        "gamma": [0.4, 0.7, 0.99] if full else [0.4, 0.99],
    }[name]


def run_experiment(name, profile, seed):
    train_episodes = 500 if profile == "full" else 20
    eval_episodes = 20 if profile == "full" else 3
    config = AppConfig()
    config.env.seed = seed
    if name == "comparison":
        return compare_algorithms(config, train_episodes, eval_episodes, seed)

    output = []
    for value in experiment_points(name, profile):
        point = copy.deepcopy(config)
        if name == "task-count":
            point.env.edge_device_num = value
        elif name == "data-size":
            point.env.task_size_range = (value, value)
        elif name == "fog-count":
            point.env.fog_node_num = value
        elif name == "edge-density":
            point.env.dag_edge_prob = value
        elif name == "learning-rate":
            point.ddpg.actor_lr, point.ddpg.critic_lr = value
        elif name == "batch-size":
            point.ddpg.batch_size = value
        elif name == "gamma":
            point.ddpg.gamma = value

        if name in {"learning-rate", "batch-size", "gamma"}:
            agent, training = train_ddpg(point, train_episodes, True, True, seed)
            result = evaluate_agent(point, agent, eval_episodes, seed + 10000)
            result["training"] = training
            algorithms = {"PER-DDPG": result}
        else:
            algorithms = compare_algorithms(
                point, train_episodes, eval_episodes, seed
            )
        group = {
            "parameter": list(value) if isinstance(value, tuple) else value,
            "algorithms": algorithms,
        }
        if name == "edge-density":
            group["configured_density"] = float(value)
            group["measured_density"] = measured_edge_density(point)
        output.append(group)
    return output


def run_multi_seed_experiment(name, profile, seeds):
    """对每个训练种子独立运行，再按种子级均值汇总。"""
    runs = {}
    for seed in seeds:
        try:
            runs[int(seed)] = run_experiment(name, profile, int(seed))
        except Exception as exc:
            raise RuntimeError(
                f"experiment {name} failed for seed {seed}"
            ) from exc
    return aggregate_experiment_runs(runs)


def write_outputs(experiment, profile, seeds, data):
    output_dir = Path("output/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_dir / f"{experiment}_{profile}_{stamp}"
    payload = {
        "provenance": "locally generated; not original paper data",
        "experiment": experiment,
        "profile": profile,
        "seeds": [int(seed) for seed in seeds],
        "data": data,
    }
    base.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows = []
    groups = data if isinstance(data, list) else [{"parameter": "default", "algorithms": data}]
    for group in groups:
        for algorithm, result in group["algorithms"].items():
            rows.append({
                "parameter": group["parameter"],
                "algorithm": algorithm,
                "mean_makespan": result["mean_makespan"],
                "std_makespan": result["std_makespan"],
                "seed_count": result.get("seed_count", 1),
                "configured_density": group.get("configured_density", ""),
                "measured_density": group.get("measured_density", ""),
            })
    with base.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return str(base.with_suffix(".json")), str(base.with_suffix(".csv"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=[
        "comparison", "task-count", "data-size", "fog-count", "edge-density",
        "learning-rate", "batch-size", "gamma"
    ], default="comparison")
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--seed", type=int, default=None,
                        help="单个训练种子；显式提供时覆盖--seeds")
    parser.add_argument("--seeds", default="42,43,44,45,46",
                        help="逗号分隔的独立训练种子")
    args = parser.parse_args()
    seeds = parse_seeds(args.seed, args.seeds)
    data = run_multi_seed_experiment(args.experiment, args.profile, seeds)
    json_path, csv_path = write_outputs(
        args.experiment, args.profile, seeds, data
    )
    print(json.dumps({"json": json_path, "csv": csv_path}, ensure_ascii=False))


if __name__ == "__main__":
    main()

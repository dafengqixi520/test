"""Industrial IoT PER-DDPG scheduling demo.

Run:
    python industrial_iot_per_ddpg_demo.py

Open:
    http://127.0.0.1:8765

This is a dependency-light implementation of the paper's chapter-3 idea:
1. Algorithm 3.1 converts a DAG into List_seq with rank, urgency, and W_i.
2. A PER-DDPG-style scheduler then chooses cloud/fog/edge execution nodes and
   continuous CPU resource ratios for each task.

The neural-network version in algorithm/ uses Torch and Flask. This file keeps
the same paper rules runnable in a plain Python + NumPy environment.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np


# ----------------------------- Scenario model -----------------------------


@dataclass
class Task:
    id: int
    name: str
    station: str
    source_edge: str
    data_kb: float
    density_cycles_per_kb: float
    min_cpu: float
    predecessors: List[int] = field(default_factory=list)
    successors: List[int] = field(default_factory=list)
    rank: float = 0.0
    urgency: float = 0.0
    priority: float = 0.0
    avg_time: float = 0.0
    ready_time: float = 0.0
    assigned_node: str = ""
    allocated_cpu: float = 0.0
    start_time: float = 0.0
    finish_time: float = 0.0

    @property
    def data_bits(self) -> float:
        return self.data_kb * 1024 * 8

    @property
    def cycles(self) -> float:
        return self.data_kb * self.density_cycles_per_kb


@dataclass
class Node:
    id: str
    label: str
    layer: str
    cpu: float
    base_latency: float
    available_at: float = 0.0
    busy_time: float = 0.0
    task_count: int = 0

    def reset(self) -> None:
        self.available_at = 0.0
        self.busy_time = 0.0
        self.task_count = 0


class IndustrialScenario:
    """A small industrial IoT DAG from shop-floor sensing to control action."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.nodes: Dict[str, Node] = {
            "edge_plc": Node("edge_plc", "PLC edge controller", "edge", 1.2, 0.004),
            "edge_vision": Node("edge_vision", "Vision edge box", "edge", 1.6, 0.005),
            "edge_robot": Node("edge_robot", "Robot arm gateway", "edge", 1.1, 0.004),
            "fog_line_a": Node("fog_line_a", "Line-A fog server", "fog", 6.0, 0.014),
            "fog_quality": Node("fog_quality", "Quality fog server", "fog", 7.0, 0.016),
            "cloud_factory": Node("cloud_factory", "Factory cloud", "cloud", 22.0, 0.055),
        }
        self.tasks = self._build_tasks()
        self.edges = self._build_edges()
        self._connect_edges()
        self._compute_average_times()

    def clone(self) -> "IndustrialScenario":
        return copy.deepcopy(self)

    def reset_runtime(self) -> None:
        for node in self.nodes.values():
            node.reset()
        for task in self.tasks:
            task.ready_time = 0.0
            task.assigned_node = ""
            task.allocated_cpu = 0.0
            task.start_time = 0.0
            task.finish_time = 0.0

    def _build_tasks(self) -> List[Task]:
        return [
            Task(0, "Vibration signal windowing", "CNC spindle", "edge_plc", 420, 1.7e6, 0.20),
            Task(1, "Thermal image preprocessing", "Furnace camera", "edge_vision", 580, 2.2e6, 0.25),
            Task(2, "Acoustic anomaly features", "Compressor", "edge_robot", 360, 1.5e6, 0.18),
            Task(3, "Surface defect CNN inference", "Inspection cell", "edge_vision", 760, 3.4e6, 0.45),
            Task(4, "Motor health diagnosis", "CNC spindle", "edge_plc", 500, 2.8e6, 0.35),
            Task(5, "Multi-sensor fusion", "Line coordinator", "edge_robot", 650, 2.6e6, 0.40),
            Task(6, "Quality drift prediction", "Quality station", "edge_vision", 540, 2.9e6, 0.35),
            Task(7, "Maintenance risk scoring", "MES gateway", "edge_plc", 460, 2.4e6, 0.30),
            Task(8, "Adaptive process tuning", "Robot arm", "edge_robot", 390, 1.9e6, 0.25),
            Task(9, "Alarm and report commit", "Factory dashboard", "edge_plc", 300, 1.2e6, 0.16),
        ]

    def _build_edges(self) -> List[Tuple[int, int, float]]:
        return [
            (0, 4, 0.28), (0, 5, 0.18),
            (1, 3, 0.35), (1, 5, 0.16),
            (2, 5, 0.24),
            (3, 6, 0.30),
            (4, 7, 0.25),
            (5, 6, 0.20), (5, 8, 0.22),
            (6, 9, 0.18), (7, 9, 0.18), (8, 9, 0.16),
        ]

    def _connect_edges(self) -> None:
        for left, right, _ratio in self.edges:
            self.tasks[left].successors.append(right)
            self.tasks[right].predecessors.append(left)

    def _compute_average_times(self) -> None:
        fog_cpu = np.mean([n.cpu for n in self.nodes.values() if n.layer == "fog"])
        cloud_cpu = self.nodes["cloud_factory"].cpu
        for task in self.tasks:
            local_cpu = self.nodes[task.source_edge].cpu
            local = task.cycles / (local_cpu * 1e9)
            fog = self.upload_time(task, "fog_line_a") + task.cycles / (fog_cpu * 1e9)
            cloud = self.upload_time(task, "cloud_factory") + task.cycles / (cloud_cpu * 1e9)
            task.avg_time = (local + fog + cloud) / 3.0

    def upload_rate(self, task: Task, node_id: str) -> float:
        node = self.nodes[node_id]
        if node.layer == "edge":
            return float("inf") if node.id == task.source_edge else 80e6
        if node.layer == "fog":
            # Deterministic pseudo distance makes each station a little different.
            station_factor = 1.0 + (task.id % 3) * 0.22
            return 42e6 / station_factor
        return 18e6

    def upload_time(self, task: Task, node_id: str) -> float:
        rate = self.upload_rate(task, node_id)
        if math.isinf(rate):
            return 0.0
        return task.data_bits / rate + self.nodes[node_id].base_latency

    def dependency_data_bits(self, pred_id: int, task_id: int) -> float:
        pred = self.tasks[pred_id]
        for left, right, ratio in self.edges:
            if left == pred_id and right == task_id:
                return pred.data_bits * ratio
        return 0.0

    def inter_node_transfer(self, from_node: str, to_node: str, data_bits: float) -> float:
        if not from_node or from_node == to_node:
            return 0.0
        a = self.nodes[from_node].layer
        b = self.nodes[to_node].layer
        if a == "edge" and b == "edge":
            rate, latency = 65e6, 0.006
        elif "cloud" in (a, b):
            rate, latency = 22e6, 0.040
        elif "fog" in (a, b):
            rate, latency = 90e6, 0.010
        else:
            rate, latency = 50e6, 0.014
        return data_bits / rate + latency

    def candidate_nodes(self, task: Task) -> List[str]:
        return [task.source_edge, "fog_line_a", "fog_quality", "cloud_factory"]


# --------------------------- Paper algorithm 3.1 ---------------------------


def build_list_seq(scenario: IndustrialScenario, deadline: float, omega: float) -> List[int]:
    tasks = scenario.tasks

    def compute_rank(task_id: int) -> float:
        task = tasks[task_id]
        if task.rank > 0:
            return task.rank
        succ_rank = max((compute_rank(sid) for sid in task.successors), default=0.0)
        task.rank = task.avg_time + succ_rank
        return task.rank

    for task in tasks:
        task.rank = 0.0
    for task in tasks:
        compute_rank(task.id)

    rank_max = max(task.rank for task in tasks) or 1.0
    ready = {task.id for task in tasks if not task.predecessors}
    completed = set()
    list_seq: List[int] = []
    virtual_now = 0.0

    while ready:
        urgency_values = {}
        for tid in ready:
            slack = deadline - virtual_now - tasks[tid].rank
            # The paper assigns highest urgency when slack is insufficient.
            urgency_values[tid] = (1.0 / slack) if slack > 1e-6 else (1.0 + abs(slack) * 10)
            tasks[tid].urgency = urgency_values[tid]

        urgency_max = max(urgency_values.values()) or 1.0
        best_tid = None
        best_priority = -1.0
        for tid in ready:
            task = tasks[tid]
            rank_norm = task.rank / rank_max
            urgency_norm = task.urgency / urgency_max
            task.priority = omega * rank_norm + (1.0 - omega) * urgency_norm
            if task.priority > best_priority:
                best_tid = tid
                best_priority = task.priority

        assert best_tid is not None
        list_seq.append(best_tid)
        ready.remove(best_tid)
        completed.add(best_tid)
        virtual_now += tasks[best_tid].avg_time

        for sid in tasks[best_tid].successors:
            if all(pid in completed for pid in tasks[sid].predecessors):
                ready.add(sid)

    return list_seq


# -------------------------- PER-DDPG-style policy --------------------------


class PrioritizedReplay:
    def __init__(self, capacity: int = 512, alpha: float = 0.6, epsilon: float = 0.01):
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon
        self.items: List[dict] = []

    def add(self, item: dict, td_error: float) -> None:
        item = dict(item)
        item["priority"] = (abs(td_error) + self.epsilon) ** self.alpha
        if len(self.items) >= self.capacity:
            self.items.pop(0)
        self.items.append(item)

    def sample(self, count: int = 8) -> List[dict]:
        if not self.items:
            return []
        weights = np.array([item["priority"] for item in self.items], dtype=float)
        weights = weights / weights.sum()
        count = min(count, len(self.items))
        indices = np.random.choice(len(self.items), size=count, replace=False, p=weights)
        return [self.items[int(idx)] for idx in indices]

    def top(self, count: int = 6) -> List[dict]:
        return sorted(self.items, key=lambda item: item["priority"], reverse=True)[:count]


class PaperAlignedScheduler:
    def __init__(self, scenario: IndustrialScenario, deadline: float, omega: float, seed: int):
        self.scenario = scenario
        self.deadline = deadline
        self.omega = omega
        self.rng = np.random.RandomState(seed)
        self.gamma = 0.99
        self.replay = PrioritizedReplay()

    def train(self, episodes: int = 16) -> dict:
        best_result = None
        metrics = []
        for episode in range(episodes):
            eta = 0.45 * (1.0 - episode / max(episodes - 1, 1))
            result = self._run_episode(episode, eta)
            metrics.append({
                "episode": episode + 1,
                "makespan": result["makespan"],
                "reward": result["total_reward"],
                "eta": eta,
                "buffer": len(self.replay.items),
                "td_mean": result["td_mean"],
            })
            if best_result is None or result["makespan"] < best_result["makespan"]:
                best_result = result

        assert best_result is not None
        best_result["metrics"] = metrics
        best_result["per_top"] = self.replay.top()
        return best_result

    def _run_episode(self, episode: int, eta: float) -> dict:
        scenario = self.scenario.clone()
        scenario.reset_runtime()
        list_seq = build_list_seq(scenario, self.deadline, self.omega)
        finish_times: Dict[int, float] = {}
        schedule = []
        rewards = []
        td_errors = []

        for order, task_id in enumerate(list_seq, start=1):
            task = scenario.tasks[task_id]
            action = self._select_action(scenario, task, finish_times, eta)
            chosen_node = action["node_id"]
            outcome = self._apply_action(scenario, task, chosen_node, action["resource_ratio"], finish_times)
            finish_times[task_id] = outcome["finish"]
            makespan = max(finish_times.values())
            reward = -makespan
            baseline = min(action["candidate_finishes"].values())
            td_error = abs(outcome["finish"] - baseline) + max(0.0, makespan - self.deadline)
            rewards.append(reward)
            td_errors.append(td_error)

            transition = {
                "episode": episode + 1,
                "task": task.name,
                "task_id": task.id,
                "node": chosen_node,
                "reward": reward,
                "finish": outcome["finish"],
                "td_error": td_error,
            }
            self.replay.add(transition, td_error)

            schedule.append({
                "order": order,
                "task_id": task.id,
                "task_name": task.name,
                "station": task.station,
                "node_id": chosen_node,
                "node_label": scenario.nodes[chosen_node].label,
                "node_layer": scenario.nodes[chosen_node].layer,
                "resource_ratio": round(action["resource_ratio"], 3),
                "allocated_cpu": round(outcome["allocated_cpu"], 3),
                "ready_time": round(outcome["ready_time"], 4),
                "input_transfer": round(outcome["input_transfer"], 4),
                "dependency_transfer": round(outcome["dependency_transfer"], 4),
                "wait_time": round(outcome["wait_time"], 4),
                "exec_time": round(outcome["exec_time"], 4),
                "start": round(outcome["start"], 4),
                "finish": round(outcome["finish"], 4),
                "reward": round(reward, 4),
                "td_error": round(td_error, 4),
                "action_probs": action["probabilities"],
            })

        makespan = max(finish_times.values()) if finish_times else 0.0
        return {
            "list_seq": list_seq,
            "tasks": [self._task_dict(task) for task in scenario.tasks],
            "edges": [
                {
                    "from": left,
                    "to": right,
                    "data_kb": round(scenario.dependency_data_bits(left, right) / 8 / 1024, 1),
                }
                for left, right, _ in scenario.edges
            ],
            "nodes": [self._node_dict(node, makespan) for node in scenario.nodes.values()],
            "schedule": schedule,
            "makespan": round(makespan, 4),
            "deadline": self.deadline,
            "deadline_hit": makespan <= self.deadline,
            "total_reward": round(float(sum(rewards)), 4),
            "td_mean": round(float(np.mean(td_errors)) if td_errors else 0.0, 4),
            "layer_counts": self._layer_counts(schedule),
            "omega": self.omega,
        }

    def _select_action(self, scenario: IndustrialScenario, task: Task, finish_times: Dict[int, float], eta: float) -> dict:
        candidates = scenario.candidate_nodes(task)
        finish_scores = {}
        raw_scores = []
        urgency_norm = min(1.0, task.urgency / max((t.urgency for t in scenario.tasks), default=1.0))
        rank_norm = min(1.0, task.rank / max((t.rank for t in scenario.tasks), default=1.0))

        for node_id in candidates:
            ratio = self._resource_ratio(scenario.nodes[node_id], task, urgency_norm, eta=0.0)
            predicted = self._predict_finish(scenario, task, node_id, ratio, finish_times)
            layer_penalty = {"edge": 0.02, "fog": 0.0, "cloud": 0.045}[scenario.nodes[node_id].layer]
            # Critical tasks tolerate cloud transfer if it sharply shortens execution.
            score = predicted + layer_penalty * (1.0 - rank_norm)
            finish_scores[node_id] = predicted
            raw_scores.append(-score)

        logits = np.array(raw_scores, dtype=float)
        temperature = 0.12 + eta
        probs = np.exp((logits - logits.max()) / temperature)
        probs = probs / probs.sum()
        if eta > 0:
            noise = self.rng.normal(0, eta * 0.10, size=probs.shape)
            probs = np.clip(probs + noise, 0.001, None)
            probs = probs / probs.sum()
        node_id = candidates[int(np.argmax(probs))]
        resource_ratio = self._resource_ratio(scenario.nodes[node_id], task, urgency_norm, eta=eta)
        return {
            "node_id": node_id,
            "resource_ratio": resource_ratio,
            "candidate_finishes": finish_scores,
            "probabilities": [
                {"node": nid, "prob": round(float(prob), 4), "finish": round(finish_scores[nid], 4)}
                for nid, prob in zip(candidates, probs)
            ],
        }

    def _resource_ratio(self, node: Node, task: Task, urgency_norm: float, eta: float) -> float:
        base = 0.32 + 0.42 * urgency_norm
        if node.layer == "cloud":
            base -= 0.10
        elif node.layer == "edge":
            base += 0.08
        noisy = base + float(self.rng.normal(0, eta * 0.08))
        min_ratio = min(0.95, task.min_cpu / node.cpu)
        return float(np.clip(max(noisy, min_ratio), 0.08, 0.95))

    def _predict_finish(self, scenario: IndustrialScenario, task: Task, node_id: str, ratio: float, finish_times: Dict[int, float]) -> float:
        node = scenario.nodes[node_id]
        ready_time, dep_transfer = self._ready_time_for_node(scenario, task, node_id, finish_times)
        input_transfer = scenario.upload_time(task, node_id)
        arrival = ready_time + dep_transfer + input_transfer
        allocated = max(task.min_cpu, node.cpu * ratio)
        exec_time = task.cycles / (allocated * 1e9)
        start = max(arrival, node.available_at)
        return start + exec_time

    def _apply_action(self, scenario: IndustrialScenario, task: Task, node_id: str, ratio: float, finish_times: Dict[int, float]) -> dict:
        node = scenario.nodes[node_id]
        ready_time, dep_transfer = self._ready_time_for_node(scenario, task, node_id, finish_times)
        input_transfer = scenario.upload_time(task, node_id)
        arrival = ready_time + dep_transfer + input_transfer
        allocated = max(task.min_cpu, min(node.cpu * ratio, node.cpu))
        exec_time = task.cycles / (allocated * 1e9)
        wait_time = max(0.0, node.available_at - arrival)
        start = max(arrival, node.available_at)
        finish = start + exec_time

        node.available_at = finish
        node.busy_time += exec_time
        node.task_count += 1
        task.ready_time = ready_time
        task.assigned_node = node_id
        task.allocated_cpu = allocated
        task.start_time = start
        task.finish_time = finish
        return {
            "ready_time": ready_time,
            "dependency_transfer": dep_transfer,
            "input_transfer": input_transfer,
            "allocated_cpu": allocated,
            "exec_time": exec_time,
            "wait_time": wait_time,
            "start": start,
            "finish": finish,
        }

    def _ready_time_for_node(self, scenario: IndustrialScenario, task: Task, node_id: str, finish_times: Dict[int, float]) -> Tuple[float, float]:
        ready = 0.0
        dep_transfer = 0.0
        for pid in task.predecessors:
            pred = scenario.tasks[pid]
            pred_finish = finish_times.get(pid, pred.finish_time)
            data_bits = scenario.dependency_data_bits(pid, task.id)
            transfer = scenario.inter_node_transfer(pred.assigned_node, node_id, data_bits)
            ready = max(ready, pred_finish)
            dep_transfer = max(dep_transfer, transfer)
        return ready, dep_transfer

    def _task_dict(self, task: Task) -> dict:
        return {
            "id": task.id,
            "name": task.name,
            "station": task.station,
            "source_edge": task.source_edge,
            "data_kb": task.data_kb,
            "density_mcycles_per_kb": round(task.density_cycles_per_kb / 1e6, 2),
            "min_cpu": task.min_cpu,
            "predecessors": task.predecessors,
            "successors": task.successors,
            "rank": round(task.rank, 4),
            "urgency": round(task.urgency, 4),
            "priority": round(task.priority, 4),
            "assigned_node": task.assigned_node,
            "allocated_cpu": round(task.allocated_cpu, 4),
            "finish_time": round(task.finish_time, 4),
        }

    def _node_dict(self, node: Node, makespan: float) -> dict:
        utilization = node.busy_time / makespan if makespan > 0 else 0.0
        return {
            "id": node.id,
            "label": node.label,
            "layer": node.layer,
            "cpu": node.cpu,
            "busy_time": round(node.busy_time, 4),
            "utilization": round(min(utilization, 1.0), 4),
            "task_count": node.task_count,
            "available_at": round(node.available_at, 4),
        }

    def _layer_counts(self, schedule: List[dict]) -> dict:
        counts = {"edge": 0, "fog": 0, "cloud": 0}
        for item in schedule:
            counts[item["node_layer"]] += 1
        return counts


# ------------------------------- Web server -------------------------------


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Industrial IoT PER-DDPG Demo</title>
<style>
:root{--bg:#101214;--panel:#f6f4ef;--ink:#1d2327;--muted:#687076;--line:#d6d0c2;--edge:#2f855a;--fog:#b7791f;--cloud:#2b6cb0;--warn:#c53030;--accent:#5b5f97}
*{box-sizing:border-box}body{margin:0;font-family:Segoe UI,Microsoft YaHei,sans-serif;background:var(--bg);color:var(--ink)}
.top{background:#e8e1d2;border-bottom:4px solid #2f855a;padding:14px 22px;display:flex;align-items:center;justify-content:space-between;gap:16px}
.top h1{font-size:20px;margin:0}.top p{margin:3px 0 0;color:#50575c;font-size:12px}.status{font-weight:700;color:#2f855a}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;background:#262a2d;color:#fff;padding:10px 22px}
label{font-size:12px;color:#d7dbe0}input{accent-color:#2f855a}button{border:0;background:#2f855a;color:white;border-radius:6px;padding:8px 13px;font-weight:700;cursor:pointer}button:hover{filter:brightness(1.08)}
.wrap{padding:18px 22px;display:grid;grid-template-columns:1.1fr .9fr;gap:16px}.full{grid-column:1 / -1}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px;box-shadow:0 8px 24px rgba(0,0,0,.18)}
.panel h2{font-size:15px;margin:0 0 10px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.card{background:white;border:1px solid var(--line);border-radius:7px;padding:10px}.k{font-size:11px;color:var(--muted)}.v{font-size:24px;font-weight:800;margin-top:4px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}.bars{height:170px;display:flex;align-items:end;gap:6px;border-left:1px solid var(--line);border-bottom:1px solid var(--line);padding:8px}.bar{flex:1;background:var(--accent);min-height:2px;border-radius:4px 4px 0 0;position:relative}.bar span{position:absolute;bottom:100%;left:50%;transform:translateX(-50%);font-size:9px;color:var(--muted);white-space:nowrap}
.dag{height:430px;background:#fff;border:1px solid var(--line);border-radius:8px;overflow:auto}.dag-canvas{position:relative;min-width:900px;height:100%;background:linear-gradient(to right,transparent calc(25% - 1px),#eee9df calc(25% - 1px),#eee9df 25%,transparent 25%,transparent calc(50% - 1px),#eee9df calc(50% - 1px),#eee9df 50%,transparent 50%,transparent calc(75% - 1px),#eee9df calc(75% - 1px),#eee9df 75%,transparent 75%)}.dag-layer{position:absolute;top:10px;transform:translateX(-50%);font-size:10px;font-weight:700;color:#777064;letter-spacing:0;text-transform:uppercase}.task-node{position:absolute;width:158px;height:74px;border-radius:7px;border:2px solid #444;background:#fff;padding:8px 10px;font-size:11px;line-height:1.25;z-index:2;box-shadow:0 2px 7px rgba(29,35,39,.08);overflow:hidden}.task-node strong{display:block;font-size:12px;margin-bottom:2px}.task-node.done{background:#effaf2}.task-node.cloud{border-color:var(--cloud)}.task-node.fog{border-color:var(--fog)}.task-node.edge{border-color:var(--edge)}.task-node .k{display:block;margin-top:3px;white-space:nowrap}.task-port{position:absolute;top:50%;width:8px;height:8px;border-radius:50%;background:#fff;border:2px solid currentColor;transform:translateY(-50%)}.task-port.in{left:-5px}.task-port.out{right:-5px}svg.edges{position:absolute;inset:0;z-index:1;overflow:visible}.dag-edge{fill:none;stroke:#938b7c;stroke-width:1.5;stroke-linejoin:round}.dag-edge:hover{stroke:#2f855a;stroke-width:2.5}
table{width:100%;border-collapse:collapse;background:white;border-radius:8px;overflow:hidden}th,td{border-bottom:1px solid var(--line);padding:7px 8px;text-align:left;font-size:12px}th{background:#e8e1d2;color:#333}.pill{display:inline-block;padding:2px 7px;border-radius:999px;color:#fff;font-size:11px}.edge{background:var(--edge)}.fog{background:var(--fog)}.cloud{background:var(--cloud)}
.seq{display:flex;gap:7px;flex-wrap:wrap}.seq span{background:white;border:1px solid var(--line);border-radius:999px;padding:5px 8px;font-size:12px}.legend{display:flex;gap:10px;flex-wrap:wrap;color:var(--muted);font-size:12px}.small{font-size:12px;color:var(--muted);line-height:1.7}.prob{display:flex;height:8px;border-radius:999px;overflow:hidden;background:#ddd}.prob i{display:block;height:100%}
@media(max-width:980px){.wrap{grid-template-columns:1fr}.cards{grid-template-columns:repeat(2,1fr)}.grid2{grid-template-columns:1fr}.dag{height:430px}}
</style>
</head>
<body>
<div class="top">
  <div>
    <h1>工业物联网 PER-DDPG 云雾边任务卸载演示</h1>
    <p>DAG排序 List_seq + 节点选择 A_t + 连续资源比例 F_t + PER高价值经验采样</p>
  </div>
  <div class="status" id="status">Ready</div>
</div>
<div class="controls">
  <button onclick="run()">运行算法</button>
  <label>拓扑/紧迫权重 ω <input id="omega" type="range" min="0" max="1" step="0.05" value="0.55" oninput="omegaText.textContent=this.value"></label><b id="omegaText">0.55</b>
  <label>截止时间(s) <input id="deadline" type="range" min="1.5" max="8" step="0.1" value="3.8" oninput="deadlineText.textContent=this.value"></label><b id="deadlineText">3.8</b>
  <label>训练回合 <input id="episodes" type="number" min="4" max="80" value="18" style="width:70px"></label>
  <label>随机种子 <input id="seed" type="number" value="42" style="width:70px"></label>
</div>
<main class="wrap">
  <section class="panel full">
    <div class="cards">
      <div class="card"><div class="k">最优总完工时延 max(FT_i)</div><div class="v" id="makespan">--</div></div>
      <div class="card"><div class="k">是否满足全局 deadline</div><div class="v" id="hit">--</div></div>
      <div class="card"><div class="k">PER经验池重点样本</div><div class="v" id="buffer">--</div></div>
      <div class="card"><div class="k">云/雾/边任务分布</div><div class="v" id="layers">--</div></div>
    </div>
  </section>
  <section class="panel">
    <h2>算法3.1 输出: DAG -> List_seq</h2>
    <div id="seq" class="seq"></div>
    <p class="small">W_i = ω * rank_i / rank_max + (1 - ω) * urgency_i / urgency_max。列表中的顺序就是后续 PER-DDPG 智能体逐个决策的任务顺序。</p>
    <div id="dag" class="dag"></div>
  </section>
  <section class="panel">
    <h2>训练收敛与探索噪声</h2>
    <div class="grid2">
      <div><div class="k">Makespan 越低越好</div><div id="chartMakespan" class="bars"></div></div>
      <div><div class="k">LD-Noise η 线性衰减</div><div id="chartEta" class="bars"></div></div>
    </div>
    <h2 style="margin-top:14px">高优先级经验样本</h2>
    <div id="per" class="small"></div>
  </section>
  <section class="panel full">
    <h2>任务卸载与资源分配结果</h2>
    <table>
      <thead><tr><th>#</th><th>工业任务</th><th>执行层</th><th>节点</th><th>CPU比例 F_t</th><th>开始/完成(s)</th><th>传输/等待/执行(s)</th><th>动作概率 A_t</th></tr></thead>
      <tbody id="schedule"></tbody>
    </table>
  </section>
  <section class="panel full">
    <h2>节点负载</h2>
    <table>
      <thead><tr><th>节点</th><th>层级</th><th>CPU(GHz)</th><th>任务数</th><th>忙碌时间</th><th>利用率</th></tr></thead>
      <tbody id="nodes"></tbody>
    </table>
  </section>
</main>
<script>
const colors={edge:'#2f855a',fog:'#b7791f',cloud:'#2b6cb0'};
async function run(){
  status.textContent='Running...';
  const url=`/api/run?omega=${omega.value}&deadline=${deadline.value}&episodes=${episodes.value}&seed=${seed.value}`;
  const res=await fetch(url);
  const data=await res.json();
  render(data);
  status.textContent='Done';
}
function render(data){
  makespan.textContent=data.makespan.toFixed(3)+'s';
  hit.textContent=data.deadline_hit?'达标':'超时';
  hit.style.color=data.deadline_hit?'#2f855a':'#c53030';
  buffer.textContent=data.per_top.length;
  layers.textContent=`边${data.layer_counts.edge}/雾${data.layer_counts.fog}/云${data.layer_counts.cloud}`;
  seq.innerHTML=data.list_seq.map((id,i)=>`<span>${i+1}. I${id} ${data.tasks[id].name}<br><b>W=${data.tasks[id].priority.toFixed(3)}</b></span>`).join('');
  renderDag(data);
  renderBars(chartMakespan,data.metrics.map(m=>m.makespan),'#5b5f97');
  renderBars(chartEta,data.metrics.map(m=>m.eta),'#b7791f');
  renderSchedule(data);
  renderNodes(data);
  per.innerHTML=data.per_top.map(x=>`I${x.task_id} ${x.task} -> ${x.node}, TD Error=${x.td_error.toFixed(4)}, priority=${x.priority.toFixed(4)}`).join('<br>');
}
function renderBars(el,values,color){
  const max=Math.max(...values,0.001), min=Math.min(...values,0);
  el.innerHTML=values.map((v,i)=>{
    const h=8+150*(v-min)/(max-min+0.0001);
    return `<div class="bar" style="height:${h}px;background:${color}"><span>${i%5===0?Number(v).toFixed(2):''}</span></div>`;
  }).join('');
}
function renderDag(data){
  const nodeW=158,nodeH=74,canvasH=428,padX=28,top=42,bottom=18;
  const predecessors=new Map(data.tasks.map(t=>[t.id,[]]));
  data.edges.forEach(e=>predecessors.get(e.to).push(e.from));
  const levels=new Map();
  const levelOf=id=>{
    if(levels.has(id)) return levels.get(id);
    const preds=predecessors.get(id)||[];
    const level=preds.length?1+Math.max(...preds.map(levelOf)):0;
    levels.set(id,level);
    return level;
  };
  data.tasks.forEach(t=>levelOf(t.id));
  const levelCount=Math.max(...levels.values())+1;
  const canvasW=Math.max(900,dag.clientWidth-2);
  const columnGap=(canvasW-nodeW-2*padX)/Math.max(1,levelCount-1);
  const groups=Array.from({length:levelCount},()=>[]);
  data.tasks.forEach(t=>groups[levels.get(t.id)].push(t));
  groups.forEach(group=>group.sort((a,b)=>b.priority-a.priority||a.id-b.id));
  const positions={};
  groups.forEach((group,level)=>{
    const usable=canvasH-top-bottom;
    const gap=usable/group.length;
    group.forEach((task,index)=>{
      positions[task.id]=[padX+level*columnGap,top+gap*(index+.5)-nodeH/2];
    });
  });
  const edgeLines=data.edges.map((e,index)=>{
    const a=positions[e.from],b=positions[e.to];
    const x1=a[0]+nodeW,y1=a[1]+nodeH/2,x2=b[0],y2=b[1]+nodeH/2;
    const laneOffset=((index%5)-2)*3;
    const mid=x1+(x2-x1)/2+laneOffset;
    return `<path class="dag-edge" d="M ${x1} ${y1} H ${mid} V ${y2} H ${x2-7}" marker-end="url(#arr)"><title>I${e.from} -> I${e.to}: ${e.data_kb} KB</title></path>`;
  }).join('');
  const nodes=data.tasks.map(t=>{
    const p=positions[t.id], layer=(data.schedule.find(s=>s.task_id===t.id)||{}).node_layer||'edge';
    return `<div class="task-node done ${layer}" style="left:${p[0]}px;top:${p[1]}px"><i class="task-port in"></i><i class="task-port out"></i><strong>I${t.id}</strong>${t.name}<span class="k">rank ${t.rank.toFixed(2)} / W ${t.priority.toFixed(2)}</span></div>`;
  }).join('');
  const layerNames=['Input sensing','Feature / diagnosis','Decision fusion','Control output'];
  const headers=groups.map((_,level)=>`<div class="dag-layer" style="left:${padX+level*columnGap+nodeW/2}px">${layerNames[level]||`Stage ${level+1}`}</div>`).join('');
  dag.innerHTML=`<div class="dag-canvas" style="width:${canvasW}px;height:${canvasH}px"><svg class="edges" width="${canvasW}" height="${canvasH}" viewBox="0 0 ${canvasW} ${canvasH}"><defs><marker id="arr" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="#938b7c"/></marker></defs>${edgeLines}</svg>${headers}${nodes}</div>`;
}
function renderSchedule(data){
  schedule.innerHTML=data.schedule.map(s=>{
    const probs=s.action_probs.map(p=>`<span title="${p.node}: ${p.prob}">${p.node.split('_')[0]} ${Math.round(p.prob*100)}%</span>`).join(' / ');
    return `<tr><td>${s.order}</td><td>I${s.task_id} ${s.task_name}<br><span class="k">${s.station}</span></td><td><span class="pill ${s.node_layer}">${s.node_layer}</span></td><td>${s.node_label}</td><td>${Math.round(s.resource_ratio*100)}% (${s.allocated_cpu}GHz)</td><td>${s.start.toFixed(3)} -> <b>${s.finish.toFixed(3)}</b></td><td>${(s.input_transfer+s.dependency_transfer).toFixed(3)} / ${s.wait_time.toFixed(3)} / ${s.exec_time.toFixed(3)}</td><td>${probs}</td></tr>`;
  }).join('');
}
function renderNodes(data){
  nodes.innerHTML=data.nodes.map(n=>`<tr><td>${n.label}</td><td><span class="pill ${n.layer}">${n.layer}</span></td><td>${n.cpu}</td><td>${n.task_count}</td><td>${n.busy_time.toFixed(3)}s</td><td>${Math.round(n.utilization*100)}%</td></tr>`).join('');
}
window.onload=run;
</script>
</body>
</html>"""


def run_algorithm(query: dict) -> dict:
    omega = float(query.get("omega", ["0.55"])[0])
    deadline = float(query.get("deadline", ["3.8"])[0])
    episodes = int(query.get("episodes", ["18"])[0])
    seed = int(query.get("seed", ["42"])[0])
    episodes = max(4, min(episodes, 80))
    scenario = IndustrialScenario(seed=seed)
    scheduler = PaperAlignedScheduler(scenario, deadline=deadline, omega=omega, seed=seed)
    return scheduler.train(episodes=episodes)


class DemoHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(HTML, "text/html; charset=utf-8")
            return
        if parsed.path == "/api/run":
            try:
                data = run_algorithm(parse_qs(parsed.query))
                self._send(json.dumps(data, ensure_ascii=False), "application/json; charset=utf-8")
            except Exception as exc:  # pragma: no cover - visible in browser
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8"))
            return
        self.send_error(404)

    def log_message(self, fmt: str, *args) -> None:
        print("[demo]", fmt % args)

    def _send(self, body: str, content_type: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description="Industrial IoT PER-DDPG scheduling demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="open the browser automatically")
    parser.add_argument("--once", action="store_true", help="print one JSON result and exit")
    args = parser.parse_args()

    if args.once:
        data = run_algorithm({"omega": ["0.55"], "deadline": ["3.8"], "episodes": ["8"], "seed": ["42"]})
        print(json.dumps({
            "makespan": data["makespan"],
            "deadline_hit": data["deadline_hit"],
            "list_seq": data["list_seq"],
            "layer_counts": data["layer_counts"],
        }, ensure_ascii=False, indent=2))
        return

    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    url = f"http://{args.host}:{args.port}"
    print("=" * 72)
    print("Industrial IoT PER-DDPG scheduling demo")
    print(f"Open: {url}")
    print("Paper rules: DAG List_seq + PER replay + LD-Noise + cloud/fog/edge allocation")
    print("=" * 72)
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()

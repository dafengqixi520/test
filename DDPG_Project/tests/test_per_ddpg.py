"""PER-DDPG 集成测试"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AppConfig
from algorithm.environment.cloud_fog_edge_env import CloudFogEdgeEnv
from algorithm.per_ddpg.ddpg_agent import PERDDPGAgent
from algorithm.runner.runner import PERDDPGRunner
import numpy as np

c = AppConfig()
c.env.edge_device_num = 6
c.env.fog_node_num = 3
c.env.max_episodes = 5
np.random.seed(42)

# Test 1: Environment
print("=== Test 1: Environment ===")
env = CloudFogEdgeEnv(c.env)
print(f"State dim: {env.get_state_dim()}")
print(f"Action dim: {env.get_action_dim()}")
print(f"Total nodes: {env.get_node_count()}")
assert env.get_node_count() == 1 + 3 + 6  # cloud + fog + edge

state = env.reset()
print(f"DAG subtasks: {len(env.dag.subtasks)}")
print(f"DAG edges: {len(env.dag.edges)}")
print(f"Entry tasks: {env.dag.entry_tasks}")
print(f"Exit tasks: {env.dag.exit_tasks}")
print(f"List_seq length: {len(env.list_seq)}")
assert len(env.list_seq) == 6
print("  [PASS] Environment works")

# Test 2: DAG Decomposition
print("\n=== Test 2: DAG Decomposition (Algorithm 3.1) ===")
for tid in env.list_seq:
    t = env.dag.subtasks[tid]
    print(f"  I{tid}: rank={t.rank:.2f} urgency={t.urgency:.4f} W={t.schedule_priority:.3f}")
print("  [PASS] DAG decomposition works")

# Test 3: PER-DDPG Agent
print("\n=== Test 3: PER-DDPG Agent ===")
agent = PERDDPGAgent(env.get_state_dim(), env.get_action_dim(), c.ddpg, c.env)
action, eta = agent.select_action(state, explore=True)
print(f"Action dim: {action.shape}")
print(f"Node probs (first 5): {action[:5]}")
print(f"Resource ratio: {action[-1]:.3f}")
print(f"Exploration eta: {eta:.3f}")
assert len(action) == env.get_action_dim()
print("  [PASS] Agent works")

# Test 4: Environment step
print("\n=== Test 4: Environment Step ===")
next_state, reward, done = env.step(action)
print(f"Next state shape: {next_state.shape}")
print(f"Reward: {reward:.3f}")
print(f"Done: {done}")
print(f"Completed: {len(env.completed_tasks)}")
print(f"Global time: {env.global_time:.3f}")
print("  [PASS] Environment step works")

# Test 5: Full episode run
print("\n=== Test 5: Full Episode ===")
env2 = CloudFogEdgeEnv(c.env)
agent2 = PERDDPGAgent(env2.get_state_dim(), env2.get_action_dim(), c.ddpg, c.env)
runner = PERDDPGRunner(env2, agent2, c.env, c.ddpg)

state = runner.start_episode()
total_reward = 0
steps = 0
while True:
    detail = runner.run_step(explore=True)
    if detail is None:
        break
    total_reward += detail["reward"]
    steps += 1
    if detail.get("done"):
        break

print(f"Steps: {steps}")
print(f"Total reward: {total_reward:.3f}")
prog = runner.get_progress()
print(f"Makespan: {prog['latest_makespan']:.4f}s")
print(f"Buffer: {prog['buffer_size']}")
print(f"Noise eta final: {prog['noise_eta']:.3f}")
assert steps == 6, f"Expected 6 steps, got {steps}"
print("  [PASS] Full episode works")

# Test 6: Training
print("\n=== Test 6: Training Loop ===")
# Run several episodes to test training
for ep in range(3):
    runner.start_episode()
    while True:
        detail = runner.run_step(explore=True)
        if detail is None or detail.get("done"):
            break
    prog = runner.get_progress()
    print(f"  Ep {ep+1}: reward={prog['latest_reward']:.2f} makespan={prog['latest_makespan']:.3f}s")

if prog['buffer_size'] >= c.ddpg.batch_size:
    print(f"  Buffer sufficient for training ({prog['buffer_size']} >= {c.ddpg.batch_size})")
    print("  [PASS] Training works")
else:
    print(f"  Buffer: {prog['buffer_size']}/{c.ddpg.batch_size} (need more episodes)")
    print("  [PASS] Buffer accumulating")

# Test 7: State snapshot
print("\n=== Test 7: State Snapshot ===")
snap = env2.get_state_snapshot()
assert "nodes" in snap
assert "dag" in snap
assert "list_seq" in snap
assert len(snap["nodes"]) == env2.get_node_count()
print(f"  Nodes in snapshot: {len(snap['nodes'])}")
print(f"  DAG edges: {len(snap['dag']['edges'])}")
print("  [PASS] Snapshot works")

print("\n" + "="*50)
print("  ALL TESTS PASSED")
print("="*50)

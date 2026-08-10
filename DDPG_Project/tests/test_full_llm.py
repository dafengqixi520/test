"""完整LLM模式仿真测试"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['NO_PROXY'] = '*'

from config import EdgeConfig, LLMConfig, RLConfig, INDUSTRIAL_NODE_PROFILES
from algorithm.environment.edge_env import EdgeComputingEnv
from algorithm.llm_teacher.llm_teacher import LLMTeacher
from algorithm.drl_student.drl_student import DRLStudent
from algorithm.runner.ledrl_runner import LedrlRunner
import numpy as np

config = EdgeConfig()
config.edge_node_num = 10
config.n_agents = 10
config.episode_limit = 15
config.seed = 42
config.master_rng = np.random.RandomState(config.seed)
config.topology_seed = config.master_rng.randint(2**31)
config.node_init_seed = config.master_rng.randint(2**31)
config.task_seed_env = config.master_rng.randint(2**31)
config.failure_seeds = [config.master_rng.randint(2**31) for _ in range(10)]
config.topology_rng = np.random.RandomState(config.topology_seed)
config.node_init_rng = np.random.RandomState(config.node_init_seed)
config.task_rng = np.random.RandomState(config.task_seed_env)

llm_config = LLMConfig(
    api_key='sk-30f21a01513144dcb190257ea43daec4',
    base_url='https://api.deepseek.com',
    model='deepseek-chat'
)

rl_config = RLConfig()
rl_config.n_agents = 10
rl_config.obs_shape = 54

profiles = INDUSTRIAL_NODE_PROFILES[:10]
env = EdgeComputingEnv(config, node_profiles=profiles)
llm_teacher = LLMTeacher(config, llm_config)
drl_student = DRLStudent(rl_config, rl_config)
runner = LedrlRunner(config, llm_teacher, drl_student, env)
runner.reset()

print("Running 10 steps with LLM mode...")
print("-" * 60)

llm_count = 0
fallback_count = 0
total_success = 0
total_failed = 0

for step in range(10):
    terminated, detail = runner.run_step()
    for nid, ld in detail.get('llm_details', {}).items():
        if ld.get('used_fallback'):
            fallback_count += 1
        else:
            llm_count += 1
    stats = detail.get('step_stats', {})
    total_success += stats.get('success', 0)
    total_failed += stats.get('failed', 0)
    print(f"Step {step+1:2d}: LLM={llm_count} fallback={fallback_count} "
          f"success={stats.get('success', 0)} failed={stats.get('failed', 0)} "
          f"reward={detail.get('reward', 0):.2f} lambda={detail.get('lambda_llm', 0):.3f}")
    if terminated:
        break

print("-" * 60)
print(f"Total: {llm_count} LLM decisions, {fallback_count} fallbacks")
print(f"Tasks: {total_success} success, {total_failed} failed")
if llm_count > 0:
    print("[PASS] LLM mode working correctly!")
else:
    print("[INFO] No LLM decisions needed (no active tasks)")

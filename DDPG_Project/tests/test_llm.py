"""测试LLM决策 - DeepSeek API"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['NO_PROXY'] = '*'

# 清除缓存的.pyc文件强制重新加载
import importlib
for mod in list(sys.modules.keys()):
    if 'algorithm' in mod:
        del sys.modules[mod]

from config import EdgeConfig, LLMConfig, INDUSTRIAL_NODE_PROFILES
from algorithm.environment.edge_env import EdgeComputingEnv
from algorithm.llm_teacher.llm_teacher import LLMTeacher
from algorithm.llm_teacher.decision_engine import DecisionEngine
from algorithm.environment.task import Task
import numpy as np

# 先测试 DecisionEngine 单独调用
print("=== 测试1: DecisionEngine 独立调用 ===")
config = EdgeConfig()
config.edge_node_num = 10
config.n_agents = 10
config.seed = 42

llm_config = LLMConfig(
    api_key='sk-30f21a01513144dcb190257ea43daec4',
    base_url='https://api.deepseek.com',
    model='deepseek-chat'
)

de = DecisionEngine(config, llm_config)
print(f"API客户端状态: {de.is_available()}")

# 测试直接API调用
test_prompt = "Say exactly: Forward to node 0"
response_text, dt = de._call_llm(test_prompt, 99)
print(f"响应内容: {repr(response_text)}")
print(f"耗时: {dt:.2f}s")

if response_text:
    action, parsed = de._parse_response(99, response_text, [99, 0])
    print(f"解析结果: action={action}, text={parsed}")
    print("[PASS] DecisionEngine 工作正常!")
else:
    print("[FAIL] API返回空响应")

print()

# 测试完整的 LLMTeacher 调用
print("=== 测试2: LLMTeacher 完整决策 ===")
config.master_rng = np.random.RandomState(config.seed)
config.topology_seed = config.master_rng.randint(2**31)
config.node_init_seed = config.master_rng.randint(2**31)
config.task_seed_env = config.master_rng.randint(2**31)
config.failure_seeds = [config.master_rng.randint(2**31) for _ in range(10)]
config.topology_rng = np.random.RandomState(config.topology_seed)
config.node_init_rng = np.random.RandomState(config.node_init_seed)
config.task_rng = np.random.RandomState(config.task_seed_env)

profiles = INDUSTRIAL_NODE_PROFILES[:10]
env = EdgeComputingEnv(config, node_profiles=profiles)
llm_teacher = LLMTeacher(config, llm_config)
llm_teacher.set_env(env)
env.reset()

task = Task(config, task_id='1_0', node_type='camera')
task.add_to_path(1)
task.arrival_time = 1
env.edge_nodes[1].new_task = task

obs_list = env.get_obs()
avail_actions = env.get_avail_actions()

action, response, used_fallback, prompt = llm_teacher.make_decision(
    1, obs_list[1], avail_actions[1])

action_label = "local" if action == 1 else f"forward to {action}"
print(f"Decision: action={action} ({action_label})")
print(f"Response: {response[:100]}")
print(f"Fallback: {used_fallback}")
print(f"Prompt length: {len(prompt) if prompt else 0}")

if not used_fallback:
    print("[PASS] LLM (DeepSeek) decision succeeded!")
else:
    print("[INFO] Used heuristic fallback")

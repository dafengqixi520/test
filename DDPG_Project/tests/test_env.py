"""端到端集成测试 - 验证环境、LLM Teacher、DRL Student完整运行"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AppConfig, EdgeConfig, LLMConfig, RLConfig, INDUSTRIAL_NODE_PROFILES
from algorithm.environment.edge_env import EdgeComputingEnv
from algorithm.llm_teacher.llm_teacher import LLMTeacher
from algorithm.drl_student.drl_student import DRLStudent
from algorithm.runner.ledrl_runner import LedrlRunner
import numpy as np


def test_environment():
    """测试环境基本运行"""
    print("\n=== 测试1: 环境初始化 ===")
    config = EdgeConfig()
    config.edge_node_num = 10  # 工业物联网场景
    config.episode_limit = 20
    config.topology_type = "star"
    config.seed = 42

    # 初始化随机数种子
    config.master_rng = np.random.RandomState(config.seed)
    config.topology_seed = config.master_rng.randint(2**31)
    config.node_init_seed = config.master_rng.randint(2**31)
    config.task_seed_env = config.master_rng.randint(2**31)
    config.failure_seeds = [config.master_rng.randint(2**31) for _ in range(config.edge_node_num)]
    config.topology_rng = np.random.RandomState(config.topology_seed)
    config.node_init_rng = np.random.RandomState(config.node_init_seed)
    config.task_rng = np.random.RandomState(config.task_seed_env)

    profiles = INDUSTRIAL_NODE_PROFILES[:config.edge_node_num]
    env = EdgeComputingEnv(config, node_profiles=profiles)

    print(f"  节点数: {len(env.edge_nodes)}")
    print(f"  动作空间: {env.n_actions}")
    print(f"  观测维度: {env.get_obs_size()}")

    # 测试reset
    obs = env.reset()
    print(f"  Reset后观测形状: {len(obs)}x{len(obs[0])}")

    # 测试拓扑
    neighbors = [len(env.network_topology.get_neighbors(i)) for i in range(config.edge_node_num)]
    print(f"  各节点邻居数: {neighbors}")

    print("  [PASS] 环境初始化成功")


def test_heuristic_teacher():
    """测试启发式教师决策"""
    print("\n=== 测试2: LLM Teacher (启发式回退模式) ===")
    config = EdgeConfig()
    config.edge_node_num = 10
    config.episode_limit = 20
    config.topology_type = "star"
    config.seed = 42
    config.master_rng = np.random.RandomState(config.seed)
    config.topology_seed = config.master_rng.randint(2**31)
    config.node_init_seed = config.master_rng.randint(2**31)
    config.task_seed_env = config.master_rng.randint(2**31)
    config.failure_seeds = [config.master_rng.randint(2**31) for _ in range(config.edge_node_num)]
    config.topology_rng = np.random.RandomState(config.topology_seed)
    config.node_init_rng = np.random.RandomState(config.node_init_seed)
    config.task_rng = np.random.RandomState(config.task_seed_env)

    llm_config = LLMConfig()
    llm_config.api_key = ""  # 无API密钥，将使用启发式回退
    llm_config.base_url = ""

    profiles = INDUSTRIAL_NODE_PROFILES[:config.edge_node_num]
    env = EdgeComputingEnv(config, node_profiles=profiles)
    llm_teacher = LLMTeacher(config, llm_config)
    llm_teacher.set_env(env)

    obs = env.reset()
    env.edge_nodes[1].new_task = True  # 手动设置节点1有任务

    obs_list = env.get_obs()
    avail_actions = env.get_avail_actions()

    # 测试决策
    action, response, used_fallback, prompt = llm_teacher.make_decision(
        1, obs_list[1], avail_actions[1])

    print(f"  节点1决策: action={action}, response='{response}', fallback={used_fallback}")
    print(f"  是否回退: {used_fallback} (预期: True)")
    assert used_fallback, "期望使用启发式回退"
    print("  [PASS] 启发式决策测试成功")


def test_runner():
    """测试Runner完整运行"""
    print("\n=== 测试3: Runner完整运行(10步) ===")
    config = EdgeConfig()
    config.edge_node_num = 10
    config.episode_limit = 30
    config.topology_type = "star"
    config.seed = 42
    config.master_rng = np.random.RandomState(config.seed)
    config.topology_seed = config.master_rng.randint(2**31)
    config.node_init_seed = config.master_rng.randint(2**31)
    config.task_seed_env = config.master_rng.randint(2**31)
    config.failure_seeds = [config.master_rng.randint(2**31) for _ in range(config.edge_node_num)]
    config.topology_rng = np.random.RandomState(config.topology_seed)
    config.node_init_rng = np.random.RandomState(config.node_init_seed)
    config.task_rng = np.random.RandomState(config.task_seed_env)

    llm_config = LLMConfig()
    llm_config.api_key = ""
    llm_config.base_url = ""

    rl_config = RLConfig()
    rl_config.n_agents = config.edge_node_num
    rl_config.obs_shape = config.edge_node_num * 5

    profiles = INDUSTRIAL_NODE_PROFILES[:config.edge_node_num]
    env = EdgeComputingEnv(config, node_profiles=profiles)

    rl_config.obs_shape = env.get_obs_size()
    rl_config.n_agents = config.edge_node_num

    llm_teacher = LLMTeacher(config, llm_config)
    drl_student = DRLStudent(rl_config, rl_config)
    runner = LedrlRunner(config, llm_teacher, drl_student, env)
    runner.reset()

    total_reward = 0
    total_steps = 0
    success_count = 0
    failure_count = 0
    drop_count = 0

    for step in range(30):
        terminated, detail = runner.run_step()
        total_reward += detail.get("reward", 0)
        total_steps += 1
        stats = detail.get("step_stats", {})
        success_count += stats.get("success", 0)
        failure_count += stats.get("failed", 0)
        drop_count += stats.get("dropped", 0)

        if step % 10 == 0:
            lambda_v = detail.get("lambda_llm", 0)
            print(f"  Step {step+1}: reward={detail.get('reward', 0):.2f}, "
                  f"finished={stats.get('finished', 0)}, "
                  f"success={stats.get('success', 0)}, "
                  f"lambda={lambda_v:.3f}")

        if terminated:
            break

    print(f"\n  总步数: {total_steps}")
    print(f"  任务统计: 成功={success_count}, 失败={failure_count}, 丢弃={drop_count}")
    print(f"  总奖励: {total_reward:.2f}")
    total_done = success_count + failure_count + drop_count
    if total_done > 0:
        print(f"  成功率: {success_count/total_done*100:.1f}%")
    print("  [PASS] Runner完整运行成功")


def test_frontend_data():
    """测试前端数据结构"""
    print("\n=== 测试4: 前端数据结构 ===")
    config = EdgeConfig()
    config.edge_node_num = 10
    config.episode_limit = 5
    config.topology_type = "star"
    config.seed = 42
    config.master_rng = np.random.RandomState(config.seed)
    config.topology_seed = config.master_rng.randint(2**31)
    config.node_init_seed = config.master_rng.randint(2**31)
    config.task_seed_env = config.master_rng.randint(2**31)
    config.failure_seeds = [config.master_rng.randint(2**31) for _ in range(config.edge_node_num)]
    config.topology_rng = np.random.RandomState(config.topology_seed)
    config.node_init_rng = np.random.RandomState(config.node_init_seed)
    config.task_rng = np.random.RandomState(config.task_seed_env)

    profiles = INDUSTRIAL_NODE_PROFILES[:config.edge_node_num]
    env = EdgeComputingEnv(config, node_profiles=profiles)
    env.reset()

    # 测试get_state_snapshot
    snapshot = env.get_state_snapshot()
    print(f"  拓扑节点数: {len(snapshot['topology']['nodes'])}")
    print(f"  拓扑边数: {len(snapshot['topology']['edges'])}")
    print(f"  节点状态数: {len(snapshot['nodes'])}")
    print(f"  活跃任务数: {len(snapshot['tasks'])}")

    # 测试get_topology_data
    topo = env.network_topology.get_topology_data()
    print(f"  拓扑数据: {len(topo['nodes'])} nodes, {len(topo['edges'])} edges")

    # 测试节点state_dict
    for node in env.edge_nodes:
        sd = node.get_state_dict()
        assert "id" in sd
        assert "name" in sd
        assert "type" in sd
        assert "queue_usage" in sd
    print("  [PASS] 前端数据结构测试成功")


if __name__ == '__main__':
    test_environment()
    test_heuristic_teacher()
    test_runner()
    test_frontend_data()
    print("\n" + "="*50)
    print("  所有测试通过！系统可正常运行")
    print("="*50)

"""PER-DDPG训练/评估运行器 — 论文算法3.2主循环"""
import numpy as np
import time
from collections import deque


class PERDDPGRunner:
    """
    PER-DDPG训练运行器

    算法3.2主循环:
    for episode in 1..Max_Episodes:
        1. 生成DAG并运行算法3.1得到List_seq
        2. for t in 1..T:
            a. 选择动作 a_t = μ(s_t) + noise
            b. 执行动作，获得奖励和下一状态
            c. 存储经验到PER缓冲区
            d. 采样训练（满足batch_size后）
            e. 软更新目标网络
    """

    def __init__(self, env, agent, env_config, ddpg_config):
        self.env = env
        self.agent = agent
        self.env_config = env_config
        self.ddpg_config = ddpg_config
        self.device = agent.device

        # 训练统计
        self.episode_rewards = []
        self.episode_makespans = []
        self.recent_rewards = deque(maxlen=100)
        self.train_step = 0
        self.total_episodes = 0

        # 当前episode状态
        self.current_state = None
        self.episode_reward = 0.0
        self.episode_actions = []
        self.episode_done = False

    def start_episode(self):
        """开始新episode"""
        self.current_state = self.env.reset()
        self.episode_reward = 0.0
        self.episode_actions = []
        self.episode_done = False
        return self.current_state

    def run_step(self, explore=True):
        """执行单步决策和训练"""
        if self.episode_done or self.current_state is None:
            return None

        # 保存论文公式3.23对应的决策前状态，供前端审计。
        state_before = self.current_state.copy()

        # 选择动作
        action, eta = self.agent.select_action(self.current_state, explore=explore)

        # 环境执行
        next_state, reward, done = self.env.step(action)

        # 存储经验
        if next_state is not None:
            self.agent.add_experience(
                self.current_state, action, reward, next_state, done)

        # 训练
        train_info = {"critic_loss": 0, "actor_loss": 0, "td_error_mean": 0}
        if explore and len(self.agent.buffer) >= self.ddpg_config.batch_size:
            train_info = self.agent.train()

        # 更新状态
        self.current_state = next_state
        self.episode_reward += reward
        action_record = {
            "action": action.tolist(),
            "reward": float(reward),
            "eta": float(eta),
        }
        if self.env.last_step_info:
            action_record.update(self.env.last_step_info)
        self.episode_actions.append(action_record)
        self.train_step += 1

        if done:
            self.episode_done = True
            self.episode_rewards.append(self.episode_reward)
            self.recent_rewards.append(self.episode_reward)

            # 计算最终makespan
            makespan = max(self.env.task_finish_times.values()) if self.env.task_finish_times else 0
            self.episode_makespans.append(makespan)
            self.total_episodes += 1

        return {
            "step": self.env.current_task_idx,
            "total_steps": self.env.N,
            "reward": float(reward),
            "episode_reward": float(self.episode_reward),
            "done": done,
            "eta": float(eta),
            "train_info": train_info,
            "state": state_before.tolist(),
            "action": action.tolist(),
            "decision": dict(self.env.last_step_info or {}),
            "actions": self.episode_actions,
        }

    def get_progress(self):
        """获取训练进度"""
        recent_avg = np.mean(self.recent_rewards) if self.recent_rewards else 0
        return {
            "total_episodes": self.total_episodes,
            "train_step": self.train_step,
            "recent_avg_reward": float(recent_avg),
            "latest_reward": float(self.episode_rewards[-1]) if self.episode_rewards else 0,
            "latest_makespan": float(self.episode_makespans[-1]) if self.episode_makespans else 0,
            "buffer_size": len(self.agent.buffer),
            "noise_eta": float(self.agent.get_noise_eta()),
        }

    def get_history(self):
        return {
            "rewards": [float(r) for r in self.episode_rewards],
            "makespans": [float(m) for m in self.episode_makespans],
        }

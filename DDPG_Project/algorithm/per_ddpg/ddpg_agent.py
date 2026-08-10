"""PER-DDPG智能体 — 论文算法3.2"""
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import json
import os

from algorithm.per_ddpg.actor_critic import Actor, Critic
from algorithm.per_ddpg.per_buffer import PERBuffer
from algorithm.per_ddpg.noise import LDNoise


MODEL_SCHEMA_VERSION = 2


class PERDDPGAgent:
    """
    融合优先经验回放与自适应探索的DDPG智能体

    核心组件:
    - Actor网络 μ(s) + 目标Actor μ'(s)  (公式3.31-3.33)
    - Critic网络 Q(s,a) + 目标Critic Q'(s,a)  (公式3.34-3.35)
    - PER优先经验回放 (公式3.28-3.30)
    - LD-Noise自适应探索 (公式3.26-3.27)
    - 软更新目标网络 (公式3.36-3.37)
    """

    def __init__(self, state_dim, action_dim, ddpg_config, env_config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = ddpg_config
        self.env_config = env_config
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        # Actor网络及目标网络
        self.actor = Actor(state_dim, action_dim,
                          hidden_dim=ddpg_config.hidden_dim,
                          n_layers=ddpg_config.hidden_layers).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        # Critic网络及目标网络
        self.critic = Critic(state_dim, action_dim,
                            hidden_dim=ddpg_config.hidden_dim,
                            n_layers=ddpg_config.hidden_layers).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        # 优化器
        self.actor_optimizer = th.optim.Adam(
            self.actor.parameters(), lr=ddpg_config.actor_lr)  # α^μ=0.001
        self.critic_optimizer = th.optim.Adam(
            self.critic.parameters(), lr=ddpg_config.critic_lr)  # α^Q=0.002

        # PER缓冲区
        self.buffer = PERBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            capacity=ddpg_config.buffer_size,       # 20000
            alpha=ddpg_config.per_alpha,            # 0.6
            beta_start=ddpg_config.per_beta_start,  # 0.4
            beta_increment=ddpg_config.per_beta_increment,  # 0.001
            epsilon=ddpg_config.per_epsilon,        # 0.01
            batch_size=ddpg_config.batch_size,      # 64
        )

        # LD-Noise探索
        T_train = env_config.max_episodes * env_config.edge_device_num
        self.noise = LDNoise(action_dim,
                            eta_max=ddpg_config.eta_max,  # 0.5
                            T_train=T_train)

        # 超参数
        self.gamma = ddpg_config.gamma          # 折扣因子 0.99
        self.tau = ddpg_config.tau               # 软更新系数 0.005
        self.train_step = 0

    def _normalize_exploratory_action(self, action):
        """将噪声动作恢复为合法概率分布和资源比例。"""
        normalized = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        if normalized.size != self.action_dim:
            raise ValueError(
                f"action length must be {self.action_dim}, got {normalized.size}"
            )
        node_probs = np.clip(normalized[:-1], 0.0, None)
        probability_sum = float(node_probs.sum())
        if probability_sum <= 1e-12:
            node_probs.fill(1.0 / len(node_probs))
        else:
            node_probs /= probability_sum
        normalized[:-1] = node_probs
        normalized[-1] = np.clip(normalized[-1], 0.0, 1.0)
        return normalized

    def select_action(self, state, explore=True):
        """选择动作（训练时加噪声，评估时不加）"""
        state_tensor = th.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with th.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        self.actor.train()

        if explore:
            noise, eta = self.noise.sample()
            action = self._normalize_exploratory_action(action + noise)
            return action, eta
        else:
            self.noise.sample_eval()
            return action, 0.0

    def train(self):
        """单步训练 — 算法3.2第11-16行"""
        batch_data = self.buffer.sample()
        if batch_data is None:
            return {"critic_loss": 0.0, "actor_loss": 0.0}

        states, actions, rewards, next_states, dones, weights, indices = batch_data

        # 转Tensor
        states = th.FloatTensor(states).to(self.device)
        actions = th.FloatTensor(actions).to(self.device)
        rewards = th.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = th.FloatTensor(next_states).to(self.device)
        dones = th.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = th.FloatTensor(weights).unsqueeze(1).to(self.device)

        # --- 更新Critic (公式3.34-3.35) ---
        # 目标Q值: y_i = r_i + γ * Q'(s_{i+1}, μ'(s_{i+1}))
        with th.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            y = rewards + self.gamma * target_q * (1 - dones)

        current_q = self.critic(states, actions)
        td_errors = (y - current_q).detach().cpu().numpy().flatten()
        td_errors = np.nan_to_num(td_errors, nan=0.0, posinf=1e6, neginf=-1e6)

        # 加权MSE损失
        critic_loss = (weights * F.mse_loss(current_q, y, reduction='none')).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        th.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # --- 更新Actor (公式3.31-3.33) ---
        # 最大化 Q(s, μ(s))
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        th.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # --- 更新PER优先级 (算法3.2第14行) ---
        self.buffer.update_priorities(indices, td_errors)

        # --- 软更新目标网络 (公式3.36-3.37) ---
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        self.train_step += 1

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "td_error_mean": float(np.mean(np.abs(td_errors))),
        }

    def _soft_update(self, target, source):
        """软更新: θ' ← τ*θ + (1-τ)*θ'"""
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    def add_experience(self, state, action, reward, next_state, done):
        """存储经验到PER缓冲区"""
        # 估算TD误差用于初始优先级
        state_t = th.FloatTensor(state).unsqueeze(0).to(self.device)
        action_t = th.FloatTensor(action).unsqueeze(0).to(self.device)
        next_state_t = th.FloatTensor(next_state).unsqueeze(0).to(self.device)

        with th.no_grad():
            current_q = self.critic(state_t, action_t)
            next_action = self.actor_target(next_state_t)
            target_q = self.critic_target(next_state_t, next_action)
            td_error = float((reward + self.gamma * target_q * (1 - done) -
                             current_q).cpu().numpy())
            td_error = float(np.nan_to_num(
                td_error, nan=0.0, posinf=1e6, neginf=-1e6
            ))

        self.buffer.add(state, action, reward, next_state, done,
                       td_error=td_error)

    def get_noise_eta(self):
        return self.noise.get_eta()

    def _checkpoint_metadata(self):
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "fog_node_num": self.env_config.fog_node_num,
            "task_num": self.env_config.edge_device_num,
        }

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        th.save(self.actor.state_dict(), f"{path}/actor.pth")
        th.save(self.critic.state_dict(), f"{path}/critic.pth")
        th.save(self.actor_target.state_dict(), f"{path}/actor_target.pth")
        th.save(self.critic_target.state_dict(), f"{path}/critic_target.pth")
        with open(f"{path}/metadata.json", "w", encoding="utf-8") as handle:
            json.dump(self._checkpoint_metadata(), handle, ensure_ascii=False, indent=2)
    def load(self, path):
        metadata_path = f"{path}/metadata.json"
        if not os.path.exists(metadata_path):
            raise ValueError("checkpoint schema metadata is missing; retraining is required")
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        expected = self._checkpoint_metadata()
        for key, expected_value in expected.items():
            if metadata.get(key) != expected_value:
                raise ValueError(
                    f"checkpoint schema mismatch for {key}: "
                    f"expected {expected_value}, got {metadata.get(key)}"
                )

        self.actor.load_state_dict(
            th.load(f"{path}/actor.pth", map_location=self.device))
        self.critic.load_state_dict(
            th.load(f"{path}/critic.pth", map_location=self.device))
        self.actor_target.load_state_dict(
            th.load(f"{path}/actor_target.pth", map_location=self.device))
        self.critic_target.load_state_dict(
            th.load(f"{path}/critic_target.pth", map_location=self.device))

"""LD-Noise 自适应动作探索机制 — 论文公式3.26-3.27"""
import numpy as np


class LDNoise:
    """
    Linear Decay Noise（线性衰减噪声）

    a_t = μ(s_t) + η_t * ε_t,  ε_t ~ N(0, I)  (公式3.26)
    η_t = η_max * (1 - t / T_train)           (公式3.27)

    训练初期探索强（η大），后期逐步减弱到0（纯确定性策略）
    """

    def __init__(self, action_dim, eta_max=0.5, T_train=10000):
        self.action_dim = action_dim
        self.eta_max = eta_max
        self.T_train = T_train  # 总训练步数
        self.t = 0              # 当前步数

    def reset(self):
        self.t = 0

    def get_eta(self):
        """当前探索系数"""
        eta = self.eta_max * (1.0 - min(self.t / self.T_train, 1.0))
        return max(eta, 0.0)

    def sample(self):
        """生成噪声并累加步数"""
        eta = self.get_eta()
        noise = eta * np.random.randn(self.action_dim).astype(np.float32)
        self.t += 1
        return noise, eta

    def sample_eval(self):
        """评估模式：无噪声"""
        self.t += 1
        return np.zeros(self.action_dim, dtype=np.float32), 0.0


class FixedGaussianNoise:
    """Fixed Gaussian exploration used by the standard-DDPG baseline."""

    def __init__(self, action_dim, sigma=0.2):
        self.action_dim = action_dim
        self.sigma = sigma

    def sample(self):
        noise = self.sigma * np.random.randn(self.action_dim).astype(np.float32)
        return noise, self.sigma

    def sample_eval(self):
        return np.zeros(self.action_dim, dtype=np.float32), 0.0

    def get_eta(self):
        return self.sigma

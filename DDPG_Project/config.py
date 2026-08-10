"""PER-DDPG 拓扑依赖型任务协同卸载 — 配置中心"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class EnvConfig:
    """云-雾-边三层计算网络配置（论文表3.1）"""
    # 节点数量
    edge_device_num: int = 12       # N: 边缘设备/子任务数 [4,24]
    fog_node_num: int = 5           # M: 雾节点数 [3,8]
    cloud_node_num: int = 1         # 云节点固定1个

    # 计算能力 (GHz)
    edge_cpu: float = 1.0           # 边缘设备 1GHz
    fog_cpu: float = 5.0            # 雾节点 5GHz
    cloud_cpu: float = 20.0         # 云节点 20GHz

    # 通信参数
    uplink_bandwidth: float = 10e6   # 上行带宽 10MHz
    wan_rate: float = 1e9            # 云雾骨干网 1Gbps
    wan_delay: float = 0.03          # 广域网传播时延 30ms (取中值)
    edge_tx_power: float = 0.4       # 设备发射功率 0.4W
    noise_power: float = 1e-13       # 噪声功率 -100dBm = 1e-13W
    path_loss_exponent: float = 4.0  # 路径损耗指数
    device_fog_distance: tuple = (50, 300)  # 设备到雾节点距离范围 (m)

    # 任务参数
    task_size_range: tuple = (350, 600)     # 任务大小 KB
    task_density_range: tuple = (1000, 2000) # 计算密度 Cycle/MB
    task_deadline: float = 5.0               # 全局截止时间（秒）

    # DAG参数
    dag_edge_prob: float = 0.3     # DAG边生成概率
    weight_factor_omega: float = 0.5  # 优先级/紧迫性权重因子

    # 仿真参数
    max_episodes: int = 500
    steps_per_episode: int = 0  # 0 = 自适应(N = edge_device_num)
    seed: int = 42

    def validate(self):
        if self.edge_device_num < 1:
            raise ValueError("edge_device_num must be at least 1")
        if not 0.0 <= self.dag_edge_prob <= 1.0:
            raise ValueError("dag_edge_prob must be in [0, 1]")


@dataclass
class DDPGConfig:
    """PER-DDPG算法超参数（论文表3.2）"""
    actor_lr: float = 0.001         # Actor学习率
    critic_lr: float = 0.002        # Critic学习率
    gamma: float = 0.99             # 折扣因子
    tau: float = 0.005              # 软更新系数
    buffer_size: int = 20000        # 经验回放池大小
    batch_size: int = 64            # 采样批次大小
    hidden_dim: int = 400           # 隐藏层神经元数
    hidden_layers: int = 2          # 隐藏层数

    # LD-Noise参数
    eta_max: float = 0.5            # 初始探索噪声系数
    # eta_t = eta_max * (1 - t/T_train)  线性衰减到0

    # PER参数
    per_epsilon: float = 0.01       # 防止优先级为0
    # 论文公式3.29-3.30直接使用 P_i=|delta_i|+epsilon。
    # alpha/beta保留为可选扩展，默认关闭额外指数和重要性加权。
    per_alpha: float = 1.0
    per_beta_start: float = 0.0
    per_beta_increment: float = 0.0


@dataclass
class AppConfig:
    """应用配置"""
    env: EnvConfig = field(default_factory=EnvConfig)
    ddpg: DDPGConfig = field(default_factory=DDPGConfig)
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    step_delay: float = 0.05        # 前端展示步延迟
    training_mode: bool = True       # 训练模式/推理模式

"""Actor-Critic网络 — 论文3.3.2节"""
import torch as th
import torch.nn as nn
import torch.nn.functional as F


def fanin_init(size, fanin=None):
    """Xavier初始化"""
    fanin = fanin or size[0]
    v = 1.0 / np.sqrt(fanin)
    return th.Tensor(size).uniform_(-v, v)


class Actor(nn.Module):
    """
    Actor策略网络 μ(s|θ^μ)
    输出: 连续动作 a_t = {A_t, F_t}
    A_t: Softmax概率分布（目标节点选择）
    F_t: Sigmoid压缩到(0,1]（资源分配比例）
    """

    def __init__(self, state_dim, action_dim, hidden_dim=400, n_layers=2):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        layers = []
        in_dim = state_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.fc = nn.Sequential(*layers)

        # 输出层
        self.output = nn.Linear(hidden_dim, action_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.01)

    def forward(self, state):
        x = self.fc(state)
        raw = self.output(x)
        # 前N个输出做Softmax（节点选择概率）
        node_probs = F.softmax(raw[:, :-1], dim=-1)
        # 最后一个输出做Sigmoid（资源分配比例0~1）
        resource_ratio = th.sigmoid(raw[:, -1:])
        return th.cat([node_probs, resource_ratio], dim=-1)


class Critic(nn.Module):
    """
    Critic值网络 Q(s, a|θ^Q)
    评估状态-动作对的Q值
    """

    def __init__(self, state_dim, action_dim, hidden_dim=400, n_layers=2):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim  # 不需要Softmax约束

        # 状态编码
        state_layers = []
        in_dim = state_dim
        for i in range(n_layers - 1):
            state_layers.append(nn.Linear(in_dim, hidden_dim))
            state_layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.state_fc = nn.Sequential(*state_layers)

        # 动作+状态融合层
        self.fc1 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.01)

    def forward(self, state, action):
        s_feat = self.state_fc(state)
        x = th.cat([s_feat, action], dim=-1)
        x = F.relu(self.fc1(x))
        q_value = self.fc2(x)
        return q_value


import numpy as np

"""优先经验回放缓冲区 — 基于SumTree实现（论文3.3.2节）"""
import numpy as np


class SumTree:
    """SumTree数据结构，用于O(log N)优先采样"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # 二叉树数组
        self.data = np.zeros(capacity, dtype=object)  # 存储经验
        self.write_idx = 0  # 当前写入位置
        self.size = 0

    def _propagate(self, idx, change):
        """向上传播优先级变化"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx, priority):
        """更新叶子节点优先级"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority, data):
        """添加新经验"""
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_leaf(self, value):
        """根据随机值采样叶子节点"""
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self.tree):
                leaf_idx = parent
                break
            if value <= self.tree[left]:
                parent = left
            else:
                value -= self.tree[left]
                parent = right
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]


class PERBuffer:
    """
    优先经验回放缓冲区

    样本i的优先级 P_i = |δ_i| + ε (公式3.29)
    采样概率 P(i) = P_i^α / Σ P_k^α (公式3.30)
    重要性采样权重 w_i = (N * P(i))^(-β) / max(w)
    """

    def __init__(self, state_dim, action_dim, capacity=20000,
                 alpha=0.6, beta_start=0.4, beta_increment=0.001,
                 epsilon=0.01, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.alpha = alpha       # 优先级指数
        self.beta = beta_start   # 重要性采样权重指数
        self.beta_increment = beta_increment
        self.epsilon = epsilon    # 防止优先级为0 (公式3.29)
        self.batch_size = batch_size

        self.tree = SumTree(capacity)
        self.max_priority = 1.0   # 新样本初始最大优先级

        # 预分配内存
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.write_idx = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done, td_error=None):
        """添加经验并计算优先级"""
        if td_error is None:
            priority = self.max_priority
        else:
            priority = self._priority(td_error)

        self.states[self.write_idx] = state
        self.actions[self.write_idx] = action
        self.rewards[self.write_idx] = reward
        self.next_states[self.write_idx] = next_state
        self.dones[self.write_idx] = float(done)

        self.tree.add(priority, self.write_idx)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self):
        """按优先级采样一个批次"""
        if self.size < self.batch_size:
            return None

        batch_indices = []
        batch_priorities = []
        batch_data_indices = []

        # 分段采样确保均匀覆盖
        total_priority = self.tree.total_priority
        if not np.isfinite(total_priority) or total_priority <= 0:
            raise RuntimeError("PER SumTree contains invalid total priority")
        segment = total_priority / self.batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = np.random.uniform(a, b)
            leaf_idx, priority, data_idx = self.tree.get_leaf(value)
            batch_indices.append(leaf_idx)
            batch_priorities.append(priority)
            batch_data_indices.append(data_idx)

        # 重要性采样权重
        sampling_probs = np.array(batch_priorities) / total_priority
        weights = (self.size * sampling_probs) ** (-self.beta)
        weights /= weights.max()  # 归一化

        batch = (
            self.states[batch_data_indices],
            self.actions[batch_data_indices],
            self.rewards[batch_data_indices],
            self.next_states[batch_data_indices],
            self.dones[batch_data_indices],
            weights,
            batch_indices,  # 用于后续更新优先级
        )
        return batch

    def update_priorities(self, indices, td_errors):
        """根据新的TD误差更新优先级（算法3.2第14行）"""
        for idx, td_error in zip(indices, td_errors):
            priority = self._priority(td_error)
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def _priority(self, td_error):
        error = float(np.nan_to_num(abs(td_error), nan=0.0, posinf=1e6, neginf=1e6))
        return max(self.epsilon, (error + self.epsilon) ** self.alpha)

    def __len__(self):
        return self.size
